"""Build the ``n x K`` score matrix that multi-PGS stacks, and fold it back down.

Two ways in, one way out.

**In, from PGS Catalog scoring files** — :func:`panel_from_catalog`. The
target genotypes are read *once*: every scoring file is harmonised against the
variant table up front, the union of matched variants is streamed from the
``.bed`` in blocks, and all ``K`` scores accumulate in that single pass. Scoring
``K`` files by calling a single-score routine ``K`` times would re-read the
genotypes ``K`` times, which is the difference between minutes and hours at
biobank scale.

**In, from GWAS summary statistics** — :func:`panel_from_sumstats`. Each trait
is fitted with :func:`ldpred3.run_ldpred3_prs` and scored on the same target.
When ``ld_cache`` is supplied, the LD reference is built once and cached on
disk; target metadata is indexed once and shared by all fit-only traits; then
all columns are scored in one dosage pass. This is the arm of the paper that
turns public GWAS for other traits into scores; it is also how the target
trait's own score is produced.

**Out** — :func:`combine_weights` takes a panel and a fitted
:class:`~multipgs.stack.MultiPGSFit` and collapses ``K`` weight sets and their
stacking coefficients into **one** per-variant weight table. That file is the
deployable artefact: a new cohort is scored from it directly, with no reference
to the ``K`` inputs, and no need to reproduce the panel.

Scales are tracked, not assumed. Catalog weights count alleles
(``sum_j w_j g_ij``); LDpred3 weights are defined on standardized genotypes
(``(g - 2f)/sd``). A panel records which convention each score used, along with
the target cohort's per-variant allele frequency and dosage SD, and
:func:`combine_weights` puts everything on the standardized scale — the one
:func:`ldpred3.score_from_weights` applies, so the combined file it writes can
be handed straight back to ldpred3 to score a new cohort.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ._validate import _positive_integer


__all__ = ["ScorePanel", "panel_from_catalog", "panel_from_sumstats",
           "panel_from_weights", "combine_weights", "check_weights",
           "read_panel", "write_panel", "save_panel", "load_panel",
           "attach_metadata", "read_trait_table"]

_PANEL_FORMAT = 1

# Genotype elements held in memory per streamed block. 8e6 float64 is 64 MB.
_BLOCK_ELEMS = 8_000_000

# Fraction of a block's (variant x active score) cells that must carry a weight
# before one dense product beats per-score column gathers. See
# :func:`_accumulate_block`; measured crossover is near 0.005 on this machine,
# and the penalty for using the dense product just below it is about 20%.
_GEMM_MIN_DENSITY = 1.0 / 128.0


class _SharedVariantTable(Mapping):
    """Pickleable read-only mapping of shared Catalog variant metadata."""

    __slots__ = ("_arrays",)

    def __init__(self, arrays):
        self._arrays = dict(arrays)
        for values in self._arrays.values():
            values.setflags(write=False)

    def __getitem__(self, key):
        return self._arrays[key]

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self):
        return len(self._arrays)

    def __reduce__(self):
        # Re-enter __init__ so pickle/deepcopy restore the read-only flags.
        return type(self), (self._arrays,)


class _CompactWeightTable(Mapping):
    """Dict-like score weights backed by one shared variant table.

    Catalog scores usually overlap heavily.  Repeating seven metadata arrays
    for every score can therefore cost more memory than the score weights
    themselves.  This mapping stores only ``index`` and ``weight`` per score;
    the usual ``id/chrom/pos/a1/a2/af/sd`` values are selected lazily from the
    panel-wide variant union.  Its public keys match the legacy dictionaries,
    so inspection and ``dict(table)`` continue to work.
    """

    _KEYS = ("id", "chrom", "pos", "a1", "a2", "weight", "af", "sd")
    __slots__ = ("_variant_table", "_index", "_weight")

    def __init__(self, variant_table, index, weight):
        self._variant_table = variant_table
        self._index = np.asarray(index)
        self._weight = np.asarray(weight, dtype=float)
        if self._index.ndim != 1 or self._weight.ndim != 1:
            raise ValueError("compact weight index and weight must be 1-D")
        if self._index.size != self._weight.size:
            raise ValueError("compact weight index and weight lengths differ")

    @property
    def variant_table(self):
        """The shared union metadata (read-only NumPy arrays)."""
        return self._variant_table

    @property
    def index(self):
        """Positions of this score's variants in :attr:`variant_table`."""
        return self._index

    def __getitem__(self, key):
        if key == "weight":
            return self._weight
        if key in self._variant_table:
            return self._variant_table[key][self._index]
        raise KeyError(key)

    def __iter__(self):
        return iter(self._KEYS)

    def __len__(self):
        return len(self._KEYS)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class ScorePanel:
    """Scores for ``n`` individuals across ``K`` polygenic scores.

    ``weights`` holds one mapping per score with keys
    ``id chrom pos a1 a2 weight`` and optionally ``sd``/``af``, where ``a1`` is
    the allele the weight counts. Catalog panels use the same mapping interface
    but keep variant metadata once in a shared union and store only an index
    plus weights per score. ``standardized[k]`` says whether score ``k``'s
    weights apply to standardized genotypes (LDpred3) or raw allele counts
    (PGS Catalog).
    """

    scores: np.ndarray
    sample_fid: np.ndarray
    sample_iid: np.ndarray
    score_ids: np.ndarray
    standardized: np.ndarray
    weights: list = field(default_factory=list)
    meta: list = field(default_factory=list)
    log: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score_ids = np.asarray(self.score_ids, dtype=object).ravel()
        _require_unique_score_ids(self.score_ids)

    def __len__(self):
        return int(self.scores.shape[0])

    @property
    def n_scores(self):
        return int(self.scores.shape[1])

    def index_of(self, score_id):
        """Column index of ``score_id``.

        Ids are compared after ``str(...)``, matching :meth:`select`. A
        duplicate id is an error, not the first hit.
        """
        key = str(score_id)
        hits = [i for i, sid in enumerate(self.score_ids) if str(sid) == key]
        if not hits:
            raise KeyError(f"no score {score_id!r} in this panel")
        if len(hits) > 1:
            raise ValueError(f"score id {key!r} is not unique")
        return int(hits[0])

    def select(self, columns):
        """A panel restricted to ``columns`` (indices, ids, or a bool mask)."""
        from .stack import _resolve_columns
        idx = _resolve_columns(columns, np.asarray(self.score_ids, dtype=object),
                               self.n_scores)
        # Integer scalars otherwise collapse the score axis (``scores[:, 1]``),
        # producing something that is no longer a panel.
        idx = np.atleast_1d(idx).astype(int, copy=False)
        return ScorePanel(
            scores=self.scores[:, idx],
            sample_fid=self.sample_fid, sample_iid=self.sample_iid,
            score_ids=np.asarray(self.score_ids, dtype=object)[idx],
            standardized=np.asarray(self.standardized)[idx],
            weights=[self.weights[i] for i in idx] if self.weights else [],
            meta=[self.meta[i] for i in idx] if self.meta else [],
            log=dict(self.log, selected_from=self.n_scores))

    def align(self, other):
        """Reorder ``other``'s rows to this panel's individuals.

        Panels built from different targets — a training cohort and a test
        cohort, say — are matched on ``FID:IID``. Individuals missing from
        either side are dropped from both, so the two panels come back with the
        same rows in the same order.
        """
        mine = _keys(self.sample_fid, self.sample_iid)
        theirs = _keys(other.sample_fid, other.sample_iid)
        _require_unique_keys(mine, "this panel")
        _require_unique_keys(theirs, "the other panel")
        lookup = {k: i for i, k in enumerate(theirs)}
        keep_mine, keep_theirs = [], []
        for i, k in enumerate(mine):
            j = lookup.get(k)
            if j is not None:
                keep_mine.append(i)
                keep_theirs.append(j)
        if not keep_mine:
            raise ValueError("the two panels share no individuals (matched on "
                             "FID:IID)")
        return self._rows(keep_mine), other._rows(keep_theirs)

    def _rows(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        return ScorePanel(
            scores=self.scores[idx], sample_fid=np.asarray(self.sample_fid)[idx],
            sample_iid=np.asarray(self.sample_iid)[idx],
            score_ids=self.score_ids, standardized=self.standardized,
            weights=self.weights, meta=self.meta, log=self.log)

    def summary(self):
        n_dead = (int(np.sum(np.std(self.scores, axis=0) <= 0))
                  if len(self) else 0)
        lines = [f"panel: {len(self)} individuals x {self.n_scores} scores"]
        matched = [m.get("n_matched") for m in self.meta
                   if isinstance(m, dict) and m.get("n_matched") is not None]
        if matched:
            lines.append(f"  variants matched per score: median "
                         f"{int(np.median(matched))}, min {min(matched)}, "
                         f"max {max(matched)}")
        mass = [m.get("weight_mass_matched") for m in self.meta
                if isinstance(m, dict)
                and m.get("weight_mass_matched") is not None]
        if mass:
            lines.append(f"  weight mass matched: median "
                         f"{np.median(mass):.3f}, min {min(mass):.3f}")
        if n_dead:
            lines.append(f"  {n_dead} score(s) are constant here and cannot "
                         f"enter a fit")
        h2 = [m.get("inference", {}).get("h2_est") for m in self.meta
              if isinstance(m, dict) and isinstance(m.get("inference"), dict)
              and m.get("inference", {}).get("h2_est") is not None]
        if h2:
            vals = np.asarray(h2, dtype=float)
            lines.append(f"  inferred h2: median {np.nanmedian(vals):.3f}, "
                         f"min {np.nanmin(vals):.3f}")
        n_eff = [m.get("n_eff") for m in self.meta
                 if isinstance(m, dict) and m.get("n_eff") is not None
                 and np.isfinite(m.get("n_eff", np.nan))]
        if n_eff:
            lines.append(f"  n_eff known for {len(n_eff)} of {self.n_scores} "
                         "scores")
        for key in ("n_failed", "n_skipped"):
            if self.log.get(key):
                lines.append(f"  {key.replace('_', ' ')}: {self.log[key]}")
        return "\n".join(lines)

    def concat(self, other):
        """Place ``other``'s scores beside this panel's, on shared individuals.

        Individuals are matched on ``FID:IID``. Score ids must be unique across
        the two panels. Scale flags, weights and metadata travel with their
        columns.
        """
        left, right = self.align(other)
        left_ids = [str(s) for s in np.asarray(left.score_ids, dtype=object)]
        right_ids = [str(s) for s in np.asarray(right.score_ids, dtype=object)]
        clash = sorted(set(left_ids) & set(right_ids))
        if clash:
            raise ValueError("concat would collide on score id(s): "
                             + ", ".join(clash[:5]))
        for side, panel in (("left", left), ("right", right)):
            if panel.weights and len(panel.weights) != panel.n_scores:
                raise ValueError(
                    f"{side} panel has {len(panel.weights)} weight tables for "
                    f"{panel.n_scores} score columns")
            if panel.meta and len(panel.meta) != panel.n_scores:
                raise ValueError(
                    f"{side} panel has {len(panel.meta)} metadata entries for "
                    f"{panel.n_scores} score columns")
        if bool(left.weights) != bool(right.weights):
            raise ValueError(
                "cannot concat a panel carrying per-score weights with one "
                "that carries none; column-to-weight identity would be lost")

        # Empty metadata means "unknown for every column". Materialise those
        # unknowns only when the other panel has metadata, so positional
        # consumers can never attach the right panel's first record to a left
        # panel score.
        left_meta = (list(left.meta) if left.meta
                     else ([{} for _ in range(left.n_scores)]
                           if right.meta else []))
        right_meta = (list(right.meta) if right.meta
                      else ([{} for _ in range(right.n_scores)]
                            if left.meta else []))
        return ScorePanel(
            scores=np.hstack([left.scores, right.scores]),
            sample_fid=left.sample_fid, sample_iid=left.sample_iid,
            score_ids=np.array(left_ids + right_ids, dtype=object),
            standardized=np.concatenate([
                np.asarray(left.standardized, dtype=bool),
                np.asarray(right.standardized, dtype=bool)]),
            weights=(list(left.weights) + list(right.weights)
                     if left.weights or right.weights else []),
            meta=left_meta + right_meta,
            log={"source": "concat",
                 "n_scores": left.n_scores + right.n_scores,
                 "left": left.n_scores, "right": right.n_scores,
                 "n_individuals": len(left)})

    def save(self, path):
        """Write this panel, including per-variant weights, to ``path`` (.npz)."""
        return save_panel(self, path)


def _keys(fid, iid):
    fid = np.asarray(fid).astype(str).ravel()
    iid = np.asarray(iid).astype(str).ravel()
    if fid.shape != iid.shape:
        raise ValueError("sample_fid and sample_iid must have the same length")
    # Tuples avoid delimiter collisions such as ("a:b", "c") versus
    # ("a", "b:c").
    return list(zip(fid, iid))


def _require_unique_keys(keys, name):
    seen = set()
    duplicate = None
    for key in keys:
        if key in seen:
            duplicate = key
            break
        seen.add(key)
    if duplicate is not None:
        raise ValueError(
            f"{name} contains duplicate FID:IID {duplicate[0]}:{duplicate[1]}; "
            "alignment would be ambiguous")


def _require_unique_score_ids(ids, *, what="score id", sources=None):
    """Reject duplicate stringified score identifiers."""
    first = {}
    for i, sid in enumerate(np.asarray(ids, dtype=object).ravel()):
        key = str(sid)
        if key in first:
            extra = ""
            if sources is not None:
                extra = f" ({sources[first[key]]} and {sources[i]})"
            raise ValueError(f"duplicate {what} {key!r}{extra}")
        first[key] = i


# ---------------------------------------------------------------------------
# From PGS Catalog scoring files
# ---------------------------------------------------------------------------

def panel_from_catalog(paths, plink, *, sample_path=None, drop_ambiguous=True,
                       standardize=False, min_matched=1, on_error="raise",
                       block=None, prefer_harmonized=True, progress=None,
                       metadata=None):
    """Score a directory or list of PGS Catalog files against one target.

    Parameters
    ----------
    paths : str or sequence of str
        Scoring files, or a directory of ``*.txt``/``*.tsv`` (gzipped forms
        included). ``*.weights`` files in the same directory are ignored.
    plink : str
        PLINK prefix, or a ``.bgen`` path. Both formats are streamed over the
        union of matched variants in memory-bounded blocks.
    drop_ambiguous : bool
        Drop palindromic A/T and C/G variants, which cannot be strand-resolved
        from alleles alone.
    standardize : bool
        ``False`` (default) applies weights to raw allele dosages, which is the
        PGS Catalog convention. ``True`` z-scores each genotype column against
        *this* cohort's frequencies first — do that only if you know the weights
        were defined that way.
    min_matched : int
        Drop a score that matches fewer than this many target variants; a score
        resting on three variants is noise in a stack.
    on_error : {"raise", "skip"}
        What to do when a file fails to parse, cannot be aligned, or matches
        nothing.
    block : int, optional
        Variants per streamed block. The default keeps a block near 64 MB.
    progress : callable, optional
        Called as ``progress(i, n, score_id)`` before each file is harmonised.
    metadata : str, optional
        Catalog ``metadata.tsv`` from :func:`multipgs.fetch.write_score_metadata`.
        When omitted, a ``metadata.tsv`` sitting next to the scoring files is
        picked up automatically.

    Returns
    -------
    ScorePanel
    """
    from ldpred3.genotype_io import read_bed, read_bim, read_fam, strip_ext

    from .catalog import harmonize_scoring_file, read_scoring_file

    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    files = _expand_paths(paths, _CATALOG_SUFFIXES)
    if not files:
        raise ValueError(f"no scoring files found in {paths!r}")

    is_bgen = str(plink).lower().endswith(".bgen")
    if is_bgen:
        from ldpred3.interop import prepare_target
        target = prepare_target(plink, sample_path=sample_path)
        variants = target.variants
        fid, iid = target.samples.fid, target.samples.iid
        n_samples, n_total = len(fid), len(variants.id)
    else:
        prefix = strip_ext(str(plink))
        variants = read_bim(prefix + ".bim")
        samples = read_fam(prefix + ".fam")
        fid, iid = samples.fid, samples.iid
        n_samples, n_total = len(fid), len(variants.id)

    score_ids, metas, per_score = [], [], []
    n_failed = n_skipped = 0
    for i, path in enumerate(files):
        try:
            sf = read_scoring_file(path, prefer_harmonized=prefer_harmonized)
        except Exception:
            if on_error == "raise":
                raise
            n_failed += 1
            continue
        if progress is not None:
            progress(i, len(files), sf.pgs_id)
        try:
            var_index, w, log = harmonize_scoring_file(
                sf, variants, drop_ambiguous=drop_ambiguous)
        except Exception:
            if on_error == "raise":
                raise
            n_failed += 1
            continue
        if var_index.size < min_matched:
            msg = (f"{sf.pgs_id}: only {var_index.size} of {len(sf)} variants "
                   f"matched the target (min_matched={min_matched})")
            if on_error == "raise":
                raise ValueError(msg + ". Check that the scoring file and the "
                                 "genotypes are on the same genome build.")
            n_skipped += 1
            continue
        score_ids.append(sf.pgs_id)
        meta = dict(sf.meta)
        meta.update(log)
        meta["path"] = str(path)
        metas.append(meta)
        per_score.append((var_index, w))

    _require_unique_score_ids(
        score_ids, sources=[m["path"] for m in metas])

    if not per_score:
        raise ValueError(f"none of the {len(files)} scoring files produced a "
                         f"usable score ({n_failed} unreadable, {n_skipped} "
                         f"below min_matched)")

    scores, union, af, sd = _accumulate_scores(
        per_score, n_samples, n_total, plink, standardize=standardize,
        block=block, read_bed=read_bed, strip_ext=strip_ext, is_bgen=is_bgen,
        sample_path=sample_path)

    # Store target metadata once for the union, not once per score.  At PGS
    # Catalog scale those repeated object/numeric arrays otherwise dominate
    # memory. combine_weights needs AF/SD to put allele-count weights onto the
    # standardized scale used by ldpred3's scorer.
    variant_arrays = {
        "id": np.asarray(variants.id)[union],
        "chrom": np.asarray(variants.chrom)[union],
        "pos": np.asarray(variants.pos)[union],
        "a1": np.asarray(variants.a1)[union],
        "a2": np.asarray(variants.a2)[union],
        "af": af,
        "sd": sd,
    }
    variant_table = _SharedVariantTable(variant_arrays)
    index_dtype = _smallest_index_dtype(union.size)
    weight_tables = []
    for var_index, w in per_score:
        at = np.searchsorted(union, var_index).astype(index_dtype, copy=False)
        at.setflags(write=False)
        weight_tables.append(_CompactWeightTable(variant_table, at, w))

    K = len(per_score)
    panel = ScorePanel(
        scores=scores, sample_fid=np.asarray(fid), sample_iid=np.asarray(iid),
        score_ids=np.array(score_ids, dtype=object),
        standardized=np.full(K, bool(standardize)),
        weights=weight_tables, meta=metas,
        log={"source": "pgs_catalog", "n_files": len(files), "n_scores": K,
             "n_failed": n_failed, "n_skipped": n_skipped,
             "min_matched": int(min_matched),
             "target": str(plink), "standardize": bool(standardize),
             "drop_ambiguous": bool(drop_ambiguous)})
    sidecar = metadata if metadata is not None else _sidecar_metadata(files)
    if sidecar is not None:
        attach_metadata(panel, sidecar)
        panel.log["metadata"] = str(sidecar)
    return panel


def _smallest_index_dtype(size):
    """Smallest unsigned dtype able to index an array of ``size`` entries."""
    largest = max(int(size) - 1, 0)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if largest <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("variant union is too large to index")


_CATALOG_SUFFIXES = (".txt", ".txt.gz", ".tsv", ".tsv.gz")
_WEIGHT_SUFFIXES = (".weights",)
_SUMSTAT_SUFFIXES = (".tsv", ".tsv.gz", ".txt", ".txt.gz",
                     ".sumstats", ".sumstats.gz")


def _expand_paths(paths, suffixes):
    """One path, a directory, or a mix of both, to a flat list of files.

    Explicit files are kept regardless of suffix. Directory listings keep only
    ``suffixes``, so a mixed work directory is not scanned as every route.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    skip = {"metadata.tsv", "n_eff.tsv"}
    out = []
    for entry in paths:
        p = str(entry)
        if os.path.isdir(p):
            out.extend(sorted(
                os.path.join(p, f) for f in os.listdir(p)
                if f.endswith(suffixes) and f not in skip
                and not f.startswith(".")))
        else:
            out.append(p)
    return out


def _variant_union(per_score, n_total):
    """Sorted variant union with workspace bounded by the target width."""
    # Concatenating K near-genome-wide index arrays would need O(Km) temporary
    # memory (900 x 1M int64 entries is 7.2 GB before ``unique``). One Boolean
    # byte per target variant plus the final sorted index vector is enough.
    present = np.zeros(n_total, dtype=bool)
    for var_index, _ in per_score:
        present[var_index] = True
    return np.flatnonzero(present)


def _accumulate_scores(per_score, n_samples, n_total, plink, *, standardize,
                       block, read_bed, strip_ext, is_bgen,
                       frozen=None, sample_path=None):
    """One pass over the genotypes: ``K`` scores, plus per-variant AF and SD."""
    K = len(per_score)
    union = _variant_union(per_score, n_total)
    m = union.size
    # Per score: positions into `union`, ascending, with matching weights.
    mapped, frozen_ord = [], None if frozen is None else []
    for i, (var_index, w) in enumerate(per_score):
        pos = np.searchsorted(union, var_index)
        order = np.argsort(pos, kind="mergesort")
        mapped.append((pos[order], np.asarray(w)[order]))
        if frozen is not None:
            item = frozen[i]
            if item is None:
                frozen_ord.append(None)
            else:
                fa, fs = item
                frozen_ord.append((np.asarray(fa, dtype=float)[order],
                                   np.asarray(fs, dtype=float)[order]))

    scores = np.zeros((n_samples, K))
    af = np.zeros(m)
    sd = np.zeros(m)
    if block is None:
        # The 16-variant floor keeps per-block overhead bounded on small
        # targets while honouring the _BLOCK_ELEMS budget at biobank n: a
        # larger floor (n x 64) would hold 256 MB of dosage at n = 500k.
        block = int(np.clip(_BLOCK_ELEMS // max(n_samples, 1), 16, 8192))
    block = max(1, int(block))

    if is_bgen:
        from ldpred3.interop import iter_bgen_dosage

        def dosage_blocks():
            consumed = 0
            for item in iter_bgen_dosage(
                    plink, union, sample_path=sample_path, chunk=block):
                stop = consumed + item.source_index.size
                if not np.array_equal(item.source_index,
                                      union[consumed:stop]):
                    raise RuntimeError(
                        "BGEN dosage stream lost a selected variant")
                yield consumed, stop, item.dosage
                consumed = stop
            if consumed != m:
                raise RuntimeError(
                    "BGEN dosage stream ended before all selected variants")
    else:
        def dosage_blocks():
            for start in range(0, m, block):
                stop = min(start + block, m)
                raw = read_bed(
                    strip_ext(str(plink)) + ".bed", n_samples, n_total,
                    variant_idx=union[start:stop])
                yield start, stop, raw

    for start, stop, raw in dosage_blocks():
        d = np.asarray(raw, dtype=float)
        miss = ~np.isfinite(d) if is_bgen else d < 0
        # A scale-free LDpred3 file means target standardisation, not raw
        # dosage. Keep the all-target case on the shared BLAS path; mixed or
        # frozen files need their own centres and are accumulated per score.
        all_target = (frozen_ord is not None
                      and all(item is None for item in frozen_ord))
        use_target = standardize or all_target
        block_matrix, mean_b, sd_b = _prepare_block(d, miss, use_target)
        af[start:stop] = mean_b / 2.0
        sd[start:stop] = sd_b
        if frozen_ord is None or all_target:
            _accumulate_block(scores, block_matrix, mapped, start, stop)
        else:
            _accumulate_frozen(scores, block_matrix, miss, mapped, frozen_ord,
                               mean_b, sd_b, start, stop)
    return scores, union, af, sd


def _accumulate_frozen(scores, dosage, miss, mapped, frozen, target_mean,
                       target_sd, start, stop):
    """Apply each score's target or frozen standardisation to one block.

    ``dosage`` has already been target-mean imputed for computing target
    moments. Frozen scores must not retain that imputation: a missing call is
    imputed to its frozen reference mean and therefore contributes exactly
    zero after centring, as in :func:`ldpred3.score_from_weights`.
    """
    for s, (pos, w) in enumerate(mapped):
        lo = int(np.searchsorted(pos, start))
        hi = int(np.searchsorted(pos, stop))
        if hi <= lo:
            continue
        local = pos[lo:hi] - start
        g = dosage[:, local]
        scale = frozen[s]
        if scale is None:
            mean = target_mean[local]
            fs = target_sd[local]
            z = g - mean
        else:
            fa, fs = scale[0][lo:hi], scale[1][lo:hi]
            z = g - 2.0 * fa
            local_miss = miss[:, local]
            if local_miss.any():
                # Frozen-mean imputation followed by centring is exactly zero.
                z[local_miss] = 0.0
        good = fs > 1e-12
        if np.any(good):
            z[:, good] /= fs[good]
        z[:, ~good] = 0.0
        scores[:, s] += z @ w[lo:hi]


def _accumulate_block(scores, block_matrix, mapped, start, stop):
    """Add one genotype block's contribution to every score that touches it.

    Two ways to do the same sum, chosen by how densely the scores populate this
    block. Scoring each score separately gathers its columns out of
    ``block_matrix`` — and a fancy index *copies*, so the work is memory-bound
    and repeated ``K`` times. Scattering the same weights into one
    ``block x active`` matrix instead lets a single BLAS-3 call do all of them,
    which is what :func:`multipgs.sumstat.score_gram` already does per LD block.

    The dense product does more arithmetic (every variant against every active
    score) and wins anyway once the block is populated enough for BLAS-3
    throughput to beat the gathers: measured crossover is near a density of
    0.005, rising to 2x by 0.008 and 6-8x by 0.06. Below
    :data:`_GEMM_MIN_DENSITY` the gather is kept, where it is the cheaper of
    the two.
    """
    active, entries = [], 0
    for s, (pos, w) in enumerate(mapped):
        lo = int(np.searchsorted(pos, start))
        hi = int(np.searchsorted(pos, stop))
        if hi > lo:
            active.append((s, lo, hi))
            entries += hi - lo
    if not active:
        return
    width = stop - start
    if entries >= _GEMM_MIN_DENSITY * width * len(active):
        block_w = np.zeros((width, len(active)))
        for local, (s, lo, hi) in enumerate(active):
            pos, w = mapped[s]
            block_w[pos[lo:hi] - start, local] = w[lo:hi]
        contribution = block_matrix @ block_w
        for local, (s, _, _) in enumerate(active):
            scores[:, s] += contribution[:, local]
        return
    for s, lo, hi in active:
        pos, w = mapped[s]
        scores[:, s] += block_matrix[:, pos[lo:hi] - start] @ w[lo:hi]


def _prepare_block(d, miss, standardize):
    """Mean-impute missing calls, and z-score the columns if asked.

    Mean imputation is the standard PRS treatment of a missing call: it
    contributes the cohort-average dosage, so a variant missing in one
    individual neither adds to nor subtracts from their score relative to the
    others.

    Returns ``(matrix, mean, sd)`` — the mean and SD of the *imputed* dosage,
    which is what a weight on the allele-count scale has to be rescaled by to
    reach the standardized scale.
    """
    if miss.any():
        d = d.copy()
        # Sum only observed dosages without ``nanmean``: an all-missing column
        # is valid (it contributes zero after standardisation) and should not
        # emit a RuntimeWarning. Reuse the caller-supplied missingness mask and
        # avoid another n x m temporary.
        observed = d.shape[0] - miss.sum(axis=0)
        d[miss] = 0.0
        col_mean = np.divide(
            d.sum(axis=0), observed, out=np.zeros(d.shape[1], dtype=float),
            where=observed > 0)
        np.copyto(d, col_mean[None, :], where=miss)
    mean = d.mean(axis=0)
    sd = d.std(axis=0)
    if not standardize:
        return d, mean, sd
    out = d - mean
    good = sd > 1e-12
    out[:, good] /= sd[good]
    out[:, ~good] = 0.0
    return out, mean, sd


def _score_ldpred3_fits_once(results, plink, *, sample_path=None, block=None,
                             prepared_target=None):
    """Score several fit-only LDpred3 results in one target-dosage pass.

    ``beta_adjusted`` is defined on target-standardized genotypes.  The shared
    scorer forms every column with the same mean-imputation and population-SD
    convention as LDpred3, while retaining the target AF/SD needed to freeze
    each resulting weight file.
    """
    from ldpred3.genotype_io import read_bed, read_bim, read_fam, strip_ext

    is_bgen = str(plink).lower().endswith(".bgen")
    if is_bgen:
        from ldpred3.interop import PreparedTarget, prepare_target
        if prepared_target is None:
            prepared_target = prepare_target(plink, sample_path=sample_path)
        if (not isinstance(prepared_target, PreparedTarget)
                or not prepared_target.matches(
                    plink, sample_path=sample_path)):
            raise ValueError(
                "prepared_target does not match the BGEN scoring target")
        fid, iid = prepared_target.samples.fid, prepared_target.samples.iid
        n_samples, n_total = len(fid), prepared_target.n_total
    else:
        prefix = strip_ext(str(plink))
        variants = read_bim(prefix + ".bim")
        samples = read_fam(prefix + ".fam")
        fid, iid = samples.fid, samples.iid
        n_samples, n_total = len(fid), len(variants)

    first = results[0]
    if (not np.array_equal(np.asarray(fid), np.asarray(first.sample_fid))
            or not np.array_equal(np.asarray(iid), np.asarray(first.sample_iid))):
        raise RuntimeError(
            "the target sample order changed between fitting and batch scoring")

    per_score = []
    for result in results:
        index = np.asarray(result.var_index)
        weight = np.asarray(result.beta_adjusted, dtype=float)
        if (index.ndim != 1 or not np.issubdtype(index.dtype, np.integer)
                or index.size != weight.size):
            raise ValueError(
                "an LDpred3 fit returned inconsistent variant indices/weights")
        index = index.astype(np.int64, copy=False)
        if (index.size == 0 or np.any(index < 0) or np.any(index >= n_total)
                or np.unique(index).size != index.size):
            raise ValueError(
                "an LDpred3 fit returned empty, duplicate, or out-of-range "
                "target variant indices")
        per_score.append((index, weight))

    scores, union, af, sd = _accumulate_scores(
        per_score, n_samples, n_total, plink, standardize=True, block=block,
        read_bed=read_bed, strip_ext=strip_ext, is_bgen=is_bgen,
        sample_path=sample_path)
    scales = []
    for index, _ in per_score:
        at = np.searchsorted(union, index)
        if np.any(at >= union.size) or not np.array_equal(union[at], index):
            raise RuntimeError("batch-scoring union lost a fitted variant")
        scales.append((af[at].copy(), sd[at].copy()))
    return scores, scales


# ---------------------------------------------------------------------------
# From GWAS summary statistics, via ldpred3
# ---------------------------------------------------------------------------

def panel_from_sumstats(sumstats, plink, *, score_ids=None, ld_cache=None,
                        ld_prefix=None, on_error="raise", progress=None,
                        n_jobs=1, weights_dir=None, preflight=False,
                        traits=None, **ldpred3_kwargs):
    """Fit one LDpred3 model per GWAS and score them all on one target.

    Parameters
    ----------
    sumstats : sequence of str, or mapping
        Summary-statistic files. A mapping's keys become the score ids.
        Ignored when ``traits`` is given.
    plink : str
        Target genotypes (PLINK prefix or ``.bgen``).
    ld_cache : str, optional
        Path for a reference-wide LD cache. It is **built by the first trait and
        reused by the rest**, which is the whole reason to fit a panel in one
        call rather than in a loop. Traits may cover different principal
        subsets; each must be covered by the cache's allele-compatible variant
        superset.
    ld_prefix : str, optional
        External LD-reference PLINK prefix, forwarded to
        :func:`ldpred3.run_ldpred3_prs`. On its own, the reference is read and
        LD is rebuilt for every trait. Pair it with a writable ``ld_cache`` to
        build the blocks once and reuse them.
    on_error : {"raise", "skip"}
    progress : callable, optional
        ``progress(i, n, score_id)`` before each fit.
    n_jobs : int, default 1
        Independent traits after the LD cache exists. The first successful fit
        always runs alone so it can write ``ld_cache``; the remainder run in a
        thread pool of this size. ``1`` is sequential. Parallel fitting from
        ``ld_prefix`` requires ``ld_cache`` so workers do not independently
        rebuild the same reference. The cache payload is fully validated once,
        target metadata is prepared once, and all fitted columns are scored
        together in one target-dosage pass.
    weights_dir : str, optional
        Write each trait's ldpred3 weight file here (``<score_id>.weights``).
    preflight : bool, default False
        Run :func:`ldpred3.preflight_prs` on every file before the first fit.
    traits : str or sequence of mappings, optional
        Per-trait table (see :func:`read_trait_table`) with its own ``n_eff`` /
        case-control counts / method / alpha. Shared kwargs still apply.
    **ldpred3_kwargs
        Passed through to :func:`ldpred3.run_ldpred3_prs` (``method``,
        ``n_eff``, ``block_size``, QC options, ...). Per-trait values from
        ``traits`` override these.

    Returns
    -------
    ScorePanel
        Scores on LDpred3's standardized convention (``standardized=True``).

    Notes
    -----
    Reusing one LD cache across traits requires a reference-wide cache, so this
    function writes ``ld_cache`` with ``subset_to_sumstats=False``. Loaded
    caches are then subset to each trait (LDpred3's default); passing
    ``subset_to_sumstats=False`` on a read is rejected. LDpred3 takes an
    exact principal subset after harmonisation and QC; the traits do not need
    identical rows or order. Fits use
    ``score=False`` against one shared, immutable prepared-target context.
    Multipgs then applies all standardized fitted weights together and records
    the target AF/SD needed for frozen deployment. An existing external or
    reference-wide cache therefore needs one full LD validation, one metadata
    preparation, and one target-dosage decode. Building in-sample LD adds the
    unavoidable dosage read that constructs that new cache. ``preflight=True``
    deliberately adds per-trait diagnostic reads before this fit path.
    """
    from ldpred3 import run_ldpred3_prs
    from ldpred3.interop import (PreparedLDCache, prepare_ld_cache,
                                 prepare_target, write_weights)

    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    n_jobs = _positive_integer(n_jobs, "n_jobs")
    if traits is not None:
        rows = read_trait_table(traits) if isinstance(traits, (str, os.PathLike)) \
            else [dict(row) for row in traits]
        items = [(str(row["id"]), str(row["path"]), row) for row in rows]
    elif hasattr(sumstats, "items"):
        items = [(sid, path, {}) for sid, path in sumstats.items()]
    else:
        paths = [str(p) for p in _expand_paths(sumstats, _SUMSTAT_SUFFIXES)]
        if score_ids is None:
            items = [(_stem(p), p, {}) for p in paths]
        else:
            ids = list(score_ids)
            if len(ids) != len(paths):
                raise ValueError(f"score_ids has {len(ids)} entries for "
                                 f"{len(paths)} sumstats files")
            items = [(sid, path, {}) for sid, path in zip(ids, paths)]
    if not items:
        raise ValueError("no summary-statistic files given")
    _require_unique_score_ids(
        [sid for sid, _, _ in items],
        sources=[path for _, path, _ in items])

    if n_jobs > 1 and len(items) > 1 and ld_prefix is not None \
            and ld_cache is None:
        raise ValueError(
            "n_jobs>1 with ld_prefix requires ld_cache=... so the first trait "
            "builds one shared LD cache instead of every worker rebuilding LD")

    kwargs = dict(ldpred3_kwargs)
    # The panel owns scoring: fit each model without another full target pass,
    # then stream all K fitted columns together below.
    kwargs["score"] = False
    if ld_prefix is not None:
        kwargs["ld_prefix"] = ld_prefix
    if ld_cache is None and kwargs.get("ld_prefix") is None:
        warnings.warn(
            "panel_from_sumstats is building LD from the target genotypes; "
            "pass ld_prefix= or an existing ld_cache= so the LD reference is "
            "not the cohort you will score", stacklevel=2)
    prepared_cache = None
    owns_prepared_cache = False
    if ld_cache is not None:
        reading = (isinstance(ld_cache, PreparedLDCache)
                   or os.path.exists(os.fspath(ld_cache)))
        if reading:
            # A loaded cache is subset to the current trait. False is only
            # valid while writing a new reference-wide generation.
            if kwargs.get("subset_to_sumstats") is False:
                raise ValueError(
                    "subset_to_sumstats=False applies only when building a "
                    "new reference-wide cache; an existing ld_cache is "
                    "subset to each trait automatically")
            kwargs.pop("subset_to_sumstats", None)
            if isinstance(ld_cache, PreparedLDCache):
                kwargs["ld_cache"] = ld_cache
                prepared_cache = ld_cache
            else:
                kwargs["ld_cache"] = ld_cache
        else:
            if kwargs.get("subset_to_sumstats") is True:
                raise ValueError(
                    "panel_from_sumstats with ld_cache requires "
                    "subset_to_sumstats=False so disjoint traits share one "
                    "reference-wide cache")
            kwargs["subset_to_sumstats"] = False
            # LDpred3 reads ld_cache immediately; a path being created must be
            # passed only as ld_out on the first successful fit. Promote it to
            # ld_cache below before dispatching the remaining traits.
            kwargs["ld_out"] = ld_cache
    if n_jobs > 1 and int(kwargs.get("ncores", 1) or 1) > 1:
        warnings.warn(
            "n_jobs>1 with ncores>1 oversubscribes the machine; leave "
            "ncores=1 when running traits concurrently", stacklevel=2)
    if weights_dir is not None:
        os.makedirs(str(weights_dir), exist_ok=True)
    if preflight:
        _preflight_traits(items, plink, kwargs, on_error=on_error)

    # Repeated fit-only traits need only target metadata. Keep one immutable
    # selector in memory instead of rescanning BIM/FAM or the BGEN variant
    # stream K times. Nonexistent paths are left to LDpred3: several lightweight
    # callers deliberately replace its runner with a test or planning stub.
    prepared_target = kwargs.get("prepared_target")
    target_present = (os.path.isfile(os.fspath(plink))
                      or os.path.isfile(os.fspath(plink) + ".bed"))
    if prepared_target is None and target_present:
        prepared_target = prepare_target(
            plink, sample_path=kwargs.get("sample_path"))
        kwargs["prepared_target"] = prepared_target
    target_prepared_by_panel = (
        prepared_target is not None
        and ldpred3_kwargs.get("prepared_target") is None)

    # Full payload validation is deliberately done once. Arbitrary path inputs
    # keep LDpred3's strict full-validation default; the resulting read-only
    # context may then be shared safely by the later trait fits.
    if (ld_cache is not None and not isinstance(ld_cache, PreparedLDCache)
            and os.path.exists(os.fspath(ld_cache))):
        prepared_cache = prepare_ld_cache(ld_cache)
        owns_prepared_cache = True
        kwargs["ld_cache"] = prepared_cache
        # Once the exact cache generation is open, reloading the external LD
        # reference cannot affect the fit; it only repeats its largest I/O.
        kwargs.pop("ld_prefix", None)
        kwargs.pop("ld_sample_path", None)
    elif isinstance(ld_cache, PreparedLDCache):
        kwargs.pop("ld_prefix", None)
        kwargs.pop("ld_sample_path", None)

    columns, ids, metas, weight_tables, fit_results = [], [], [], [], []
    fid = iid = None
    n_failed = 0
    n_items = len(items)

    def _consume(sid, path, res, trait):
        nonlocal fid, iid
        if fid is None:
            fid, iid = res.sample_fid, res.sample_iid
        elif (not np.array_equal(np.asarray(res.sample_fid), np.asarray(fid))
              or not np.array_equal(np.asarray(res.sample_iid),
                                    np.asarray(iid))):
            raise RuntimeError(f"{sid} was scored on a different sample order "
                               f"than the earlier traits")
        used = _trait_kwargs(kwargs, trait)
        inference = _inference_metadata(res, used)
        meta = {"path": str(path), "n_matched": int(res.var_index.size),
                "harmonize_log": dict(res.harmonize_log),
                "qc_log": dict(res.qc_log or {}),
                "inference": inference}
        n_eff = used.get("n_eff")
        if n_eff is not None:
            meta["n_eff"] = float(n_eff)
        fit_results.append(res)
        if res.scores is not None:  # compatibility with a pre-0.5/fake result
            columns.append(np.asarray(res.scores, dtype=float))
        ids.append(sid)
        metas.append(meta)
        weight_tables.append({
            "id": np.asarray(res.variant_id), "chrom": np.asarray(res.chrom),
            "pos": np.asarray(res.pos), "a1": np.asarray(res.effect_allele),
            "a2": np.asarray(res.other_allele),
            "weight": np.asarray(res.beta_adjusted, dtype=float),
            "af": None if res.af is None else np.asarray(res.af, dtype=float),
            "sd": None if res.sd is None else np.asarray(res.sd, dtype=float),
        })

    def _fit(sid, path, trait):
        try:
            return (sid, path, trait,
                    run_ldpred3_prs(str(path), plink,
                                    **_trait_kwargs(kwargs, trait)), None)
        except Exception as exc:
            return sid, path, trait, None, exc

    try:
        remaining = list(enumerate(items))
        # When requested, the first successful trait writes the LD cache; later
        # traits only read it. Without ld_cache each fit builds its own LD.
        first = None
        while remaining:
            i, (sid, path, trait) = remaining.pop(0)
            if progress is not None:
                progress(i, n_items, sid)
            sid, path, trait, res, exc = _fit(sid, path, trait)
            if exc is not None:
                if on_error == "raise":
                    raise RuntimeError(f"LDpred3 failed on {sid} ({path}): "
                                       f"{exc}") from exc
                n_failed += 1
                continue
            if kwargs.pop("ld_out", None) is not None:
                # The first successful fit produced this generation. Validate it
                # once before sharing it with sequential or concurrent fits.
                # Subsequent traits load the cache and must not keep the
                # write-only subset_to_sumstats=False flag.
                kwargs.pop("subset_to_sumstats", None)
                if os.path.exists(os.fspath(ld_cache)):
                    prepared_cache = prepare_ld_cache(ld_cache)
                    owns_prepared_cache = True
                    kwargs["ld_cache"] = prepared_cache
                    kwargs.pop("ld_prefix", None)
                    kwargs.pop("ld_sample_path", None)
                else:  # mocked compatibility tests do not create a real file
                    kwargs["ld_cache"] = ld_cache
            first = (sid, path, res, trait)
            break
        if first is None:
            raise ValueError(f"all {n_items} LDpred3 fits failed")
        _consume(*first)

        if n_jobs == 1 or not remaining:
            later = []
            for i, (sid, path, trait) in remaining:
                if progress is not None:
                    progress(i, n_items, sid)
                later.append(_fit(sid, path, trait))
        else:
            for i, (sid, path, trait) in remaining:
                if progress is not None:
                    progress(i, n_items, sid)
            workers = min(n_jobs, len(remaining))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                later = list(pool.map(
                    lambda item: _fit(item[1][0], item[1][1], item[1][2]),
                    remaining))

        for sid, path, trait, res, exc in later:
            if exc is not None:
                if on_error == "raise":
                    raise RuntimeError(f"LDpred3 failed on {sid} ({path}): "
                                       f"{exc}") from exc
                n_failed += 1
                continue
            _consume(sid, path, res, trait)
    finally:
        if owns_prepared_cache and prepared_cache is not None:
            prepared_cache.close()

    if not fit_results:
        raise ValueError(f"all {len(items)} LDpred3 fits failed")
    if columns and len(columns) != len(fit_results):
        raise RuntimeError(
            "LDpred3 returned a mixture of scored and fit-only results")
    if columns:
        panel_scores = np.column_stack(columns)
    else:
        panel_scores, scales = _score_ldpred3_fits_once(
            fit_results, plink,
            sample_path=ldpred3_kwargs.get("sample_path"),
            prepared_target=prepared_target)
        for table, (af, sd) in zip(weight_tables, scales):
            table["af"], table["sd"] = af, sd

    if weights_dir is not None:
        for sid, meta, table in zip(ids, metas, weight_tables):
            dest = os.path.join(str(weights_dir), f"{sid}.weights")
            write_weights(
                dest, id=table["id"], chrom=table["chrom"], pos=table["pos"],
                effect_allele=table["a1"], other_allele=table["a2"],
                weight=table["weight"], af=table["af"], sd=table["sd"])
            meta["weights"] = dest

    K = len(fit_results)
    return ScorePanel(
        scores=panel_scores, sample_fid=np.asarray(fid),
        sample_iid=np.asarray(iid), score_ids=np.array(ids, dtype=object),
        standardized=np.ones(K, dtype=bool), weights=weight_tables,
        meta=metas,
        log={"source": "ldpred3", "n_scores": K, "n_failed": n_failed,
             "n_jobs": int(n_jobs),
             "target": str(plink),
             "ld_cache": (str(os.fspath(ld_cache))
                          if ld_cache is not None else None),
             "ld_prefix": str(ld_prefix) if ld_prefix else None,
             "ld_reused": bool(ld_cache),
             "cache_prepared": prepared_cache is not None,
             "cache_validations": int(owns_prepared_cache),
             "cache_validation_owner": (
                 "panel" if owns_prepared_cache else
                 "caller" if isinstance(ld_cache, PreparedLDCache) else
                 "none"),
             "target_prepared": prepared_target is not None,
             "target_preparation_owner": (
                 "panel" if target_prepared_by_panel else
                 "caller" if prepared_target is not None else "none"),
             "target_scoring_passes": 1,
             "weights_dir": str(weights_dir) if weights_dir else None})


def _inference_metadata(result, run_kwargs):
    """Preserve inference output plus run controls absent from older results.

    Current LDpred3 ``PRSResult.inference`` records ``n_chains_kept`` but older
    releases omit the total number run. The total is recoverable when the
    caller explicitly supplied ``auto_chains`` or ``infer_params.n_chains``;
    the number retained alone is not enough, so no value is invented otherwise.
    """
    raw = getattr(result, "inference", None)
    out = {} if raw is None else dict(raw)

    # Prefer metadata a newer result object may expose directly.
    for key in ("n_chains", "shrinkage", "shrink_corr"):
        value = getattr(result, key, None)
        if key not in out and value is not None:
            out[key] = value

    params = dict(run_kwargs.get("infer_params") or {})
    if "n_chains" not in out:
        auto_chains = run_kwargs.get("auto_chains")
        if auto_chains is not None and int(auto_chains) > 1:
            # LDpred3 overwrites infer_params.n_chains with auto_chains on this
            # path, so it is the authoritative total.
            out["n_chains"] = int(auto_chains)
        elif params.get("n_chains") is not None:
            out["n_chains"] = int(params["n_chains"])
    if "shrinkage" not in out and "shrink_corr" not in out:
        if params.get("shrink_corr") is not None:
            out["shrink_corr"] = float(params["shrink_corr"])
    return out


def _stem(path):
    base = os.path.basename(str(path))
    for suffix in (".tsv.gz", ".txt.gz", ".gz", ".tsv", ".txt", ".sumstats",
                   ".weights"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def _trait_kwargs(shared, trait):
    """Merge per-trait overrides onto the shared ``run_ldpred3_prs`` kwargs."""
    out = dict(shared)
    if not trait:
        return out
    n_eff = trait.get("n_eff")
    n_cases, n_controls = trait.get("n_cases"), trait.get("n_controls")
    if n_cases is not None or n_controls is not None:
        if n_cases is None or n_controls is None:
            raise ValueError(f"{trait.get('id', '?')}: n_cases and n_controls "
                             "must be given together")
        from ldpred3 import n_eff_case_control
        n_eff = n_eff_case_control(float(n_cases), float(n_controls))
    if n_eff is not None:
        out["n_eff"] = float(n_eff)
    for key in ("method", "alpha"):
        if trait.get(key) is not None:
            out[key] = trait[key]
    return out


def _preflight_traits(items, plink, kwargs, *, on_error):
    from ldpred3 import preflight_prs

    warnings_out = []
    for sid, path, trait in items:
        used = _trait_kwargs(kwargs, trait)
        try:
            report = preflight_prs(
                str(path), plink, n_eff=used.get("n_eff"),
                sample_path=used.get("sample_path"),
                qc=used.get("qc", True),
                subset_to_sumstats=used.get("subset_to_sumstats", True))
        except Exception as exc:
            if on_error == "raise":
                raise RuntimeError(f"preflight failed on {sid} ({path}): "
                                   f"{exc}") from exc
            warnings_out.append(f"{sid}: {exc}")
            continue
        missing = report.get("missing") or []
        if missing:
            msg = f"{sid}: unresolved columns {missing}"
            if on_error == "raise":
                raise ValueError(msg)
            warnings_out.append(msg)
        for note in report.get("warnings") or []:
            warnings_out.append(f"{sid}: {note}")
    if warnings_out:
        warnings.warn("preflight: " + "; ".join(warnings_out[:8]),
                      stacklevel=3)


def read_trait_table(path):
    """Read a per-trait table for :func:`panel_from_sumstats`.

    Required columns (any case): ``TRAIT``/``SCORE``/``ID`` and ``PATH``.
    Optional: ``N_EFF``, ``N_CASES``, ``N_CONTROLS``, ``METHOD``, ``ALPHA``.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        header = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if header is None:
                header = [p.lstrip("#").upper() for p in parts]
                continue
            if len(parts) != len(header):
                raise ValueError(f"{path}: expected {len(header)} columns, "
                                 f"got {len(parts)}")
            rows.append(dict(zip(header, parts)))
    if header is None:
        raise ValueError(f"{path}: no header row")
    id_col = next((c for c in ("TRAIT", "SCORE", "ID") if c in header), None)
    if id_col is None or "PATH" not in header:
        raise ValueError(f"{path}: need TRAIT/SCORE/ID and PATH columns")

    def _num(value):
        if value in ("", ".", "NA", "N/A", "NULL"):
            return None
        return float(value)

    out = []
    for row in rows:
        rec = {"id": row[id_col], "path": row["PATH"],
               "n_eff": _num(row["N_EFF"]) if "N_EFF" in row else None,
               "n_cases": _num(row["N_CASES"]) if "N_CASES" in row else None,
               "n_controls": _num(row["N_CONTROLS"]) if "N_CONTROLS" in row
               else None,
               "method": row.get("METHOD") or None,
               "alpha": _num(row["ALPHA"]) if "ALPHA" in row else None}
        if rec["method"] is not None:
            rec["method"] = rec["method"].strip() or None
        out.append(rec)
    if not out:
        raise ValueError(f"{path}: no trait rows")
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate trait id(s)")
    return out


def _sidecar_metadata(files):
    dirs = {os.path.dirname(os.path.abspath(f)) for f in files}
    if len(dirs) != 1:
        return None
    candidate = os.path.join(next(iter(dirs)), "metadata.tsv")
    return candidate if os.path.isfile(candidate) else None


def attach_metadata(panel, metadata):
    """Merge a SCORE-keyed metadata table into ``panel.meta``.

    ``N_EFF`` is stored as ``meta[k]['n_eff']`` so
    :func:`architectures_from_panel` can read it without a second argument.
    """
    from .fetch import read_score_metadata

    table = read_score_metadata(metadata) if isinstance(
        metadata, (str, os.PathLike)) else dict(metadata)
    if not panel.meta:
        panel.meta = [{} for _ in range(panel.n_scores)]
    if len(panel.meta) != panel.n_scores:
        raise ValueError("panel.meta length does not match n_scores")
    for i, sid in enumerate(np.asarray(panel.score_ids, dtype=object)):
        row = table.get(str(sid))
        if not row:
            continue
        meta = dict(panel.meta[i] or {})
        meta.update(row)
        if "N_EFF" in row and np.isfinite(row["N_EFF"]):
            meta["n_eff"] = float(row["N_EFF"])
        panel.meta[i] = meta
    return panel


# ---------------------------------------------------------------------------
# From saved ldpred3 weight files
# ---------------------------------------------------------------------------

def panel_from_weights(paths, plink, *, sample_path=None, drop_ambiguous=True,
                       min_matched=1, on_error="raise", block=None,
                       progress=None):
    """Score a directory of ldpred3 weight files against one target.

    Files written by :meth:`ldpred3.pipeline.PRSResult.write_weights` (columns
    ``ID CHR POS A1 A2 WEIGHT``, optionally ``AF_REF SD_REF``) are harmonised
    once and applied in a single genotype pass. Frozen ``AF_REF``/``SD_REF``
    are used when present, matching
    :func:`ldpred3.score_from_weights` ``scaling="frozen"``. Files without
    those columns use this target's AF/SD, matching ``scaling="target"``.
    """
    from ldpred3.genotype_io import read_bed, read_bim, read_fam, strip_ext
    from ldpred3.interop import (Sumstats, harmonize, prepare_target,
                                 read_weights)

    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    files = _expand_paths(paths, _WEIGHT_SUFFIXES)
    if not files:
        raise ValueError(f"no weight files found in {paths!r}")

    is_bgen = str(plink).lower().endswith(".bgen")
    if is_bgen:
        target = prepare_target(plink, sample_path=sample_path)
        variants = target.variants
        fid, iid = target.samples.fid, target.samples.iid
        n_samples, n_total = len(fid), len(variants.id)
    else:
        prefix = strip_ext(str(plink))
        variants = read_bim(prefix + ".bim")
        samples = read_fam(prefix + ".fam")
        fid, iid = samples.fid, samples.iid
        n_samples, n_total = len(fid), len(variants.id)

    score_ids, metas, per_score, frozen, n_failed, n_skipped = (
        [], [], [], [], 0, 0)
    for i, path in enumerate(files):
        try:
            wt = read_weights(path)
        except Exception:
            if on_error == "raise":
                raise
            n_failed += 1
            continue
        sid = _stem(path)
        if progress is not None:
            progress(i, len(files), sid)
        m = len(wt)
        ss = Sumstats(id=wt.id, chrom=wt.chrom, pos=wt.pos, ea=wt.a1,
                      oa=wt.a2, beta=wt.weight, se=np.ones(m),
                      n_eff=np.ones(m), eaf=np.full(m, np.nan),
                      info=np.full(m, np.nan))
        try:
            h = harmonize(ss, variants, drop_ambiguous=drop_ambiguous)
        except Exception:
            if on_error == "raise":
                raise
            n_failed += 1
            continue
        if len(h) < min_matched:
            msg = (f"{sid}: only {len(h)} of {m} variants matched the target "
                   f"(min_matched={min_matched})")
            if on_error == "raise":
                raise ValueError(msg)
            n_skipped += 1
            continue
        w = np.asarray(h.beta, dtype=float)
        var_index = np.asarray(h.var_index, dtype=np.int64)
        if wt.has_scale:
            fa = np.asarray(wt.af_ref, dtype=float)[h.src_index]
            fs = np.asarray(wt.sd_ref, dtype=float)[h.src_index]
            fa = np.where(h.flipped, 1.0 - fa, fa)
            frozen.append((fa, fs))
        else:
            frozen.append(None)
        score_ids.append(sid)
        metas.append({"path": str(path), "n_matched": int(len(h)),
                      "has_scale": bool(wt.has_scale),
                      "harmonize_log": dict(h.log)})
        per_score.append((var_index, w))

    _require_unique_score_ids(
        score_ids, sources=[m["path"] for m in metas])

    if not per_score:
        raise ValueError(f"none of the {len(files)} weight files produced a "
                         f"usable score ({n_failed} unreadable, {n_skipped} "
                         f"below min_matched)")

    scores, union, af, sd = _accumulate_scores(
        per_score, n_samples, n_total, plink, standardize=False,
        block=block, read_bed=read_bed, strip_ext=strip_ext, is_bgen=is_bgen,
        frozen=frozen, sample_path=sample_path)
    variant_arrays = {
        "id": np.asarray(variants.id)[union],
        "chrom": np.asarray(variants.chrom)[union],
        "pos": np.asarray(variants.pos)[union],
        "a1": np.asarray(variants.a1)[union],
        "a2": np.asarray(variants.a2)[union],
        "af": af, "sd": sd,
    }
    variant_table = _SharedVariantTable(variant_arrays)
    index_dtype = _smallest_index_dtype(union.size)
    weight_tables = []
    for (var_index, w), scale in zip(per_score, frozen):
        at = np.searchsorted(union, var_index).astype(index_dtype, copy=False)
        at.setflags(write=False)
        table = _CompactWeightTable(variant_table, at, w)
        if scale is not None:
            # Prefer the file's frozen AF/SD on this score's variants.
            weight_tables.append({
                "id": table["id"], "chrom": table["chrom"],
                "pos": table["pos"], "a1": table["a1"], "a2": table["a2"],
                "weight": w, "af": scale[0], "sd": scale[1],
            })
        else:
            weight_tables.append(table)
    K = len(per_score)
    return ScorePanel(
        scores=scores, sample_fid=np.asarray(fid), sample_iid=np.asarray(iid),
        score_ids=np.array(score_ids, dtype=object),
        standardized=np.ones(K, dtype=bool), weights=weight_tables,
        meta=metas,
        log={"source": "ldpred3_weights", "n_files": len(files),
             "n_scores": K, "n_failed": n_failed, "n_skipped": n_skipped,
             "target": str(plink)})


def check_weights(panel, fit, path, plink, *, min_corr=1.0 - 1e-8,
                  rtol=1e-6, atol=1e-8):
    """Require frozen scoring to reproduce the fitted centred predictor.

    Combined raw-allele scores can differ from the fitted predictor by one
    intercept, so both vectors are centred after aligning ``FID:IID``. A high
    correlation alone is insufficient: multiplying every deployed weight by
    two has correlation one but is not the fitted model. This check therefore
    also requires the centred residual to be no larger than
    ``atol + rtol * max(abs(fitted - mean(fitted)))``.

    Returns correlation, unit-slope diagnostics, the removable intercept,
    centred RMSE / maximum error, tolerance, and the aligned sample count.
    """
    from ldpred3 import score_from_weights

    validated = {}
    for name, value in (("min_corr", min_corr), ("rtol", rtol),
                        ("atol", atol)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite numeric value")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite numeric value") from None
        if not np.isfinite(number):
            raise ValueError(f"{name} must be a finite numeric value")
        validated[name] = number
    min_corr = validated["min_corr"]
    rtol = validated["rtol"]
    atol = validated["atol"]
    if not -1.0 <= min_corr <= 1.0:
        raise ValueError("min_corr must be in [-1, 1]")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")

    scored = score_from_weights(path, plink, scaling="frozen")
    direct = np.asarray(fit.multi_pgs(panel), dtype=float)
    mine = _keys(panel.sample_fid, panel.sample_iid)
    theirs = _keys(scored.sample_fid, scored.sample_iid)
    _require_unique_keys(mine, "the panel")
    _require_unique_keys(theirs, "the scored cohort")
    lookup = {k: i for i, k in enumerate(theirs)}
    idx = [lookup[k] for k in mine if k in lookup]
    keep = [i for i, k in enumerate(mine) if k in lookup]
    if not keep:
        raise ValueError("the panel and the scored cohort share no "
                         "individuals (matched on FID:IID)")
    expected = direct[keep]
    observed = np.asarray(scored.scores, dtype=float)[idx]
    expected_centered = expected - expected.mean()
    observed_centered = observed - observed.mean()
    corr = float(np.corrcoef(expected_centered, observed_centered)[0, 1])
    denom = float(expected_centered @ expected_centered)
    slope = (float(expected_centered @ observed_centered) / denom
             if denom > 0.0 else np.nan)
    residual = observed_centered - expected_centered
    max_abs_error = float(np.max(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    scale = float(np.max(np.abs(expected_centered)))
    tolerance = atol + rtol * scale
    intercept = float(observed.mean() - expected.mean())
    if not np.isfinite(corr) or corr < min_corr:
        raise ValueError(
            f"frozen scoring of {path} correlates {corr:.8f} with the fitted "
            f"combination (required >= {min_corr:.8f})")
    if not np.isfinite(max_abs_error) or max_abs_error > tolerance:
        raise ValueError(
            f"frozen scoring of {path} does not reproduce the fitted centred "
            f"combination: slope={slope:.8g}, max centred error="
            f"{max_abs_error:.8g} (allowed {tolerance:.8g})")
    return {"corr": corr, "slope": slope, "intercept": intercept,
            "rmse": rmse, "max_abs_error": max_abs_error,
            "tolerance": tolerance, "n": len(keep)}


# ---------------------------------------------------------------------------
# Folding a fitted stack back into one weight vector
# ---------------------------------------------------------------------------

def combine_weights(panel, fit, *, path=None):
    """Collapse a panel and its stacking coefficients into one weight table.

    For each variant, every component is first written as
    ``c_k (g - mu_k)`` on one allele orientation. The folded coefficient is
    ``c = sum_k beta_k c_k`` and its frozen centre is
    ``mu = sum_k beta_k c_k mu_k / c``. This retains different component
    AF/SD references, including their missing-genotype behaviour, rather than
    attaching the first component's scale to an untransformed weight sum. This
    is the deployable artefact: a new cohort is scored from it directly, with
    no reference to the ``K`` inputs and no need to rebuild the panel.

    The common scale is LDpred3's **standardized** one — weights apply to
    ``(g - 2f)/sd`` — because that is what :func:`ldpred3.score_from_weights`
    applies, so the file this writes can be handed straight to it::

        multipgs.combine_weights(panel, fit, path="multi.weights")
        ldpred3.score_from_weights("multi.weights", "new_cohort",
                                   scaling="frozen")

    Allele-count weights (everything from the PGS Catalog) enter training as
    ``w*g`` but are written as ``w*(g-mu)`` on the frozen scale. Their omitted
    ``w*mu`` terms sum to one additive intercept, so the combined file
    reproduces the centred training-time score exactly. That intercept is
    irrelevant to R², AUC and ranking, and is absorbed by any downstream
    regression. ``scaling="frozen"`` is the mode to score with: it reuses the
    effective ``AF_REF``/``SD_REF`` written here, so a cohort with different
    allele frequencies is still scored on the scale the coefficients were
    fitted on. A signed combination whose required effective AF falls outside
    ``[0, 1]`` is genuinely not representable by that file format and is
    rejected rather than silently changed.

    Parameters
    ----------
    panel : ScorePanel
        Must carry ``weights`` — a panel read back with :func:`read_panel` from
        a plain score matrix does not.
    fit : MultiPGSFit, MetaPGS, or SumstatFit
        Its raw-score ``beta`` and ``score_ids`` must match the panel's, in
        order. In particular, ``SumstatFit.beta`` is already on this raw score
        scale; ``beta_std`` is for comparing effects, not deployment.
    path : str, optional
        Write the table here, in ldpred3's
        ``ID CHR POS A1 A2 WEIGHT AF_REF SD_REF`` format.

    Returns
    -------
    dict with keys ``id chrom pos a1 a2 weight af sd``.
    """
    if not panel.weights:
        raise ValueError("this panel carries no per-variant weights, so there "
                         "is nothing to combine; rebuild it with "
                         "panel_from_catalog or panel_from_sumstats")
    beta = np.asarray(fit.beta, dtype=float)
    if beta.ndim != 1:
        raise ValueError(f"fit.beta must be a 1-D raw-score coefficient "
                         f"vector, got shape {beta.shape}")
    if not np.all(np.isfinite(beta)):
        raise ValueError("fit.beta contains non-finite coefficients")
    if beta.size != panel.n_scores:
        raise ValueError(f"fit has {beta.size} coefficients but the panel has "
                         f"{panel.n_scores} scores")
    if len(panel.weights) != panel.n_scores:
        raise ValueError(f"panel has {len(panel.weights)} weight tables for "
                         f"{panel.n_scores} score columns")
    standardized = np.asarray(panel.standardized)
    if standardized.shape != (panel.n_scores,):
        raise ValueError("panel.standardized must have one entry per score")
    panel_id_array = np.asarray(panel.score_ids, dtype=object)
    fit_id_array = np.asarray(fit.score_ids, dtype=object)
    if panel_id_array.shape != (panel.n_scores,):
        raise ValueError("panel.score_ids must have one entry per score")
    if fit_id_array.shape != (panel.n_scores,):
        raise ValueError("fit.score_ids must have one entry per coefficient")
    panel_ids = [str(s) for s in panel_id_array]
    fit_ids = [str(s) for s in fit_id_array]
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("panel.score_ids must be unique before combining "
                         "weights")
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit.score_ids must be unique before combining "
                         "weights")
    if fit_ids != panel_ids:
        raise ValueError("the fit's score_ids do not match the panel's; "
                         "combining them would attach coefficients to the "
                         "wrong scores")

    acc = {}
    for k, table in enumerate(panel.weights):
        b = beta[k]
        if b == 0.0:
            continue
        if table is None:
            raise ValueError(
                f"score {panel_ids[k]!r} carries no weight table, so its "
                "non-zero coefficient cannot be deployed")
        metadata, index = _weight_table_storage(table)
        w = np.asarray(table["weight"], dtype=float)
        if w.ndim != 1 or not np.all(np.isfinite(w)):
            raise ValueError(
                f"score {panel_ids[k]!r} weights must be a finite 1-D array")
        sd = metadata.get("sd")
        sd = None if sd is None else np.asarray(sd, dtype=float)
        af = metadata.get("af")
        af = None if af is None else np.asarray(af, dtype=float)
        if af is None or sd is None:
            scale = "standardized" if standardized[k] else "allele-count"
            raise ValueError(
                f"score {panel_ids[k]!r} has {scale} weights but no complete "
                "per-variant AF/SD, so its centring cannot be represented. "
                "Rebuild the panel from genotypes to record that scale.")
        ids = np.asarray(metadata["id"], dtype=object)
        chrom = np.asarray(metadata["chrom"], dtype=object)
        pos = np.asarray(metadata["pos"])
        a1 = np.asarray(metadata["a1"], dtype=object)
        a2 = np.asarray(metadata["a2"], dtype=object)
        # Non-zero weights only: the per-variant reconciliation below is
        # Python-level, so dense-but-mostly-zero tables should not pay for it.
        for j in np.flatnonzero(w):
            q = j if index is None else int(index[j])
            e1, e2 = str(a1[q]).upper(), str(a2[q]).upper()
            # The unordered allele pair distinguishes multiallelic variants at
            # one coordinate while still coalescing A/G with its G/A swap.
            pair = tuple(sorted((e1, e2)))
            key = (str(chrom[q]), int(pos[q]), pair[0], pair[1])
            a_j = float(af[q])
            s_j = float(sd[q])
            if not np.isfinite(s_j) or s_j < 0.0:
                raise ValueError(
                    f"score {panel_ids[k]!r} has invalid SD for variant "
                    f"{ids[q]!r}")
            # A non-positive target/reference SD makes this variant constant.
            # It contributes only an intercept even for a raw allele-count
            # score, so its nominal weight must not reappear through another
            # component's scale.
            if s_j <= 1e-12:
                continue
            if not np.isfinite(a_j) or not 0.0 <= a_j <= 1.0:
                raise ValueError(
                    f"score {panel_ids[k]!r} has invalid AF for variant "
                    f"{ids[q]!r}")

            coef = b * w[j]
            if standardized[k]:
                coef /= s_j
            mean_j = 2.0 * a_j
            entry = acc.get(key)
            if entry is None:
                # Last two fields retain absolute summation scales so ordinary
                # floating cancellation is distinguished from a genuinely
                # tiny non-zero effect.
                entry = [str(ids[q]), e1, e2, 0.0, 0.0, np.nan, 0.0, 0.0]
                acc[key] = entry
            elif e1 == entry[2] and e2 == entry[1]:
                # Reorient c(g-mu): g_other=2-g_ref and
                # mu_other=2-mu_ref, hence both c's sign and mu's orientation
                # change together.
                coef = -coef
                mean_j = 2.0 - mean_j
            elif e1 != entry[1] or e2 != entry[2]:
                raise ValueError(
                    f"incompatible allele orientations at {key[0]}:{key[1]}")
            entry[3] += coef
            entry[4] += coef * mean_j
            entry[6] += abs(coef)
            entry[7] += abs(coef * mean_j)
            if not np.isfinite(entry[5]) and s_j > 1e-12:
                # SD only sets the numeric units of the output weight. The
                # coefficient-weighted centre above carries the substantive
                # cross-component scale information.
                entry[5] = s_j

    if not acc:
        raise ValueError("every selected score contributed no non-zero weight; "
                         "the fit is null, so there is nothing to deploy")
    # Natural chromosome order: numeric chromosomes first in numeric order,
    # then non-numeric ones — not "1", "10", "11", ..., "2".
    def _chrom_key(chrom):
        return (0, int(chrom)) if chrom.isdigit() else (1, chrom)

    keys = sorted(acc, key=lambda kv: (_chrom_key(kv[0]), kv[1], kv[2], kv[3]))
    rows = []
    for key in keys:
        entry = acc[key]
        coef, centred_sum, output_sd = entry[3], entry[4], entry[5]
        coef_zero = abs(coef) <= 1e-14 * entry[6]
        centre_zero = abs(centred_sum) <= 1e-14 * entry[7]
        if coef_zero:
            if centre_zero:
                continue
            raise ValueError(
                f"combined weights at {key[0]}:{key[1]} cancel in dosage "
                "effect but not in frozen centre, so missing-genotype "
                "behaviour cannot be represented by one weight row")
        mean = centred_sum / coef
        if mean < -1e-12 or mean > 2.0 + 1e-12:
            raise ValueError(
                f"the exact combined frozen centre at {key[0]}:{key[1]} "
                f"implies AF={mean / 2.0:.8g}, outside [0, 1]; one LDpred3 "
                "weight row cannot represent this signed combination")
        if not np.isfinite(output_sd) or output_sd <= 1e-12:
            raise ValueError(
                f"combined variant {key[0]}:{key[1]} has no positive "
                "reference SD and cannot be written on a standardized scale")
        rows.append((key, entry, coef * output_sd,
                     float(np.clip(mean / 2.0, 0.0, 1.0)), output_sd))
    if not rows:
        raise ValueError("all combined variant weights cancel exactly; there "
                         "is no non-zero weight set to deploy")
    out = {
        "id": np.array([entry[0] for _, entry, _, _, _ in rows], dtype=object),
        "chrom": np.array([key[0] for key, _, _, _, _ in rows], dtype=object),
        "pos": np.array([key[1] for key, _, _, _, _ in rows], dtype=np.int64),
        "a1": np.array([entry[1] for _, entry, _, _, _ in rows], dtype=object),
        "a2": np.array([entry[2] for _, entry, _, _, _ in rows], dtype=object),
        "weight": np.array([weight for _, _, weight, _, _ in rows], dtype=float),
        "af": np.array([af_j for _, _, _, af_j, _ in rows], dtype=float),
        "sd": np.array([sd_j for _, _, _, _, sd_j in rows], dtype=float),
    }
    if path is not None:
        from ldpred3.interop import write_weights
        write_weights(path, id=out["id"], chrom=out["chrom"], pos=out["pos"],
                      effect_allele=out["a1"], other_allele=out["a2"],
                      weight=out["weight"], af=out["af"], sd=out["sd"])
    return out


def _weight_table_storage(table):
    """Return ``(metadata, index)`` for compact or legacy weight mappings."""
    if isinstance(table, _CompactWeightTable):
        return table.variant_table, table.index
    # Accept the explicit equivalent as well, so callers need not construct
    # the private mapping class to supply compact tables themselves.
    if (isinstance(table, Mapping) and "variant_table" in table
            and "index" in table):
        return table["variant_table"], np.asarray(table["index"])
    return table, None


# ---------------------------------------------------------------------------
# Plain-text panels
# ---------------------------------------------------------------------------

def write_panel(panel, path):
    """Write the score matrix as TSV: ``FID IID <score ids...>``.

    Per-variant weights are *not* written — this is the matrix, for reuse in a
    fit, not the artefact for scoring a new cohort. Use
    :func:`combine_weights` for that.
    """
    ids = [str(s) for s in np.asarray(panel.score_ids, dtype=object)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(["FID", "IID", *ids]) + "\n")
        for i in range(len(panel)):
            row = [str(panel.sample_fid[i]), str(panel.sample_iid[i])]
            row.extend(f"{v:.6g}" for v in panel.scores[i])
            fh.write("\t".join(row) + "\n")
    return path


def read_panel(path):
    """Read a panel written by :func:`write_panel` or :func:`save_panel`.

    A TSV carries the score matrix only, so ``standardized`` comes back
    all-``False`` and weights/metadata are empty regardless of what the panel
    had when written; use the ``.npz`` form (:func:`save_panel`) to preserve
    them. The flags are currently consulted only by :func:`combine_weights`,
    which refuses weightless panels outright.
    """
    if str(path).endswith(".npz"):
        return load_panel(path)
    if os.path.exists(str(path) + ".npz"):
        return load_panel(str(path) + ".npz")
    fid, iid, rows = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if len(header) < 2 or header[0] != "FID" or header[1] != "IID":
            raise ValueError(f"{path}: expected a 'FID\\tIID\\t...' header")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(header):
                raise ValueError(f"{path}: expected {len(header)} columns, got "
                                 f"{len(parts)}")
            fid.append(parts[0])
            iid.append(parts[1])
            rows.append([float(v) for v in parts[2:]])
    K = len(header) - 2
    scores = (np.asarray(rows, dtype=float).reshape(len(rows), K)
              if rows else np.empty((0, K), dtype=float))
    return ScorePanel(
        scores=scores, sample_fid=np.array(fid, dtype=object),
        sample_iid=np.array(iid, dtype=object),
        score_ids=np.array(header[2:], dtype=object),
        standardized=np.zeros(K, dtype=bool), weights=[], meta=[],
        log={"source": str(path),
             "standardized_unknown": "TSV carries no scale flags; assumed raw"})


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def save_panel(panel, path):
    """Serialize a :class:`ScorePanel` including weights and metadata."""
    payload = {
        "format": np.array([_PANEL_FORMAT], dtype=np.int16),
        "scores": np.asarray(panel.scores, dtype=float),
        "sample_fid": np.asarray(panel.sample_fid, dtype=object),
        "sample_iid": np.asarray(panel.sample_iid, dtype=object),
        "score_ids": np.asarray(panel.score_ids, dtype=object),
        "standardized": np.asarray(panel.standardized, dtype=bool),
        "meta_json": np.asarray(json.dumps(_jsonable(list(panel.meta)))),
        "log_json": np.asarray(json.dumps(_jsonable(dict(panel.log)))),
    }
    tables, table_of = [], {}
    kinds, indices, weights = [], [], []
    for table in panel.weights:
        if isinstance(table, _CompactWeightTable):
            tid = id(table.variant_table)
            if tid not in table_of:
                table_of[tid] = len(tables)
                tables.append(table.variant_table)
            kinds.append(1)
            indices.append(np.asarray(table.index))
            weights.append(np.asarray(table["weight"], dtype=float))
            payload[f"w_table_{len(kinds) - 1}"] = np.array(
                [table_of[tid]], dtype=np.int32)
        else:
            kinds.append(0)
            indices.append(np.array([], dtype=np.int64))
            weights.append(np.asarray(table["weight"], dtype=float))
            for key in ("id", "chrom", "pos", "a1", "a2"):
                payload[f"w_{len(kinds) - 1}_{key}"] = np.asarray(table[key])
            for key in ("af", "sd"):
                values = table.get(key)
                if values is not None:
                    payload[f"w_{len(kinds) - 1}_{key}"] = np.asarray(
                        values, dtype=float)
    payload["w_kind"] = np.asarray(kinds, dtype=np.int8)
    for i, (idx, w) in enumerate(zip(indices, weights)):
        payload[f"w_index_{i}"] = idx
        payload[f"w_weight_{i}"] = w
    for t, vt in enumerate(tables):
        for key in ("id", "chrom", "pos", "a1", "a2", "af", "sd"):
            payload[f"vt_{t}_{key}"] = np.asarray(vt[key])
    payload["n_variant_tables"] = np.array([len(tables)], dtype=np.int32)
    # numpy appends ".npz" itself; doing it here first keeps the returned and
    # readable path identical to what lands on disk.
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **payload)
    return path


def load_panel(path):
    """Load a panel written by :func:`save_panel`.

    Panel files contain NumPy object arrays and therefore require pickle while
    loading. Pickle can execute code: load only files from a source you trust.
    """
    path = str(path)
    if not path.endswith(".npz") and os.path.exists(path + ".npz"):
        path += ".npz"
    with np.load(path, allow_pickle=True) as z:
        fmt = int(z["format"][0]) if "format" in z else 0
        if fmt != _PANEL_FORMAT:
            raise ValueError(f"{path}: unsupported panel format {fmt}")
        n_tables = int(z["n_variant_tables"][0]) if "n_variant_tables" in z \
            else 0
        vtables = []
        for t in range(n_tables):
            arrays = {key: z[f"vt_{t}_{key}"]
                      for key in ("id", "chrom", "pos", "a1", "a2", "af", "sd")}
            vtables.append(_SharedVariantTable(arrays))
        kinds = np.asarray(z["w_kind"]) if "w_kind" in z else np.zeros(0)
        weights = []
        for i, kind in enumerate(kinds):
            if int(kind) == 1:
                tid = int(z[f"w_table_{i}"][0])
                weights.append(_CompactWeightTable(
                    vtables[tid], z[f"w_index_{i}"], z[f"w_weight_{i}"]))
            else:
                table = {key: z[f"w_{i}_{key}"]
                         for key in ("id", "chrom", "pos", "a1", "a2")
                         if f"w_{i}_{key}" in z}
                table["weight"] = z[f"w_weight_{i}"]
                for key in ("af", "sd"):
                    if f"w_{i}_{key}" in z:
                        table[key] = z[f"w_{i}_{key}"]
                weights.append(table)
        meta = json.loads(str(z["meta_json"]))
        log = json.loads(str(z["log_json"]))
        return ScorePanel(
            scores=np.asarray(z["scores"], dtype=float),
            sample_fid=np.asarray(z["sample_fid"], dtype=object),
            sample_iid=np.asarray(z["sample_iid"], dtype=object),
            score_ids=np.asarray(z["score_ids"], dtype=object),
            standardized=np.asarray(z["standardized"], dtype=bool),
            weights=weights, meta=meta, log=log)
