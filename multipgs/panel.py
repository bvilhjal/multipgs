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
The LD reference is built once and cached on disk, so trait 2 onwards pay only
for the fit. This is the arm of the paper that turns public GWAS for other
traits into scores; it is also how the target trait's own score is produced.

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

    def __len__(self):
        return int(self.scores.shape[0])

    @property
    def n_scores(self):
        return int(self.scores.shape[1])

    def index_of(self, score_id):
        """Column index of ``score_id``."""
        hits = np.flatnonzero(np.asarray(self.score_ids, dtype=object)
                              == score_id)
        if hits.size == 0:
            raise KeyError(f"no score {score_id!r} in this panel")
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
        return ScorePanel(
            scores=np.hstack([left.scores, right.scores]),
            sample_fid=left.sample_fid, sample_iid=left.sample_iid,
            score_ids=np.array(left_ids + right_ids, dtype=object),
            standardized=np.concatenate([
                np.asarray(left.standardized, dtype=bool),
                np.asarray(right.standardized, dtype=bool)]),
            weights=(list(left.weights) + list(right.weights)
                     if left.weights or right.weights else []),
            meta=(list(left.meta) + list(right.meta)
                  if left.meta or right.meta else []),
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
        Scoring files, or a directory to take ``*.txt``/``*.txt.gz`` from.
    plink : str
        PLINK prefix, or a ``.bgen`` path. PLINK is streamed; BGEN is loaded
        whole.
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
        What to do when a file fails to parse or matches nothing.
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

    from .catalog import read_scoring_file

    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    files = _expand_paths(paths)
    if not files:
        raise ValueError(f"no scoring files found in {paths!r}")

    is_bgen = str(plink).endswith(".bgen")
    if is_bgen:
        from ldpred3 import load_genotypes
        geno = load_genotypes(plink, sample_path=sample_path)
        variants = geno.variants
        fid, iid = geno.samples.fid, geno.samples.iid
        dosage = geno.dosage
        n_samples, n_total = len(fid), len(variants.id)
    else:
        prefix = strip_ext(str(plink))
        variants = read_bim(prefix + ".bim")
        samples = read_fam(prefix + ".fam")
        fid, iid = samples.fid, samples.iid
        n_samples, n_total = len(fid), len(variants.id)
        dosage = None

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
        from .catalog import harmonize_scoring_file
        var_index, w, log = harmonize_scoring_file(
            sf, variants, drop_ambiguous=drop_ambiguous)
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

    if not per_score:
        raise ValueError(f"none of the {len(files)} scoring files produced a "
                         f"usable score ({n_failed} unreadable, {n_skipped} "
                         f"below min_matched)")

    scores, union, af, sd = _accumulate_scores(
        per_score, n_samples, n_total, plink, dosage, standardize=standardize,
        block=block, read_bed=read_bed, strip_ext=strip_ext, is_bgen=is_bgen)

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


def _expand_paths(paths, suffixes=(".txt", ".txt.gz", ".tsv", ".tsv.gz",
                                   ".weights")):
    """One path, a directory, or a mix of both, to a flat list of files."""
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


def _accumulate_scores(per_score, n_samples, n_total, plink, dosage, *,
                       standardize, block, read_bed, strip_ext, is_bgen,
                       frozen=None):
    """One pass over the genotypes: ``K`` scores, plus per-variant AF and SD."""
    K = len(per_score)
    union = np.unique(np.concatenate([vi for vi, _ in per_score]))
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
        block = int(np.clip(_BLOCK_ELEMS // max(n_samples, 1), 64, 8192))
    block = max(1, int(block))

    for start in range(0, m, block):
        stop = min(start + block, m)
        cols = union[start:stop]
        if is_bgen:
            d = np.asarray(dosage[:, cols], dtype=float)
            miss = ~np.isfinite(d)
        else:
            raw = read_bed(strip_ext(str(plink)) + ".bed", n_samples, n_total,
                           variant_idx=cols)
            d = np.asarray(raw, dtype=float)
            miss = d < 0
        use_target = standardize and frozen_ord is None
        block_matrix, mean_b, sd_b = _prepare_block(d, miss, use_target)
        af[start:stop] = mean_b / 2.0
        sd[start:stop] = sd_b
        if frozen_ord is None:
            _accumulate_block(scores, block_matrix, mapped, start, stop)
        else:
            _accumulate_frozen(scores, block_matrix, mapped, frozen_ord,
                               start, stop)
    return scores, union, af, sd


def _accumulate_frozen(scores, dosage, mapped, frozen, start, stop):
    """Apply per-score frozen ``(g - 2 AF_REF) / SD_REF`` weights to a block."""
    for s, (pos, w) in enumerate(mapped):
        lo = int(np.searchsorted(pos, start))
        hi = int(np.searchsorted(pos, stop))
        if hi <= lo:
            continue
        g = dosage[:, pos[lo:hi] - start]
        scale = frozen[s]
        if scale is None:
            scores[:, s] += g @ w[lo:hi]
            continue
        fa, fs = scale[0][lo:hi], scale[1][lo:hi]
        good = fs > 1e-12
        z = np.zeros_like(g)
        if np.any(good):
            z[:, good] = (g[:, good] - 2.0 * fa[good]) / fs[good]
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
        d[miss] = np.nan
        col_mean = np.nanmean(d, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        d = np.where(np.isnan(d), col_mean, d)
    mean = d.mean(axis=0)
    sd = d.std(axis=0)
    if not standardize:
        return d, mean, sd
    out = d - mean
    good = sd > 1e-12
    out[:, good] /= sd[good]
    out[:, ~good] = 0.0
    return out, mean, sd


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
        Path for the LD cache. It is **built by the first trait and reused by
        the rest**, which is the whole reason to fit a panel in one call rather
        than in a loop. Traits must therefore share a variant set; that is the
        normal case when the target genotypes are fixed and
        ``subset_to_sumstats`` is left off.
    ld_prefix : str, optional
        External LD-reference PLINK prefix, forwarded to
        :func:`ldpred3.run_ldpred3_prs`. Prefer this (or an existing
        ``ld_cache``) over building LD from the target cohort.
    on_error : {"raise", "skip"}
    progress : callable, optional
        ``progress(i, n, score_id)`` before each fit.
    n_jobs : int, default 1
        Independent traits after the LD cache exists. The first successful fit
        always runs alone so it can write ``ld_cache``; the remainder run in a
        thread pool of this size. ``1`` is sequential.
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
    Reusing one LD cache across traits requires ``subset_to_sumstats=False``
    (the LD blocks must span the same variants for every trait). This function
    sets that default when ``ld_cache`` is given; override it explicitly if you
    know what you are doing, and expect the cache to be rebuilt per trait if
    the variant set moves.
    """
    from ldpred3 import run_ldpred3_prs

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
        paths = [str(p) for p in _expand_paths(sumstats)]
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

    kwargs = dict(ldpred3_kwargs)
    if ld_prefix is not None:
        kwargs["ld_prefix"] = ld_prefix
    if ld_cache is None and kwargs.get("ld_prefix") is None:
        warnings.warn(
            "panel_from_sumstats is building LD from the target genotypes; "
            "pass ld_prefix= or an existing ld_cache= so the LD reference is "
            "not the cohort you will score", stacklevel=2)
    if ld_cache is not None:
        kwargs.setdefault("subset_to_sumstats", False)
        kwargs["ld_cache"] = ld_cache
        if not os.path.exists(str(ld_cache)):
            kwargs["ld_out"] = ld_cache
    if n_jobs > 1 and int(kwargs.get("ncores", 1) or 1) > 1:
        warnings.warn(
            "n_jobs>1 with ncores>1 oversubscribes the machine; leave "
            "ncores=1 when running traits concurrently", stacklevel=2)
    if weights_dir is not None:
        os.makedirs(str(weights_dir), exist_ok=True)
    if preflight:
        _preflight_traits(items, plink, kwargs, on_error=on_error)

    columns, ids, metas, weight_tables = [], [], [], []
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
        if weights_dir is not None:
            dest = os.path.join(str(weights_dir), f"{sid}.weights")
            writer = getattr(res, "write_weights", None)
            if writer is not None:
                writer(dest)
            else:
                from ldpred3.weights import write_weights
                write_weights(
                    dest, id=res.variant_id, chrom=res.chrom, pos=res.pos,
                    effect_allele=res.effect_allele,
                    other_allele=res.other_allele, weight=res.beta_adjusted,
                    af=res.af, sd=res.sd)
            meta["weights"] = dest

    def _fit(sid, path, trait):
        try:
            return (sid, path, trait,
                    run_ldpred3_prs(str(path), plink,
                                    **_trait_kwargs(kwargs, trait)), None)
        except Exception as exc:
            return sid, path, trait, None, exc

    remaining = list(enumerate(items))
    # The first successful trait writes the LD cache; later traits only read it.
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
        kwargs.pop("ld_out", None)
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

    if not columns:
        raise ValueError(f"all {len(items)} LDpred3 fits failed")
    K = len(columns)
    return ScorePanel(
        scores=np.column_stack(columns), sample_fid=np.asarray(fid),
        sample_iid=np.asarray(iid), score_ids=np.array(ids, dtype=object),
        standardized=np.ones(K, dtype=bool), weights=weight_tables,
        meta=metas,
        log={"source": "ldpred3", "n_scores": K, "n_failed": n_failed,
             "n_jobs": int(n_jobs),
             "target": str(plink),
             "ld_cache": str(ld_cache) if ld_cache else None,
             "ld_prefix": str(ld_prefix) if ld_prefix else None,
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
    :func:`ldpred3.score_from_weights` ``scaling="frozen"``.
    """
    from ldpred3.genotype_io import read_bed, read_bim, read_fam, strip_ext
    from ldpred3.harmonize import harmonize
    from ldpred3.sumstats import Sumstats
    from ldpred3.weights import read_weights

    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    files = _expand_paths(paths)
    if not files:
        raise ValueError(f"no weight files found in {paths!r}")

    is_bgen = str(plink).endswith(".bgen")
    if is_bgen:
        from ldpred3 import load_genotypes
        geno = load_genotypes(plink, sample_path=sample_path)
        variants = geno.variants
        fid, iid = geno.samples.fid, geno.samples.iid
        dosage = geno.dosage
        n_samples, n_total = len(fid), len(variants.id)
    else:
        prefix = strip_ext(str(plink))
        variants = read_bim(prefix + ".bim")
        samples = read_fam(prefix + ".fam")
        fid, iid = samples.fid, samples.iid
        n_samples, n_total = len(fid), len(variants.id)
        dosage = None

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
        h = harmonize(ss, variants, drop_ambiguous=drop_ambiguous)
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

    if not per_score:
        raise ValueError(f"none of the {len(files)} weight files produced a "
                         f"usable score ({n_failed} unreadable, {n_skipped} "
                         f"below min_matched)")

    scores, union, af, sd = _accumulate_scores(
        per_score, n_samples, n_total, plink, dosage, standardize=False,
        block=block, read_bed=read_bed, strip_ext=strip_ext, is_bgen=is_bgen,
        frozen=frozen)
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


def check_weights(panel, fit, path, plink, *, min_corr=1.0 - 1e-8):
    """Require that scoring ``path`` with frozen scaling reproduces the fit.

    Aligns individuals on ``FID:IID``. Returns ``{"corr", "n"}``.
    """
    from ldpred3 import score_from_weights

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
    corr = float(np.corrcoef(direct[keep],
                             np.asarray(scored.scores, dtype=float)[idx])[0, 1])
    if not np.isfinite(corr) or corr < float(min_corr):
        raise ValueError(
            f"frozen scoring of {path} correlates {corr:.8f} with the fitted "
            f"combination (required >= {float(min_corr):.8f})")
    return {"corr": corr, "n": len(keep)}


# ---------------------------------------------------------------------------
# Folding a fitted stack back into one weight vector
# ---------------------------------------------------------------------------

def combine_weights(panel, fit, *, path=None):
    """Collapse a panel and its stacking coefficients into one weight table.

    The combined weight of variant ``v`` is ``sum_k beta_k * w_kv`` over the
    scores containing it, after putting every ``w_kv`` on one common scale and
    orienting every score to one reference allele. This is the deployable
    artefact: a new cohort is scored from it directly, with no reference to the
    ``K`` inputs and no need to rebuild the panel.

    The common scale is LDpred3's **standardized** one — weights apply to
    ``(g - 2f)/sd`` — because that is what :func:`ldpred3.score_from_weights`
    applies, so the file this writes can be handed straight to it::

        multipgs.combine_weights(panel, fit, path="multi.weights")
        ldpred3.score_from_weights("multi.weights", "new_cohort",
                                   scaling="frozen")

    Allele-count weights (everything from the PGS Catalog) are multiplied by the
    panel cohort's per-variant dosage SD to get there, so the combined score
    reproduces the training-time score up to an additive constant — irrelevant
    to R², AUC and ranking, and absorbed by the intercept of any downstream
    regression. ``scaling="frozen"`` is the mode to score with: it reuses the
    ``AF_REF``/``SD_REF`` written here, so a cohort with different allele
    frequencies is still scored on the scale the coefficients were fitted on.

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
        metadata, index = _weight_table_storage(table)
        w = np.asarray(table["weight"], dtype=float)
        sd = metadata.get("sd")
        sd = None if sd is None else np.asarray(sd, dtype=float)
        af = metadata.get("af")
        af = None if af is None else np.asarray(af, dtype=float)
        if not standardized[k]:
            if sd is None:
                raise ValueError(
                    f"score {panel_ids[k]!r} has allele-count weights but no "
                    f"per-variant SD, so it cannot be put on the standardized "
                    f"scale. Rebuild the panel with panel_from_catalog, which "
                    f"records the target cohort's SD.")
            w = w * (sd if index is None else sd[index])
        ids = np.asarray(metadata["id"], dtype=object)
        chrom = np.asarray(metadata["chrom"], dtype=object)
        pos = np.asarray(metadata["pos"])
        a1 = np.asarray(metadata["a1"], dtype=object)
        a2 = np.asarray(metadata["a2"], dtype=object)
        for j in range(w.size):
            if w[j] == 0.0:
                continue
            q = j if index is None else int(index[j])
            e1, e2 = str(a1[q]).upper(), str(a2[q]).upper()
            # The unordered allele pair distinguishes multiallelic variants at
            # one coordinate while still coalescing A/G with its G/A swap.
            pair = tuple(sorted((e1, e2)))
            key = (str(chrom[q]), int(pos[q]), pair[0], pair[1])
            a_j = np.nan if af is None else float(af[q])
            s_j = np.nan if sd is None else float(sd[q])
            entry = acc.get(key)
            if entry is None:
                acc[key] = [str(ids[q]), e1, e2, b * w[j], a_j, s_j]
                continue
            if e1 == entry[1] and e2 == entry[2]:
                entry[3] += b * w[j]
            elif e1 == entry[2] and e2 == entry[1]:
                # Counts the other allele: flip the weight, and the frequency
                # with it, before adding.
                entry[3] -= b * w[j]
                a_j = np.nan if not np.isfinite(a_j) else 1.0 - a_j
            if not np.isfinite(entry[4]) and np.isfinite(a_j):
                entry[4] = a_j
            if not np.isfinite(entry[5]) and np.isfinite(s_j):
                entry[5] = s_j

    if not acc:
        raise ValueError("every selected score contributed no non-zero weight; "
                         "the fit is null, so there is nothing to deploy")
    keys = sorted(acc, key=lambda kv: (str(kv[0]), kv[1], kv[2], kv[3]))
    out = {
        "id": np.array([acc[k][0] for k in keys], dtype=object),
        "chrom": np.array([k[0] for k in keys], dtype=object),
        "pos": np.array([k[1] for k in keys], dtype=np.int64),
        "a1": np.array([acc[k][1] for k in keys], dtype=object),
        "a2": np.array([acc[k][2] for k in keys], dtype=object),
        "weight": np.array([acc[k][3] for k in keys], dtype=float),
        "af": np.array([acc[k][4] for k in keys], dtype=float),
        "sd": np.array([acc[k][5] for k in keys], dtype=float),
    }
    keep = out["weight"] != 0.0
    if not np.any(keep):
        raise ValueError("all combined variant weights cancel exactly; there "
                         "is no non-zero weight set to deploy")
    out = {k: v[keep] for k, v in out.items()}
    if path is not None:
        from ldpred3.weights import write_weights
        complete = bool(np.all(np.isfinite(out["af"]))
                        and np.all(np.isfinite(out["sd"])))
        write_weights(path, id=out["id"], chrom=out["chrom"], pos=out["pos"],
                      effect_allele=out["a1"], other_allele=out["a2"],
                      weight=out["weight"],
                      af=out["af"] if complete else None,
                      sd=out["sd"] if complete else None)
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
    """Read a panel written by :func:`write_panel` or :func:`save_panel`."""
    if str(path).endswith(".npz"):
        return load_panel(path)
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
        log={"source": str(path)})


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
    np.savez_compressed(path, **payload)
    return path


def load_panel(path):
    """Load a panel written by :func:`save_panel`."""
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
