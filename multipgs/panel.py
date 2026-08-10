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

import os
from dataclasses import dataclass, field

import numpy as np


__all__ = ["ScorePanel", "panel_from_catalog", "panel_from_sumstats",
           "combine_weights", "read_panel", "write_panel"]

# Genotype elements held in memory per streamed block. 8e6 float64 is 64 MB.
_BLOCK_ELEMS = 8_000_000


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class ScorePanel:
    """Scores for ``n`` individuals across ``K`` polygenic scores.

    ``weights`` holds one dict per score with keys ``id chrom pos a1 a2 weight``
    and optionally ``sd``/``af``, where ``a1`` is the allele the weight counts.
    ``standardized[k]`` says whether score ``k``'s weights apply to standardized
    genotypes (LDpred3) or raw allele counts (PGS Catalog).
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
        idx = np.asarray(idx, dtype=int)
        return ScorePanel(
            scores=self.scores[idx], sample_fid=np.asarray(self.sample_fid)[idx],
            sample_iid=np.asarray(self.sample_iid)[idx],
            score_ids=self.score_ids, standardized=self.standardized,
            weights=self.weights, meta=self.meta, log=self.log)

    def summary(self):
        n_dead = int(np.sum(np.std(self.scores, axis=0) <= 0))
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
        for key in ("n_failed", "n_skipped"):
            if self.log.get(key):
                lines.append(f"  {key.replace('_', ' ')}: {self.log[key]}")
        return "\n".join(lines)


def _keys(fid, iid):
    return [f"{f}:{i}" for f, i in zip(np.asarray(fid).astype(str),
                                       np.asarray(iid).astype(str))]


# ---------------------------------------------------------------------------
# From PGS Catalog scoring files
# ---------------------------------------------------------------------------

def panel_from_catalog(paths, plink, *, sample_path=None, drop_ambiguous=True,
                       standardize=False, min_matched=1, on_error="raise",
                       block=None, prefer_harmonized=True, progress=None):
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

    score_ids, metas, weight_tables, per_score = [], [], [], []
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
        weight_tables.append({
            "id": np.asarray(variants.id)[var_index],
            "chrom": np.asarray(variants.chrom)[var_index],
            "pos": np.asarray(variants.pos)[var_index],
            "a1": np.asarray(variants.a1)[var_index],
            "a2": np.asarray(variants.a2)[var_index],
            "weight": w,
        })
        per_score.append((var_index, w))

    if not per_score:
        raise ValueError(f"none of the {len(files)} scoring files produced a "
                         f"usable score ({n_failed} unreadable, {n_skipped} "
                         f"below min_matched)")

    scores, union, af, sd = _accumulate_scores(
        per_score, n_samples, n_total, plink, dosage, standardize=standardize,
        block=block, read_bed=read_bed, strip_ext=strip_ext, is_bgen=is_bgen)

    # Carry the target cohort's allele frequency and dosage SD per matched
    # variant. combine_weights needs the SD to put allele-count weights onto
    # the standardized scale that ldpred3's scorer applies, and without it a
    # catalog panel could be fitted but never deployed.
    for table, (var_index, _) in zip(weight_tables, per_score):
        at = np.searchsorted(union, var_index)
        table["af"] = af[at]
        table["sd"] = sd[at]

    K = len(per_score)
    return ScorePanel(
        scores=scores, sample_fid=np.asarray(fid), sample_iid=np.asarray(iid),
        score_ids=np.array(score_ids, dtype=object),
        standardized=np.full(K, bool(standardize)),
        weights=weight_tables, meta=metas,
        log={"source": "pgs_catalog", "n_files": len(files), "n_scores": K,
             "n_failed": n_failed, "n_skipped": n_skipped,
             "target": str(plink), "standardize": bool(standardize),
             "drop_ambiguous": bool(drop_ambiguous)})


def _expand_paths(paths):
    """One path, a directory, or a mix of both, to a flat list of files."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = []
    for entry in paths:
        p = str(entry)
        if os.path.isdir(p):
            out.extend(sorted(
                os.path.join(p, f) for f in os.listdir(p)
                if f.endswith((".txt", ".txt.gz", ".tsv", ".tsv.gz"))))
        else:
            out.append(p)
    return out


def _accumulate_scores(per_score, n_samples, n_total, plink, dosage, *,
                       standardize, block, read_bed, strip_ext, is_bgen):
    """One pass over the genotypes: ``K`` scores, plus per-variant AF and SD."""
    K = len(per_score)
    union = np.unique(np.concatenate([vi for vi, _ in per_score]))
    m = union.size
    # Per score: positions into `union`, ascending, with matching weights.
    mapped = []
    for var_index, w in per_score:
        pos = np.searchsorted(union, var_index)
        order = np.argsort(pos, kind="mergesort")
        mapped.append((pos[order], np.asarray(w)[order]))

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
        block_matrix, mean_b, sd_b = _prepare_block(d, miss, standardize)
        af[start:stop] = mean_b / 2.0
        sd[start:stop] = sd_b
        for s, (pos, w) in enumerate(mapped):
            lo = np.searchsorted(pos, start)
            hi = np.searchsorted(pos, stop)
            if hi > lo:
                scores[:, s] += block_matrix[:, pos[lo:hi] - start] @ w[lo:hi]
    return scores, union, af, sd


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
                        on_error="raise", progress=None, **ldpred3_kwargs):
    """Fit one LDpred3 model per GWAS and score them all on one target.

    Parameters
    ----------
    sumstats : sequence of str, or mapping
        Summary-statistic files. A mapping's keys become the score ids.
    plink : str
        Target genotypes (PLINK prefix or ``.bgen``).
    ld_cache : str, optional
        Path for the LD cache. It is **built by the first trait and reused by
        the rest**, which is the whole reason to fit a panel in one call rather
        than in a loop. Traits must therefore share a variant set; that is the
        normal case when the target genotypes are fixed and
        ``subset_to_sumstats`` is left off.
    on_error : {"raise", "skip"}
    progress : callable, optional
        ``progress(i, n, score_id)`` before each fit.
    **ldpred3_kwargs
        Passed through to :func:`ldpred3.run_ldpred3_prs` (``method``,
        ``n_eff``, ``block_size``, QC options, ...).

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
    if hasattr(sumstats, "items"):
        items = list(sumstats.items())
    else:
        paths = [str(p) for p in _expand_paths(sumstats)]
        if score_ids is None:
            items = [(_stem(p), p) for p in paths]
        else:
            ids = list(score_ids)
            if len(ids) != len(paths):
                raise ValueError(f"score_ids has {len(ids)} entries for "
                                 f"{len(paths)} sumstats files")
            items = list(zip(ids, paths))
    if not items:
        raise ValueError("no summary-statistic files given")

    kwargs = dict(ldpred3_kwargs)
    if ld_cache is not None:
        kwargs.setdefault("subset_to_sumstats", False)
        kwargs["ld_cache"] = ld_cache
        if not os.path.exists(str(ld_cache)):
            kwargs["ld_out"] = ld_cache

    columns, ids, metas, weight_tables = [], [], [], []
    fid = iid = None
    n_failed = 0
    for i, (sid, path) in enumerate(items):
        if progress is not None:
            progress(i, len(items), sid)
        try:
            res = run_ldpred3_prs(str(path), plink, **kwargs)
        except Exception as exc:
            if on_error == "raise":
                raise RuntimeError(f"LDpred3 failed on {sid} ({path}): "
                                   f"{exc}") from exc
            n_failed += 1
            continue
        # After the first successful fit the cache exists; stop asking for it
        # to be written again.
        kwargs.pop("ld_out", None)
        if fid is None:
            fid, iid = res.sample_fid, res.sample_iid
        elif not np.array_equal(np.asarray(res.sample_iid), np.asarray(iid)):
            raise RuntimeError(f"{sid} was scored on a different sample order "
                               f"than the earlier traits")
        columns.append(np.asarray(res.scores, dtype=float))
        ids.append(sid)
        metas.append({"path": str(path), "n_matched": int(res.var_index.size),
                      "harmonize_log": dict(res.harmonize_log),
                      "qc_log": dict(res.qc_log),
                      "inference": dict(res.inference)})
        weight_tables.append({
            "id": np.asarray(res.variant_id), "chrom": np.asarray(res.chrom),
            "pos": np.asarray(res.pos), "a1": np.asarray(res.effect_allele),
            "a2": np.asarray(res.other_allele),
            "weight": np.asarray(res.beta_adjusted, dtype=float),
            "af": None if res.af is None else np.asarray(res.af, dtype=float),
            "sd": None if res.sd is None else np.asarray(res.sd, dtype=float),
        })

    if not columns:
        raise ValueError(f"all {len(items)} LDpred3 fits failed")
    K = len(columns)
    return ScorePanel(
        scores=np.column_stack(columns), sample_fid=np.asarray(fid),
        sample_iid=np.asarray(iid), score_ids=np.array(ids, dtype=object),
        standardized=np.ones(K, dtype=bool), weights=weight_tables,
        meta=metas,
        log={"source": "ldpred3", "n_scores": K, "n_failed": n_failed,
             "target": str(plink), "ld_cache": str(ld_cache) if ld_cache
             else None})


def _stem(path):
    base = os.path.basename(str(path))
    for suffix in (".tsv.gz", ".txt.gz", ".gz", ".tsv", ".txt", ".sumstats"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


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
    fit : MultiPGSFit or MetaPGS
        Its ``score_ids`` must match the panel's, in order.
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
    if beta.size != panel.n_scores:
        raise ValueError(f"fit has {beta.size} coefficients but the panel has "
                         f"{panel.n_scores} scores")
    panel_ids = [str(s) for s in np.asarray(panel.score_ids, dtype=object)]
    fit_ids = [str(s) for s in np.asarray(fit.score_ids, dtype=object)]
    if fit_ids != panel_ids:
        raise ValueError("the fit's score_ids do not match the panel's; "
                         "combining them would attach coefficients to the "
                         "wrong scores")

    acc = {}
    for k, table in enumerate(panel.weights):
        b = beta[k]
        if b == 0.0:
            continue
        w = np.asarray(table["weight"], dtype=float)
        sd = table.get("sd")
        sd = None if sd is None else np.asarray(sd, dtype=float)
        af = table.get("af")
        af = None if af is None else np.asarray(af, dtype=float)
        if not panel.standardized[k]:
            if sd is None:
                raise ValueError(
                    f"score {panel_ids[k]!r} has allele-count weights but no "
                    f"per-variant SD, so it cannot be put on the standardized "
                    f"scale. Rebuild the panel with panel_from_catalog, which "
                    f"records the target cohort's SD.")
            w = w * sd
        ids = np.asarray(table["id"], dtype=object)
        chrom = np.asarray(table["chrom"], dtype=object)
        pos = np.asarray(table["pos"])
        a1 = np.asarray(table["a1"], dtype=object)
        a2 = np.asarray(table["a2"], dtype=object)
        for j in range(w.size):
            if w[j] == 0.0:
                continue
            key = (str(chrom[j]), int(pos[j]))
            e1, e2 = str(a1[j]).upper(), str(a2[j]).upper()
            a_j = np.nan if af is None else float(af[j])
            s_j = np.nan if sd is None else float(sd[j])
            entry = acc.get(key)
            if entry is None:
                acc[key] = [str(ids[j]), e1, e2, b * w[j], a_j, s_j]
                continue
            if e1 == entry[1] and e2 == entry[2]:
                entry[3] += b * w[j]
            elif e1 == entry[2] and e2 == entry[1]:
                # Counts the other allele: flip the weight, and the frequency
                # with it, before adding.
                entry[3] -= b * w[j]
                a_j = np.nan if not np.isfinite(a_j) else 1.0 - a_j
            else:
                # A third allele pair at the same coordinate is a different
                # variant. Dropping it beats adding a weight for an allele the
                # target may not carry.
                continue
            if not np.isfinite(entry[4]) and np.isfinite(a_j):
                entry[4] = a_j
            if not np.isfinite(entry[5]) and np.isfinite(s_j):
                entry[5] = s_j

    if not acc:
        raise ValueError("every selected score contributed no non-zero weight; "
                         "the fit is null, so there is nothing to deploy")
    keys = sorted(acc, key=lambda kv: (str(kv[0]), kv[1]))
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
        fh.write("FID\tIID\t" + "\t".join(ids) + "\n")
        for i in range(len(panel)):
            row = "\t".join(f"{v:.6g}" for v in panel.scores[i])
            fh.write(f"{panel.sample_fid[i]}\t{panel.sample_iid[i]}\t{row}\n")
    return path


def read_panel(path):
    """Read a panel written by :func:`write_panel`."""
    fid, iid, rows = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if len(header) < 3 or header[0] != "FID" or header[1] != "IID":
            raise ValueError(f"{path}: expected a 'FID\\tIID\\t...' header")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            fid.append(parts[0])
            iid.append(parts[1])
            rows.append([float(v) for v in parts[2:]])
    scores = np.array(rows, dtype=float)
    K = scores.shape[1] if scores.size else len(header) - 2
    return ScorePanel(
        scores=scores, sample_fid=np.array(fid, dtype=object),
        sample_iid=np.array(iid, dtype=object),
        score_ids=np.array(header[2:], dtype=object),
        standardized=np.zeros(K, dtype=bool), weights=[], meta=[],
        log={"source": str(path)})
