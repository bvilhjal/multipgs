"""Genetic-correlation screen against a shared ldpred3 LD cache.

Optional: requires :mod:`bipred`. Catalog-only workflows never import this
module. The LDSC chi-square cap is applied to regression rows only -- never to
a joint-fit or stacking variant set.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


__all__ = ["align_sumstats_to_cache", "ldsc_rg_screen", "RgScreen"]


@contextmanager
def _cache_contents(ld_cache):
    """Borrow a prepared cache or own one ordinary path-based load."""
    from ldpred3.interop import PreparedLDCache, load_ld_blocks

    if isinstance(ld_cache, PreparedLDCache):
        if ld_cache.closed:
            raise ValueError("prepared ld_cache is closed")
        yield ld_cache.blocks, ld_cache.variant_ids, ld_cache.metadata
        return
    blocks, ids, meta = load_ld_blocks(ld_cache, return_metadata=True)
    try:
        yield blocks, ids, meta
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()


def _require_bipred():
    try:
        import bipred  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ldsc_rg_screen needs bipred; install multipgs[bipred] or "
            "the sibling bipred checkout") from exc


def _cache_variants(ids, meta):
    """Validate cache provenance and return LDpred3's variant data model.

    Harmonisation deliberately receives the real :class:`VariantTable`, not a
    duck-typed namespace.  Besides making the public contract explicit, this
    matters because LDpred3 validates the table with ``len(variants)`` before
    touching its columns.
    """
    if not hasattr(meta, "get"):
        raise ValueError("ld_cache metadata is not a mapping")
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("ld_cache variant IDs must be a non-empty vector")
    fields = {}
    for key in ("counted_allele", "other_allele", "chrom", "pos"):
        values = meta.get(key)
        if values is None:
            raise ValueError(
                "ld_cache lacks allele/coordinate provenance; rebuild it "
                "with a current ldpred3 ld_out=")
        values = np.asarray(values)
        if values.ndim != 1 or values.size != ids.size:
            raise ValueError(
                f"ld_cache metadata {key!r} has length {values.size}, "
                f"expected {ids.size}")
        fields[key] = values
    from ldpred3.interop import VariantTable
    return VariantTable(
        id=ids, chrom=fields["chrom"], pos=fields["pos"],
        cm=np.zeros(ids.size, dtype=float),
        a1=fields["counted_allele"], a2=fields["other_allele"])


def _align_sumstats(sumstats, variants, *, n_eff=None, qc=True):
    """Align one GWAS to an already validated cache variant map."""
    from ldpred3.interop import (
        harmonize,
        qc_sumstats,
        read_sumstats,
        standardize_betas,
    )

    ss = read_sumstats(sumstats, n_eff=n_eff)
    qc_log = {}
    if qc:
        keep, qc_log = qc_sumstats(ss)
        ss = ss.subset(keep)
    h = harmonize(ss, variants, drop_ambiguous=True)
    m = int(variants.id.size)
    beta = np.full(m, np.nan)
    n_vec = np.full(m, np.nan)
    if len(h):
        index = np.asarray(h.var_index)
        if index.ndim != 1 or np.any(index < 0) or np.any(index >= m):
            raise ValueError("harmonized variant index is outside ld_cache")
        std, _ = standardize_betas(h.beta, h.se, h.n_eff)
        beta[index] = std
        n_vec[index] = np.asarray(h.n_eff, dtype=float)
    log = {"n_cache": m, "n_matched": int(len(h)), "qc": qc_log,
           "harmonize": dict(h.log)}
    return beta, n_vec, log


def align_sumstats_to_cache(sumstats, ld_cache, *, n_eff=None, qc=True):
    """Standardized ``beta_hat`` in ``ld_cache`` variant order.

    Unmatched or QC-dropped variants are ``nan``. Returns
    ``(beta_hat, n_eff_vector, log)``.
    """
    with _cache_contents(ld_cache) as (_blocks, ids, meta):
        variants = _cache_variants(ids, meta)
        return _align_sumstats(sumstats, variants, n_eff=n_eff, qc=qc)


@dataclass
class RgScreen:
    """One focal GWAS against a list of auxiliaries."""

    score_ids: np.ndarray
    rg: np.ndarray
    rg_se: np.ndarray
    overlap_corr: np.ndarray
    overlap_valid: np.ndarray
    n_used: np.ndarray
    log: dict

    def summary(self):
        finite = np.isfinite(self.rg)
        lines = [f"rg screen: {int(finite.sum())} of {self.rg.size} "
                 f"auxiliaries estimated"]
        if finite.any():
            lines.append(
                f"  |rg| median {np.median(np.abs(self.rg[finite])):.3f}, "
                f"max {np.max(np.abs(self.rg[finite])):.3f}")
        n_overlap = int(np.count_nonzero(
            np.isfinite(self.overlap_corr) & (np.abs(self.overlap_corr) > 0.05)))
        if n_overlap:
            lines.append(f"  {n_overlap} pair(s) have |overlap_corr| > 0.05")
        return "\n".join(lines)


def ldsc_rg_screen(focal, auxiliaries, ld_cache, *, n_eff_focal=None,
                   n_eff=None, qc=True, min_snps=10_000,
                   exclude_long_range_ld=False):
    """``ldsc_rg`` of ``focal`` vs each auxiliary on one ldpred3 cache.

    ``auxiliaries`` is a mapping ``score_id -> path`` or a sequence of
    ``(score_id, path)``. ``n_eff`` is an optional mapping of auxiliary
    effective sample sizes. The LDSC chi-square cap is applied to regression
    rows only; it does not change the model's ``m_snps``.

    ``min_snps`` is a computational safeguard, not evidence that an estimate
    is scientifically reliable; LDSC needs a genome-wide, well-QCed marker
    set. ``exclude_long_range_ld=True`` excludes bipred's standard hg19/GRCh37
    long-range-LD regions (including MHC and APOE). Leave it false for another
    genome build and preprocess coordinates with a build-appropriate mask.
    """
    _require_bipred()
    if (isinstance(min_snps, (bool, np.bool_))
            or not isinstance(min_snps, (int, np.integer))
            or int(min_snps) < 2):
        raise ValueError("min_snps must be an integer >= 2")
    min_snps = int(min_snps)
    if not isinstance(exclude_long_range_ld, (bool, np.bool_)):
        raise ValueError("exclude_long_range_ld must be boolean")

    from bipred import estimate_sample_overlap, ldsc_chi2_mask, ldsc_rg
    from ldpred3 import ld_scores

    if hasattr(auxiliaries, "items"):
        pairs = list(auxiliaries.items())
    else:
        pairs = list(auxiliaries)
    n_eff_map = {} if n_eff is None else {
        str(key): float(value) for key, value in dict(n_eff).items()}

    with _cache_contents(ld_cache) as (blocks, ids, meta):
        variants = _cache_variants(ids, meta)
        ell = np.asarray(ld_scores(blocks), dtype=float)
        if ell.ndim != 1 or ell.size != variants.id.size:
            raise ValueError("ld_scores length does not match ld_cache")
        b1, n1, log1 = _align_sumstats(
            focal, variants, n_eff=n_eff_focal, qc=qc)

        if exclude_long_range_ld:
            from bipred import in_long_range_ld
            long_range = np.asarray(
                in_long_range_ld(variants.chrom, variants.pos), dtype=bool)
            if long_range.shape != ell.shape:
                raise ValueError("long-range-LD mask does not match ld_cache")
        else:
            long_range = np.zeros(ell.size, dtype=bool)
        # M is the reference marker count. Long-range-LD exclusion changes
        # regression rows, not that LDSC scaling denominator.
        m_snps = int(variants.id.size)

        rgs, ses, overlap, valid, n_used = [], [], [], [], []
        logs = {
            "focal": log1,
            "aux": {},
            "screen": {
                "min_snps": min_snps,
                "exclude_long_range_ld": bool(exclude_long_range_ld),
                "n_long_range_excluded": int(np.count_nonzero(long_range)),
                "m_snps": m_snps,
            },
        }

        def missing(count):
            rgs.append(np.nan)
            ses.append(np.nan)
            overlap.append(np.nan)
            valid.append(False)
            n_used.append(int(count))

        for sid, path in pairs:
            sid = str(sid)
            b2, n2, log2 = _align_sumstats(
                path, variants, n_eff=n_eff_map.get(sid), qc=qc)
            logs["aux"][sid] = log2
            keep = (~long_range & np.isfinite(b1) & np.isfinite(b2)
                    & np.isfinite(ell) & np.isfinite(n1) & np.isfinite(n2)
                    & (np.abs(b1) < 1.0) & (np.abs(b2) < 1.0)
                    & (ell > 0))
            if int(keep.sum()) < min_snps:
                missing(keep.sum())
                continue
            cap = (ldsc_chi2_mask(b1[keep], n1[keep])
                   & ldsc_chi2_mask(b2[keep], n2[keep]))
            rows = np.flatnonzero(keep)[cap]
            if rows.size < min_snps:
                missing(rows.size)
                continue
            try:
                fit = ldsc_rg(
                    b1[rows], b2[rows], ell[rows], n1[rows], n2[rows],
                    m_snps=m_snps)
                ov = estimate_sample_overlap(
                    fit, float(np.nanmedian(n1[rows])),
                    float(np.nanmedian(n2[rows])))
            except Exception as exc:
                logs["aux"][sid]["error"] = str(exc)
                missing(rows.size)
                continue
            rgs.append(float(fit.rg))
            se = getattr(fit, "rg_se", np.nan)
            ses.append(float(se) if se is not None and np.isfinite(se)
                       else float("nan"))
            overlap.append(float(ov.get("overlap_corr", np.nan)))
            valid.append(bool(ov.get("cross_corr_valid", False)))
            n_used.append(int(rows.size))

        result = RgScreen(
            score_ids=np.array([str(s) for s, _ in pairs], dtype=object),
            rg=np.asarray(rgs, dtype=float),
            rg_se=np.asarray(ses, dtype=float),
            overlap_corr=np.asarray(overlap, dtype=float),
            overlap_valid=np.asarray(valid, dtype=bool),
            n_used=np.asarray(n_used, dtype=int), log=logs)
    return result
