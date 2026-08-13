"""Genetic-correlation screen against a shared ldpred3 LD cache.

Optional: requires :mod:`bipred`. Catalog-only workflows never import this
module. The LDSC χ² cap is applied to regression rows only — never to a
joint-fit or stacking variant set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["align_sumstats_to_cache", "ldsc_rg_screen", "RgScreen"]


def _require_bipred():
    try:
        import bipred  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ldsc_rg_screen needs bipred; install multipgs[bipred] or "
            "the sibling bipred checkout") from exc


def align_sumstats_to_cache(sumstats, ld_cache, *, n_eff=None, qc=True):
    """Standardized ``beta_hat`` in ``ld_cache`` variant order.

    Unmatched or QC-dropped variants are ``nan``. Returns
    ``(beta_hat, n_eff_vector, log)``.
    """
    from types import SimpleNamespace

    from ldpred3 import standardize_betas
    from ldpred3.harmonize import harmonize
    from ldpred3.ld import load_ld_blocks
    from ldpred3.qc import qc_sumstats
    from ldpred3.sumstats import read_sumstats

    loaded = load_ld_blocks(ld_cache, return_metadata=True)
    blocks, ids, meta = loaded
    try:
        counted = meta.get("counted_allele")
        other = meta.get("other_allele")
        chrom = meta.get("chrom")
        pos = meta.get("pos")
        if counted is None or other is None or chrom is None or pos is None:
            raise ValueError(
                "ld_cache lacks allele/coordinate provenance; rebuild it "
                "with a current ldpred3 ld_out=")
        variants = SimpleNamespace(
            id=np.asarray(ids), chrom=np.asarray(chrom),
            pos=np.asarray(pos), a1=np.asarray(counted),
            a2=np.asarray(other))
        ss = read_sumstats(sumstats, n_eff=n_eff)
        qc_log = {}
        if qc:
            keep, qc_log = qc_sumstats(ss)
            ss = ss.subset(keep)
        h = harmonize(ss, variants, drop_ambiguous=True)
        m = len(ids)
        beta = np.full(m, np.nan)
        n_vec = np.full(m, np.nan)
        if len(h):
            std, _ = standardize_betas(h.beta, h.se, h.n_eff)
            beta[np.asarray(h.var_index)] = std
            n_vec[np.asarray(h.var_index)] = np.asarray(h.n_eff, dtype=float)
        log = {"n_cache": m, "n_matched": int(len(h)), "qc": qc_log,
               "harmonize": dict(h.log)}
        return beta, n_vec, log
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()


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
            lines.append(f"  |rg| median {np.median(np.abs(self.rg[finite])):.3f}, "
                         f"max {np.max(np.abs(self.rg[finite])):.3f}")
        n_overlap = int(np.count_nonzero(
            np.isfinite(self.overlap_corr) & (np.abs(self.overlap_corr) > 0.05)))
        if n_overlap:
            lines.append(f"  {n_overlap} pair(s) have |overlap_corr| > 0.05")
        return "\n".join(lines)


def ldsc_rg_screen(focal, auxiliaries, ld_cache, *, n_eff_focal=None,
                   n_eff=None, qc=True):
    """``ldsc_rg`` of ``focal`` vs each auxiliary on one ldpred3 cache.

    ``auxiliaries`` is a mapping ``score_id -> path`` or a sequence of
    ``(score_id, path)``. ``n_eff`` is an optional mapping of auxiliary
    effective sample sizes. The LDSC χ² cap is applied to the regression
    rows only; ``m_snps`` stays the full cache length.
    """
    _require_bipred()
    from ldpred3 import ld_scores
    from ldpred3.ld import load_ld_blocks
    from bipred import estimate_sample_overlap, ldsc_chi2_mask, ldsc_rg

    if hasattr(auxiliaries, "items"):
        pairs = list(auxiliaries.items())
    else:
        pairs = list(auxiliaries)
    n_eff_map = {} if n_eff is None else {str(k): float(v)
                                          for k, v in dict(n_eff).items()}

    b1, n1, log1 = align_sumstats_to_cache(
        focal, ld_cache, n_eff=n_eff_focal, qc=qc)
    loaded = load_ld_blocks(ld_cache, return_metadata=True)
    blocks, ids, _meta = loaded
    try:
        ell = ld_scores(blocks)
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()
    m_snps = int(len(ids))

    rgs, ses, overlap, valid, n_used = [], [], [], [], []
    logs = {"focal": log1, "aux": {}}
    for sid, path in pairs:
        sid = str(sid)
        b2, n2, log2 = align_sumstats_to_cache(
            path, ld_cache, n_eff=n_eff_map.get(sid), qc=qc)
        logs["aux"][sid] = log2
        keep = (np.isfinite(b1) & np.isfinite(b2) & np.isfinite(ell)
                & np.isfinite(n1) & np.isfinite(n2)
                & (np.abs(b1) < 1.0) & (np.abs(b2) < 1.0) & (ell > 0))
        if int(keep.sum()) < 50:
            rgs.append(np.nan); ses.append(np.nan)
            overlap.append(np.nan); valid.append(False)
            n_used.append(int(keep.sum()))
            continue
        cap = (ldsc_chi2_mask(b1[keep], n1[keep])
               & ldsc_chi2_mask(b2[keep], n2[keep]))
        rows = np.flatnonzero(keep)[cap]
        if rows.size < 50:
            rgs.append(np.nan); ses.append(np.nan)
            overlap.append(np.nan); valid.append(False)
            n_used.append(int(rows.size))
            continue
        try:
            fit = ldsc_rg(b1[rows], b2[rows], ell[rows], n1[rows], n2[rows],
                          m_snps=m_snps)
            ov = estimate_sample_overlap(
                fit, float(np.nanmedian(n1[rows])),
                float(np.nanmedian(n2[rows])))
        except Exception as exc:
            logs["aux"][sid]["error"] = str(exc)
            rgs.append(np.nan); ses.append(np.nan)
            overlap.append(np.nan); valid.append(False)
            n_used.append(int(rows.size))
            continue
        rgs.append(float(fit.rg))
        se = getattr(fit, "rg_se", np.nan)
        ses.append(float(se) if se is not None and np.isfinite(se)
                   else float("nan"))
        overlap.append(float(ov.get("overlap_corr", np.nan)))
        valid.append(bool(ov.get("cross_corr_valid", False)))
        n_used.append(int(rows.size))

    return RgScreen(
        score_ids=np.array([str(s) for s, _ in pairs], dtype=object),
        rg=np.asarray(rgs, dtype=float), rg_se=np.asarray(ses, dtype=float),
        overlap_corr=np.asarray(overlap, dtype=float),
        overlap_valid=np.asarray(valid, dtype=bool),
        n_used=np.asarray(n_used, dtype=int), log=logs)
