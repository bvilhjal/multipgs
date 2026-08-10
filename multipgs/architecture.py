"""Per-score genetic architecture: which scores belong in a stack, and how hard
to shrink them.

A multi-PGS stack is only as good as the scores fed into it, and the choice of
inputs is normally made blind — every scoring file in a directory goes in, and
the penalty is left to sort them out. It does not have to be blind. Fitting each
GWAS with LDpred3's ``auto`` model yields, from summary statistics alone,
SNP heritability :math:`h^2`, polygenicity :math:`p`, and an inferred predictive
:math:`r^2` — so each candidate score arrives with a statement about how much
signal it could carry before any individual-level data is touched.

This module turns that into two things:

* :func:`screen` — the inclusion gates of Hansen et al., *Mapping Genetic
  Architecture of Thousands of Complex Traits Using GWAS Summary Statistics*
  (`Research Square, 2026 <https://doi.org/10.21203/rs.3.rs-9415305/v1>`_),
  which applied them across 1,523 GWAS Catalog traits: minimum effective sample
  size, minimum variant count, chain convergence, and a plausible heritability
  range. A GWAS that fails these does not produce a usable score, and a stack
  is better off without it than shrinking it toward zero at the cost of a
  degree of freedom.
* :func:`penalty_from_accuracy` — expected accuracy converted into a per-score
  elastic-net penalty factor, so scores that could not possibly contribute much
  are penalised harder than scores that could.

:func:`daetwyler_r2` is the theoretical accuracy bound those rest on
(`Daetwyler et al. 2008 <https://doi.org/10.1371/journal.pone.0003395>`_), with
the number of causal variants taken as ``n_variants * p`` from the fit rather
than assumed a priori — Hansen et al.'s adaptation, which lets the bound follow
each trait's own inferred architecture.

**The caveat that governs all of it.** These quantities describe how well score
``k`` predicts *trait* ``k``. They say nothing about how well it predicts your
*target* trait, which additionally requires genetic correlation. Accuracy for
its own trait is an upper bound on the correlation a score can have with
anything else, so it is a defensible prior — a score that predicts nothing
cannot predict your trait either — but it is a bound, not a relevance estimate.
Weighting by it will down-weight a weak score that happens to be genetically
identical to your target. That is why :func:`penalty_from_accuracy` is opt-in
and the CMSA fit's default penalty is flat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["Architecture", "daetwyler_r2", "architectures_from_panel",
           "screen", "ScreenResult", "penalty_from_accuracy"]


@dataclass
class Architecture:
    """Architecture summary for one candidate score.

    ``r2_infer`` is LDpred3's own inferred predictive r² (signed, and it can be
    negative for a score with no signal). ``n_variants`` is the count that
    survived QC and entered the fit.
    """

    score_id: str
    h2: float = float("nan")
    p: float = float("nan")
    r2_infer: float = float("nan")
    n_chains_kept: int = 0
    n_chains: int = 0
    n_variants: int = 0
    n_eff: float = float("nan")

    def expected_r2(self):
        """Daetwyler bound from this score's own inferred architecture."""
        return daetwyler_r2(self.h2, self.p, self.n_eff, self.n_variants)


def daetwyler_r2(h2, p, n_eff, n_variants):
    """Expected squared correlation of a polygenic score with its own **phenotype**.

    With ``M = n_variants * p`` causal variants and ``x = n_eff * h2 / M``::

        R^2 = h2 * x / (1 + x)

    This is the Daetwyler et al. (2008) bound with ``M`` taken from the fitted
    polygenicity rather than assumed, so it reflects the trait's own inferred
    architecture. It is an *upper* bound in practice: it assumes the causal
    variants are known and independent, so it ignores LD-induced dilution and
    every source of between-cohort heterogeneity.

    **Which scale.** The value returned is the r² against the *phenotype*, so
    ``sqrt(daetwyler_r2(...))`` is ``h · R``, not ``R``. The accuracy against
    the *genetic value* is ``R = sqrt(x/(1+x)) = sqrt(r2/h2)``. The factor is a
    constant within one trait and cancels from any weight direction, so
    :func:`multipgs.meta_pgs` is unaffected; across a panel of different traits
    it does not cancel, which is one of the two reasons
    :func:`penalty_from_accuracy` is a ranking heuristic rather than a bound
    (the other being that it ignores genetic correlation with the target).

    Returns ``nan`` where any input is missing or out of range, rather than a
    number that looks usable.
    """
    h2 = np.asarray(h2, dtype=float)
    p = np.asarray(p, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)
    n_variants = np.asarray(n_variants, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = n_variants * p
        x = n_eff * h2 / m
        out = h2 * x / (1.0 + x)
    bad = ~(np.isfinite(out) & (h2 > 0) & (p > 0) & (n_eff > 0) & (m > 0))
    out = np.where(bad, np.nan, out)
    return float(out) if out.ndim == 0 else out


def architectures_from_panel(panel, *, n_eff=None):
    """Pull per-score architecture out of a panel built by ldpred3.

    :func:`multipgs.panel.panel_from_sumstats` stores each fit's inference dict
    in ``panel.meta``; this reads it back. Scores with no inference recorded —
    every PGS Catalog score, which arrives as weights with no model behind it —
    come back with ``nan`` fields, and :func:`screen` will report them as
    unscreenable rather than silently passing or failing them.

    ``n_eff`` supplies the discovery GWAS effective sample size per score, which
    ldpred3 does not carry in its inference dict. Without it the Daetwyler bound
    is ``nan``; the convergence and heritability gates still work.
    """
    ids = [str(s) for s in np.asarray(panel.score_ids, dtype=object)]
    if n_eff is None:
        n_eff_map = {}
    elif hasattr(n_eff, "items"):
        n_eff_map = {str(k): float(v) for k, v in n_eff.items()}
    else:
        vals = np.asarray(n_eff, dtype=float).ravel()
        if vals.size != len(ids):
            raise ValueError(f"n_eff has {vals.size} entries for {len(ids)} "
                             f"scores")
        n_eff_map = dict(zip(ids, vals.tolist()))

    out = []
    for k, sid in enumerate(ids):
        meta = panel.meta[k] if k < len(panel.meta) else {}
        inf = meta.get("inference") or {}
        out.append(Architecture(
            score_id=sid,
            h2=float(inf.get("h2_est", np.nan)),
            p=float(inf.get("p_est", np.nan)),
            r2_infer=float(inf.get("r2_est", np.nan)),
            n_chains_kept=int(inf.get("n_chains_kept", 0) or 0),
            n_chains=int(inf.get("n_chains", 0) or 0),
            n_variants=int(meta.get("n_matched", 0) or 0),
            n_eff=n_eff_map.get(sid, np.nan)))
    return out


@dataclass
class ScreenResult:
    """Which scores passed, and why the rest did not."""

    keep: np.ndarray                 # boolean mask over the input order
    reasons: dict                    # score_id -> first failed gate
    score_ids: np.ndarray
    unscreenable: np.ndarray         # boolean: no architecture available

    @property
    def n_kept(self):
        return int(np.count_nonzero(self.keep))

    def summary(self):
        from collections import Counter
        counts = Counter(self.reasons.values())
        lines = [f"screen: kept {self.n_kept} of {self.keep.size} scores"]
        if int(np.count_nonzero(self.unscreenable)):
            lines.append(f"  {int(np.count_nonzero(self.unscreenable))} had no "
                         f"architecture to screen on")
        for reason, count in counts.most_common():
            lines.append(f"  {count:5d}  {reason}")
        return "\n".join(lines)


def screen(architectures, *, h2_range=(0.01, 1.0), min_chains_kept=20,
           min_variants=60_000, min_n_eff=10_000, min_expected_r2=None,
           keep_unscreenable=True):
    """Apply the Hansen et al. inclusion gates to candidate scores.

    Defaults are that paper's thresholds (their Table 3, plus the
    ``n_eff > 10,000`` requirement from their §3.1.1). The two that matter most
    in practice are ``min_chains_kept`` (non-convergence usually means the GWAS
    and the LD reference disagree — a mixed-ancestry discovery sample, most
    often) and ``h2_range`` (an h² below 0.01 leaves nothing to predict with;
    above 1 is not a heritability).

    One gate of theirs is **not** implemented: they also required an LDpred2-auto
    shrinkage coefficient of 0.4 or larger, as a guard against model
    misspecification. ldpred3's inference dict does not carry that quantity, so
    it cannot be checked here. If you fit the panel yourself and have it, apply
    it before calling this.

    These gates screen a GWAS on its own terms and say nothing about ancestry
    match: ``h2``, ``p`` and ``n_eff`` are all properties of the discovery
    cohort. A score that passes every gate can still transfer poorly to a target
    of different ancestry.

    Parameters
    ----------
    architectures : sequence of Architecture
    h2_range : (float, float)
    min_chains_kept : int
        Converged chains required, out of however many were run.
    min_variants : int
        Variants surviving QC.
    min_n_eff : float
        Discovery GWAS effective sample size. Use
        :func:`ldpred3.n_eff_case_control` for case/control studies.
    min_expected_r2 : float, optional
        Also require the Daetwyler bound to clear this. Off by default: it needs
        ``n_eff``, and the bound is optimistic enough that a threshold on it is
        a blunt instrument.
    keep_unscreenable : bool
        What to do with scores carrying no architecture at all (PGS Catalog
        weights, typically). ``True`` lets them through and flags them; ``False``
        drops them. Silently failing them would empty a catalog panel.

    Returns
    -------
    ScreenResult
    """
    lo, hi = float(h2_range[0]), float(h2_range[1])
    ids, keep, reasons, unscreenable = [], [], {}, []
    for a in architectures:
        ids.append(a.score_id)
        has_arch = np.isfinite(a.h2) or a.n_chains_kept > 0
        unscreenable.append(not has_arch)
        if not has_arch:
            keep.append(bool(keep_unscreenable))
            if not keep_unscreenable:
                reasons[a.score_id] = "no architecture available"
            continue
        fail = None
        if not np.isfinite(a.h2):
            fail = "heritability not estimated"
        elif a.h2 < lo:
            fail = f"heritability below {lo:g}"
        elif a.h2 > hi:
            fail = f"heritability above {hi:g}"
        elif a.n_chains_kept < int(min_chains_kept):
            fail = f"fewer than {int(min_chains_kept)} chains converged"
        elif a.n_variants and a.n_variants < int(min_variants):
            fail = f"fewer than {int(min_variants):,} variants after QC"
        elif np.isfinite(a.n_eff) and a.n_eff < float(min_n_eff):
            fail = f"effective sample size below {float(min_n_eff):,.0f}"
        elif min_expected_r2 is not None:
            exp = a.expected_r2()
            if not np.isfinite(exp) or exp < float(min_expected_r2):
                fail = f"expected r2 below {float(min_expected_r2):g}"
        keep.append(fail is None)
        if fail is not None:
            reasons[a.score_id] = fail
    return ScreenResult(keep=np.array(keep, dtype=bool), reasons=reasons,
                        score_ids=np.array(ids, dtype=object),
                        unscreenable=np.array(unscreenable, dtype=bool))


def penalty_from_accuracy(expected_r2, *, power=1.0, clip=4.0, floor=1e-4):
    """Per-score elastic-net penalty factors from expected accuracy.

    A score's expected correlation with its own trait, ``a_k = sqrt(r2_k)``,
    bounds the correlation it can have with anything else. Penalising in
    inverse proportion to that bound::

        pf_k  =  (gmean(a) / a_k) ** power,   clipped to [1/clip, clip]

    and rescaled to geometric mean 1, so the overall penalty scale — and hence
    the :math:`\\lambda` grid — is unchanged and only the *relative* shrinkage
    moves. This is an adaptive-lasso weighting whose prior comes from summary
    statistics rather than from a first-stage fit.

    Feed the result to :func:`multipgs.multi_pgs_fit` as ``penalty_factor``.

    Parameters
    ----------
    expected_r2 : array
        Per-score expected r², e.g. from :func:`daetwyler_r2` or an ldpred3
        ``r2_est``. Non-finite or non-positive entries are treated as ``floor``,
        which makes them the most-penalised scores rather than an error.
    power : float
        ``0`` gives a flat penalty (no weighting); ``1`` is the inverse-accuracy
        rule above; larger values sharpen it.
    clip : float
        Bound on each individual factor, which is held to ``[1/clip, clip]``.
        The largest-to-smallest *ratio* is therefore up to ``clip**2``. This
        stops one implausible accuracy estimate forcing a score in or out on
        its own.
    floor : float
        Accuracy floor before taking the square root.

    Notes
    -----
    This weighting ignores genetic correlation with the target, so it will
    over-penalise a low-powered GWAS of a trait that happens to be genetically
    identical to yours. It is worth using when the candidate set is large and
    mostly irrelevant, and worth skipping when it is small and hand-picked.
    """
    r2 = np.asarray(expected_r2, dtype=float).ravel().copy()
    r2[~np.isfinite(r2)] = floor
    r2 = np.maximum(r2, float(floor))
    a = np.sqrt(r2)
    pf = (np.exp(np.mean(np.log(a))) / a) ** float(power)
    if clip is not None:
        c = float(clip)
        if c < 1.0:
            raise ValueError("clip must be >= 1")
        pf = np.clip(pf, 1.0 / c, c)
    return pf / np.exp(np.mean(np.log(pf)))
