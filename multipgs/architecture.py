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

* :func:`screen` — the model-level inclusion gates represented here from
  Hansen et al., *Mapping Genetic Architecture of Thousands of Complex Traits
  Using GWAS Summary Statistics*
  (`Research Square, 2026 <https://doi.org/10.21203/rs.3.rs-9415305/v1>`_),
  which applied them across 1,523 GWAS Catalog traits: minimum effective sample
  size, minimum variant count, total and converged chain counts, an optional
  fitted shrinkage coefficient, and a plausible heritability range. Hansen et
  al. excluded fits that failed these quality-control rules; here every failure
  is reported so callers can apply or disable the corresponding gate explicitly.
* :func:`penalty_from_accuracy` — expected own-trait accuracy converted into a
  per-score elastic-net penalty factor, so scores estimated to be weaker are
  penalised harder.

:func:`daetwyler_r2` is the theoretical accuracy bound those rest on
(`Daetwyler et al. 2008 <https://doi.org/10.1371/journal.pone.0003395>`_), with
the number of causal variants taken as ``n_variants * p`` from the fit rather
than assumed a priori — Hansen et al.'s adaptation, which lets the bound follow
each trait's own inferred architecture.

**The caveat that governs all of it.** These quantities describe how well score
``k`` predicts *trait* ``k``. They say nothing about how well it predicts your
*target* trait, which additionally requires genetic correlation. The expected
correlation with its own phenotype is not a universal bound on cross-trait
correlation when trait heritabilities differ. It is only a ranking heuristic
here, not a relevance estimate. Weighting by it can down-weight a weak score
that is highly genetically correlated with your target. That is why
:func:`penalty_from_accuracy` is opt-in and the CMSA fit's default penalty is
flat.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


__all__ = ["Architecture", "daetwyler_r2", "architectures_from_panel",
           "screen", "ScreenResult", "penalty_from_accuracy",
           "penalty_from_relevance"]


@dataclass
class Architecture:
    """Architecture summary for one candidate score.

    ``r2_infer`` is LDpred3's own inferred predictive r² (signed, and it can be
    negative for a score with no signal). ``n_variants`` is the count that
    survived QC and entered the fit. ``shrinkage`` is Hansen et al.'s final
    LDpred2-auto shrinkage coefficient; it is not ldpred3's ``shrink_corr`` LD
    regularisation input.
    """

    score_id: str
    h2: float = float("nan")
    p: float = float("nan")
    r2_infer: float = float("nan")
    n_chains_kept: int = 0
    n_chains: int = 0
    n_variants: int = 0
    n_eff: float = float("nan")
    shrinkage: float = float("nan")
    rg: float = float("nan")

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
    bad = ~(np.isfinite(out)
            & np.isfinite(h2) & (h2 > 0) & (h2 <= 1)
            & np.isfinite(p) & (p > 0) & (p <= 1)
            & np.isfinite(n_eff) & (n_eff > 0)
            & np.isfinite(n_variants) & (n_variants > 0)
            & np.isfinite(m) & (m > 0))
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
    is ``nan`` and the default effective-sample-size gate rejects a fitted
    architecture; pass ``min_n_eff=None`` to :func:`screen` only when that gate
    is deliberately unavailable.
    """
    ids = [str(s) for s in np.asarray(panel.score_ids, dtype=object).ravel()]
    duplicate_ids = _duplicates(ids)
    if duplicate_ids:
        raise ValueError("panel score_ids contains duplicate(s): "
                         + ", ".join(duplicate_ids[:3]))
    if n_eff is None:
        n_eff_map = {}
    elif hasattr(n_eff, "items"):
        items = list(n_eff.items())
        item_ids = [str(key) for key, _ in items]
        duplicate_n_eff = _duplicates(item_ids)
        if duplicate_n_eff:
            raise ValueError("n_eff contains duplicate score id(s) after "
                             "string conversion: "
                             + ", ".join(duplicate_n_eff[:3]))
        n_eff_map = {sid: float(value)
                     for sid, (_, value) in zip(item_ids, items)}
    else:
        vals = np.asarray(n_eff, dtype=float).ravel()
        if vals.size != len(ids):
            raise ValueError(f"n_eff has {vals.size} entries for {len(ids)} "
                             f"scores")
        n_eff_map = dict(zip(ids, vals.tolist()))

    out = []
    for k, sid in enumerate(ids):
        meta = panel.meta[k] if k < len(panel.meta) else {}
        if not isinstance(meta, Mapping):
            raise ValueError(f"panel.meta[{k}] for {sid!r} must be a mapping")
        inf = meta.get("inference") or {}
        if not isinstance(inf, Mapping):
            raise ValueError(f"inference metadata for {sid!r} must be a "
                             "mapping")
        n_variants = meta.get("n_variants", meta.get("n_matched", 0))
        stored_n = meta.get("n_eff", n_eff_map.get(sid, np.nan))
        out.append(Architecture(
            score_id=sid,
            h2=float(inf.get("h2_est", np.nan)),
            p=float(inf.get("p_est", np.nan)),
            r2_infer=float(inf.get("r2_est", np.nan)),
            n_chains_kept=int(inf.get("n_chains_kept", 0) or 0),
            n_chains=int(inf.get("n_chains", meta.get("n_chains", 0)) or 0),
            n_variants=int(n_variants or 0),
            n_eff=float(n_eff_map[sid] if sid in n_eff_map else stored_n),
            # Only this exact field has Hansen's meaning. In particular, never
            # reinterpret ldpred3's shrink_corr LD-regularisation control.
            shrinkage=float(inf.get(
                "shrinkage", meta.get("shrinkage", np.nan))),
            rg=float(meta.get("rg", inf.get("rg", np.nan)))))
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
           min_chains_total=50, min_variants=60_000, min_n_eff=10_000,
           min_shrinkage=None, min_expected_r2=None, min_abs_rg=None,
           keep_unscreenable=True):
    """Apply the Hansen et al. inclusion gates to candidate scores.

    Defaults implement the supported subset of that paper's thresholds (their
    Table 3, plus the strict ``n_eff > 10,000`` requirement from their §3.1.1):
    20 converged chains out of at least 50, 60,000 post-QC variants, and h² in
    [0.01, 1]. Their LDpred2-auto shrinkage gate is available as
    ``min_shrinkage=0.4`` but is off by default because ldpred3 does not infer
    that quantity. It must be supplied explicitly as ``Architecture.shrinkage``;
    ``shrink_corr`` is a different LD-regularisation input and is never used.

    These gates screen a GWAS on its own terms and say nothing about ancestry
    match: ``h2``, ``p`` and ``n_eff`` are all properties of the discovery
    cohort. A score that passes every gate can still transfer poorly to a target
    of different ancestry.

    Parameters
    ----------
    architectures : sequence of Architecture
    h2_range : (float, float)
    min_chains_kept : int
        Converged chains required. ``None`` disables the gate.
    min_chains_total : int
        Total chains required. ``None`` disables the gate.
    min_variants : int
        Variants surviving QC. ``None`` disables the gate.
    min_n_eff : float
        Discovery GWAS effective sample size, which must be strictly greater
        than this threshold. Use ``None`` to disable the gate and
        :func:`ldpred3.n_eff_case_control` for case/control studies.
    min_shrinkage : float, optional
        Hansen et al.'s fitted LDpred2-auto shrinkage coefficient. Off by
        default because ldpred3 does not estimate it.
    min_expected_r2 : float, optional
        Also require the Daetwyler bound to clear this. Off by default: it needs
        ``n_eff``, and the bound is optimistic enough that a threshold on it is
        a blunt instrument.
    min_abs_rg : float, optional
        Drop a fitted score whose ``Architecture.rg`` (typically from
        :func:`multipgs.ldsc_rg_screen`) is missing or below this absolute
        genetic correlation with the focal trait. Off by default.
    keep_unscreenable : bool
        What to do with scores carrying no architecture at all (PGS Catalog
        weights, typically). ``True`` lets them through and flags them; ``False``
        drops them. Silently failing them would empty a catalog panel.

    Returns
    -------
    ScreenResult
    """
    if not isinstance(keep_unscreenable, (bool, np.bool_)):
        raise ValueError("keep_unscreenable must be boolean")
    if h2_range is None:
        lo = hi = None
    else:
        try:
            if len(h2_range) != 2:
                raise ValueError
            lo, hi = float(h2_range[0]), float(h2_range[1])
        except (TypeError, ValueError, OverflowError):
            raise ValueError("h2_range must contain two finite values in "
                             "[0, 1]") from None
        if not (np.isfinite(lo) and np.isfinite(hi)
                and 0 <= lo <= hi <= 1):
            raise ValueError("h2_range must contain two finite values in "
                             "[0, 1]")

    min_chains_kept = _minimum("min_chains_kept", min_chains_kept,
                               integer=True)
    min_chains_total = _minimum("min_chains_total", min_chains_total,
                                integer=True)
    min_variants = _minimum("min_variants", min_variants, integer=True)
    min_n_eff = _minimum("min_n_eff", min_n_eff)
    min_shrinkage = _minimum("min_shrinkage", min_shrinkage, upper=1.0)
    min_expected_r2 = _minimum("min_expected_r2", min_expected_r2,
                               upper=1.0)
    min_abs_rg = _minimum("min_abs_rg", min_abs_rg, upper=1.0)

    architectures = list(architectures)
    normalized_ids = [str(a.score_id) for a in architectures]
    duplicate_ids = _duplicates(normalized_ids)
    if duplicate_ids:
        raise ValueError("architectures contains duplicate score_id(s): "
                         + ", ".join(duplicate_ids[:3]))

    ids, keep, reasons, unscreenable = [], [], {}, []
    for a, sid in zip(architectures, normalized_ids):
        if not sid:
            raise ValueError("Architecture.score_id must not be empty")
        h2 = _optional_number("h2", a.h2, sid)
        p = _optional_number("p", a.p, sid)
        r2_infer = _optional_number("r2_infer", a.r2_infer, sid)
        shrinkage = _optional_number("shrinkage", a.shrinkage, sid)
        n_eff = _optional_number("n_eff", a.n_eff, sid)
        rg = _optional_number("rg", a.rg, sid)
        if np.isfinite(shrinkage) and not 0 <= shrinkage <= 1:
            raise ValueError(f"{sid}: shrinkage must be in [0, 1] or nan")
        n_chains_kept = _count("n_chains_kept", a.n_chains_kept, sid)
        n_chains = _count("n_chains", a.n_chains, sid)
        n_variants = _count("n_variants", a.n_variants, sid)
        if n_chains and n_chains_kept > n_chains:
            raise ValueError(f"{sid}: n_chains_kept cannot exceed n_chains")

        ids.append(sid)
        # A Catalog score can carry a matched-variant count and a caller may
        # supply n_eff, but neither proves that an architecture model was fit.
        has_model_arch = (np.isfinite(h2) or np.isfinite(p)
                          or np.isfinite(r2_infer)
                          or np.isfinite(shrinkage)
                          or n_chains_kept > 0 or n_chains > 0)
        has_arch = has_model_arch or np.isfinite(rg)
        unscreenable.append(not has_arch)
        if not has_arch:
            keep.append(bool(keep_unscreenable))
            if not keep_unscreenable:
                reasons[sid] = "no architecture available"
            continue
        fail = None
        if has_model_arch:
            if lo is not None and not np.isfinite(h2):
                fail = "heritability not estimated"
            elif lo is not None and h2 < lo:
                fail = f"heritability below {lo:g}"
            elif hi is not None and h2 > hi:
                fail = f"heritability above {hi:g}"
            elif min_chains_kept is not None and n_chains_kept == 0:
                fail = "converged chain count not available"
            elif (min_chains_kept is not None
                  and n_chains_kept < min_chains_kept):
                fail = f"fewer than {min_chains_kept} chains converged"
            elif min_chains_total is not None and n_chains == 0:
                fail = "total chain count not available"
            elif min_chains_total is not None and n_chains < min_chains_total:
                fail = f"fewer than {min_chains_total} total chains"
            elif min_variants is not None and n_variants == 0:
                fail = "post-QC variant count not available"
            elif min_variants is not None and n_variants < min_variants:
                fail = f"fewer than {min_variants:,} variants after QC"
            elif min_n_eff is not None and not np.isfinite(n_eff):
                fail = "effective sample size not available"
            elif min_n_eff is not None and n_eff <= min_n_eff:
                fail = (f"effective sample size not above "
                        f"{min_n_eff:,.0f}")
            elif min_shrinkage is not None and not np.isfinite(shrinkage):
                fail = "shrinkage coefficient not available"
            elif min_shrinkage is not None and shrinkage < min_shrinkage:
                fail = f"shrinkage coefficient below {min_shrinkage:g}"
        if fail is None and min_expected_r2 is not None:
            exp = a.expected_r2()
            if not np.isfinite(exp) or exp < min_expected_r2:
                fail = f"expected r2 below {min_expected_r2:g}"
        if fail is None and min_abs_rg is not None and not np.isfinite(rg):
            fail = "genetic correlation not available"
        elif fail is None and min_abs_rg is not None and abs(rg) < min_abs_rg:
            fail = f"|rg| below {min_abs_rg:g}"
        keep.append(fail is None)
        if fail is not None:
            reasons[sid] = fail
    return ScreenResult(keep=np.array(keep, dtype=bool), reasons=reasons,
                        score_ids=np.array(ids, dtype=object),
                        unscreenable=np.array(unscreenable, dtype=bool))


def penalty_from_accuracy(expected_r2, *, power=1.0, clip=4.0, floor=1e-4):
    """Per-score elastic-net penalty factors from expected accuracy.

    Let ``a_k = sqrt(r2_k)`` be a score's expected correlation with its own
    phenotype. Penalising in inverse proportion to this ranking heuristic::

        raw_pf_k  =  (gmean(a) / a_k) ** power

    The log factors are projected onto ``[-log(clip), log(clip)]`` while their
    mean remains zero. Thus every returned factor is in ``[1/clip, clip]`` and
    their geometric mean is one, which fixes a neutral scale for comparing the
    relative penalties. It does not make the :math:`\\lambda` grid invariant:
    :func:`multipgs.multi_pgs_fit` recomputes that grid for the supplied
    factors. This is an adaptive-lasso weighting whose prior comes from summary
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
        R² floor before taking the square root, in ``(0, 1]``.

    Notes
    -----
    This weighting ignores genetic correlation with the target, so it will
    over-penalise a low-powered GWAS of a trait that happens to be genetically
    identical to yours. It is worth using when the candidate set is large and
    mostly irrelevant, and worth skipping when it is small and hand-picked.
    """
    r2 = np.asarray(expected_r2, dtype=float).ravel().copy()
    if r2.size == 0:
        raise ValueError("expected_r2 must contain at least one value")
    try:
        power = float(power)
        floor = float(floor)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("power and floor must be finite numbers") from None
    if not np.isfinite(power) or power < 0:
        raise ValueError("power must be finite and >= 0")
    if not np.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("floor must be finite and in (0, 1]")
    if np.any(np.isfinite(r2) & (r2 > 1)):
        raise ValueError("finite expected_r2 values must be <= 1")
    r2[~np.isfinite(r2) | (r2 <= 0)] = floor
    r2 = np.maximum(r2, floor)
    a = np.sqrt(r2)
    with np.errstate(over="ignore", invalid="ignore"):
        log_pf = power * (np.mean(np.log(a)) - np.log(a))
    if not np.all(np.isfinite(log_pf)):
        raise ValueError("power is too large for the supplied expected_r2")
    if clip is None:
        pf = np.exp(log_pf)
        if not np.all(np.isfinite(pf)) or np.any(pf <= 0):
            raise ValueError("unclipped penalty factors underflow or overflow; "
                             "set clip")
        return pf

    try:
        c = float(clip)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("clip must be a finite number >= 1 or None") from None
    if not np.isfinite(c) or c < 1.0:
        raise ValueError("clip must be a finite number >= 1 or None")
    if c == 1.0 or power == 0.0:
        return np.ones(r2.size, dtype=float)

    limit = np.log(c)
    # A common shift followed by symmetric clipping is the log-space
    # projection that preserves both constraints. The clipped mean is monotone
    # in the shift, so bisection is deterministic and O(K).
    lower = -limit - float(np.max(log_pf))
    upper = limit - float(np.min(log_pf))
    for _ in range(80):
        shift = (lower + upper) / 2.0
        mean = float(np.mean(np.clip(log_pf + shift, -limit, limit)))
        if mean < 0:
            lower = shift
        else:
            upper = shift
    projected = np.clip(log_pf + (lower + upper) / 2.0, -limit, limit)
    return np.exp(projected)


def penalty_from_relevance(expected_r2, rg, *, power=1.0, clip=4.0,
                           floor=1e-4):
    """Penalty factors from own-trait accuracy times ``r_G²``.

    Uses :func:`penalty_from_accuracy` on ``r2_k * rg_k²``. Because finite-
    sample LDSC estimates can fall outside the correlation parameter space,
    finite ``rg`` is clipped to ``[-1, 1]`` before squaring. Missing or
    non-finite ``rg`` is treated as zero relevance (the most-penalised rank).
    This is still a ranking heuristic, not a bound on target-trait prediction.
    """
    r2 = np.asarray(expected_r2, dtype=float).ravel()
    g = np.asarray(rg, dtype=float).ravel()
    if r2.shape != g.shape:
        raise ValueError("expected_r2 and rg must have the same length")
    bounded_g = np.clip(np.where(np.isfinite(g), g, 0.0), -1.0, 1.0)
    rel = r2 * np.square(bounded_g)
    return penalty_from_accuracy(rel, power=power, clip=clip, floor=floor)


def _duplicates(values):
    """Unique duplicate strings, preserving first duplicate order."""
    seen, reported, out = set(), set(), []
    for value in values:
        if value in seen and value not in reported:
            out.append(value)
            reported.add(value)
        seen.add(value)
    return out


def _minimum(name, value, *, integer=False, upper=None):
    """Validate an optional non-negative screening threshold."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite non-negative number or "
                         "None") from None
    if not np.isfinite(number) or number < 0 or (upper is not None
                                                 and number > upper):
        bound = f" in [0, {upper:g}]" if upper is not None else " non-negative"
        raise ValueError(f"{name} must be finite and{bound}, or None")
    if integer and not number.is_integer():
        raise ValueError(f"{name} must be an integer or None")
    return int(number) if integer else number


def _optional_number(name, value, score_id):
    """Coerce an optional architecture scalar; NaN represents missing."""
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{score_id}: {name} must be numeric") from None
    if np.isinf(number):
        raise ValueError(f"{score_id}: {name} must be finite or nan")
    return number


def _count(name, value, score_id):
    """Coerce a non-negative integer architecture count."""
    if value is None:
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{score_id}: {name} must be a non-negative integer") \
            from None
    if not np.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{score_id}: {name} must be a non-negative integer")
    return int(number)
