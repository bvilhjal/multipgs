"""Held-out accuracy for a polygenic score, and what each number actually is.

The metrics here take a score and a phenotype **in individuals who were in
neither the discovery GWAS nor the training set that fitted the combination**.
Nothing in this module can check that for you, and it is the assumption every
number below silently depends on.

Three distinctions this module refuses to blur:

* **R² of the score, not of the model.** :func:`r2` is the squared correlation
  between score and phenotype. A regression that also contains age, sex and
  principal components will report a much larger R²; that number describes the
  covariates. When covariates are present the score-attributable quantity is
  :func:`incremental_r2` — how much the score adds on top of them.
* **Observed vs liability scale.** For a case/control phenotype, R² on the 0/1
  scale depends on how many cases were sampled, so it is not comparable across
  studies. :func:`liability_r2` converts it, given the population prevalence.
* **Point estimate vs interval.** With a few thousand individuals the sampling
  interval on R² is wide enough to swallow most reported method differences.
  :func:`evaluate` bootstraps by default for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np


__all__ = ["evaluate", "EvalResult", "r2", "incremental_r2", "auc",
           "nagelkerke_r2", "liability_r2"]


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def _clean(y, pred):
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if y.shape != pred.shape:
        raise ValueError(f"y and pred must have the same length, got "
                         f"{y.shape} and {pred.shape}")
    if y.size < 3:
        raise ValueError("need at least 3 individuals")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")
    if not np.all(np.isfinite(pred)):
        raise ValueError("pred contains non-finite values")
    return y, pred


def _clean_covar(covar, n):
    if covar is None:
        return None
    covar = np.asarray(covar, dtype=float)
    if covar.ndim not in (1, 2):
        raise ValueError("covar must be a vector or 2-dimensional matrix")
    if covar.shape[0] != n:
        raise ValueError("covar has the wrong number of rows")
    if not np.all(np.isfinite(covar)):
        raise ValueError("covar contains non-finite values")
    return covar


def r2(y, pred):
    """Squared Pearson correlation between score and phenotype.

    This is the number usually meant by "PGS R²" when no covariates are
    involved. It is invariant to the score's scale and location, so it does not
    care whether the score was standardized.
    """
    y, pred = _clean(y, pred)
    if np.std(pred) <= 0 or np.std(y) <= 0:
        return 0.0
    return float(np.corrcoef(y, pred)[0, 1] ** 2)


def _design(n, *parts):
    cols = [np.ones((n, 1))]
    for p in parts:
        if p is None:
            continue
        p = np.asarray(p, dtype=float)
        cols.append(p[:, None] if p.ndim == 1 else p)
    return np.hstack(cols)


def _ols_r2(y, D):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    tss = float(np.sum((y - y.mean()) ** 2))
    if tss <= 0:
        return 0.0
    return float(1.0 - np.sum(resid ** 2) / tss)


def incremental_r2(y, pred, covar=None):
    """R² added by the score on top of the covariates (``ΔR²``).

    Fits ``y ~ covar`` and ``y ~ covar + pred`` by least squares and returns
    the difference in R². With ``covar=None`` this reduces to :func:`r2` up to
    floating-point error. This is the standard way to report PGS accuracy in a
    cohort where age, sex, batch and principal components matter.
    """
    y, pred = _clean(y, pred)
    n = y.size
    covar = _clean_covar(covar, n)
    base = _ols_r2(y, _design(n, covar))
    full = _ols_r2(y, _design(n, covar, pred))
    return float(max(full - base, 0.0))


def auc(y, pred):
    """Area under the ROC curve, by the Mann–Whitney identity (ties count ½)."""
    y, pred = _clean(y, pred)
    vals = np.unique(y)
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise ValueError("auc needs y coded 0/1")
    n1 = int(np.sum(y == 1))
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("auc needs both cases and controls")
    order = np.argsort(pred, kind="mergesort")
    ranks = np.empty(pred.size, dtype=float)
    # Average ranks within ties, so a score with no resolution gives 0.5.
    s = pred[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    stops = np.r_[starts[1:], s.size]
    average = (starts + stops + 1.0) / 2.0
    ranks[order] = np.repeat(average, stops - starts)
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _logistic_deviance(y, D, *, max_iter=100, tol=1e-10):
    b = np.zeros(D.shape[1])
    ybar = min(max(float(y.mean()), 1e-9), 1 - 1e-9)
    b[0] = np.log(ybar / (1 - ybar))
    for _ in range(max_iter):
        p = np.clip(1.0 / (1.0 + np.exp(-(D @ b))), 1e-12, 1 - 1e-12)
        w = np.maximum(p * (1 - p), 1e-10)
        H = D.T @ (D * w[:, None])
        H[np.diag_indices_from(H)] += 1e-10
        try:
            step = np.linalg.solve(H, D.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < tol:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-(D @ b))), 1e-12, 1 - 1e-12)
    return float(-2.0 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nagelkerke_r2(y, pred, covar=None):
    """Nagelkerke pseudo-R² added by the score over the covariate-only model.

    With ``covar=None`` the baseline is the intercept-only model, which is the
    usual reported quantity for a case/control PGS.
    """
    y, pred = _clean(y, pred)
    covar = _clean_covar(covar, y.size)
    if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
        raise ValueError("nagelkerke_r2 needs y coded 0/1")
    n = y.size
    d_null = _logistic_deviance(y, _design(n, covar))
    d_full = _logistic_deviance(y, _design(n, covar, pred))
    denom = 1.0 - np.exp(-d_null / n)
    if denom <= 0:
        return 0.0
    return float(max((1.0 - np.exp((d_full - d_null) / n)) / denom, 0.0))


def liability_r2(r2_observed, prevalence, prop_cases):
    """Convert an observed-scale R² to the liability scale (Lee et al. 2012).

    ``prevalence`` is the population disease risk ``K``; ``prop_cases`` is the
    case fraction ``P`` in the sample the R² was measured in. With
    ``t = −Φ⁻¹(K)`` (equivalently ``Φ⁻¹(1−K)``, computed tail-safely as the
    former), ``z = φ(t)`` and ``i = z/K``::

        C  = [K(1−K)]² / (z² · P(1−P))
        θ  = i·(P−K)/(1−K) · [ i·(P−K)/(1−K) − t ]
        R²_liab = C·R²_obs / (1 + C·θ·R²_obs)

    The ``θ`` term is the ascertainment correction of
    `Lee et al. 2012 <https://doi.org/10.1002/gepi.21614>`_ and matters once
    R² is appreciable. Note that :func:`ldpred3.h2_liability` applies only the
    leading factor ``C`` (Lee et al. 2011), which is the right transformation
    for a heritability but understates the shrinkage for a large R².
    """
    K = float(prevalence)
    P = float(prop_cases)
    if not 0.0 < K < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    if not 0.0 < P < 1.0:
        raise ValueError("prop_cases must be in (0, 1)")
    nd = NormalDist()
    # -inv_cdf(K), not inv_cdf(1 - K): the subtraction discards K below the
    # float64 spacing of 1.0 and raises out of NormalDist for K <= 1.1e-16,
    # while Phi^-1(1 - K) = -Phi^-1(K) holds exactly. Identical to the last bit
    # for realistic prevalences; matches ltpred.thresholds and ldpred3.scale.
    t = -nd.inv_cdf(K)
    z = nd.pdf(t)
    i = z / K
    c = (K * (1.0 - K)) ** 2 / (z * z * P * (1.0 - P))
    theta = i * ((P - K) / (1.0 - K)) * (i * ((P - K) / (1.0 - K)) - t)
    r2o = np.asarray(r2_observed, dtype=float)
    if not np.all(np.isfinite(r2o)):
        raise ValueError("r2_observed must contain only finite values")
    if np.any((r2o < 0.0) | (r2o > 1.0)):
        raise ValueError("r2_observed must be in [0, 1]")
    out = c * r2o / (1.0 + c * theta * r2o)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# One call for the whole report
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Metrics with optional bootstrap intervals.

    ``metrics`` maps a name to its point estimate; ``ci`` maps the same names
    to ``(low, high)`` percentile bounds, and is empty when ``n_boot == 0``.
    ``n_boot_skipped`` counts requested bootstrap replicates that produced no
    draw — a single-class resample, or a metric that raised ``ValueError`` —
    so silent interval shrinkage is visible.
    """

    metrics: dict
    ci: dict = field(default_factory=dict)
    n: int = 0
    n_cases: int = 0
    family: str = "gaussian"
    level: float = 0.95
    n_boot_skipped: int = 0

    def __str__(self):
        head = f"n = {self.n}"
        if self.family == "binomial":
            head += f" ({self.n_cases} cases)"
        lines = [head]
        for name, value in self.metrics.items():
            line = f"  {name:<18s} {value: .4f}"
            if name in self.ci:
                lo, hi = self.ci[name]
                line += f"   [{lo: .4f}, {hi: .4f}]"
            lines.append(line)
        if self.ci:
            lines.append(f"  ({self.level:.0%} percentile bootstrap "
                         f"intervals)")
        if self.n_boot_skipped:
            lines.append(f"  ({self.n_boot_skipped} bootstrap replicate(s) "
                         "skipped: the resample could not be evaluated)")
        return "\n".join(lines)


def evaluate(y, pred, *, covar=None, family="gaussian", prevalence=None,
             n_boot=1000, level=0.95, seed=None):
    """Evaluate a score in held-out individuals.

    Parameters
    ----------
    y : array ``(n,)``
        Held-out phenotype. Coded 0/1 for ``family="binomial"``.
    pred : array ``(n,)``
        The score — for a multi-PGS fit, ``fit.multi_pgs(scores)``. Pass the
        score alone, not a prediction that already contains the covariates.
    covar : array ``(n, P)``, optional
        Covariates. When given, an incremental metric is reported alongside the
        raw one.
    family : {"gaussian", "binomial"}
    prevalence : float, optional
        Population prevalence. Adds a liability-scale R² for a binary
        phenotype; the sample case fraction is used as ``P``.
    n_boot : int
        Bootstrap replicates for the intervals. ``0`` skips them.
    level : float
        Interval coverage.
    seed : int, optional
        Bootstrap reproducibility.

    Returns
    -------
    EvalResult
    """
    y, pred = _clean(y, pred)
    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be 'gaussian' or 'binomial'")
    if isinstance(n_boot, (bool, np.bool_)):
        raise ValueError("n_boot must be a non-negative integer")
    try:
        n_boot_int = int(n_boot)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("n_boot must be a non-negative integer") from None
    if n_boot_int < 0 or n_boot_int != n_boot:
        raise ValueError("n_boot must be a non-negative integer")
    try:
        level = float(level)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("level must be a finite number in (0, 1)") from None
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be a finite number in (0, 1)")
    covar_arr = _clean_covar(covar, y.size)

    def compute(idx):
        yy, pp = y[idx], pred[idx]
        cc = None if covar_arr is None else covar_arr[idx]
        out = {"r2": r2(yy, pp)}
        if cc is not None:
            out["incremental_r2"] = incremental_r2(yy, pp, cc)
        if family == "binomial":
            out["auc"] = auc(yy, pp)
            out["nagelkerke_r2"] = nagelkerke_r2(yy, pp, cc)
            if prevalence is not None:
                base = out.get("incremental_r2", out["r2"])
                out["liability_r2"] = liability_r2(base, prevalence,
                                                   float(np.mean(yy)))
        return out

    all_idx = np.arange(y.size)
    metrics = compute(all_idx)

    ci = {}
    n_boot_skipped = 0
    if n_boot_int:
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in metrics}
        for _ in range(n_boot_int):
            idx = rng.integers(0, y.size, y.size)
            if family == "binomial" and len(np.unique(y[idx])) < 2:
                n_boot_skipped += 1
                continue          # a resample with no cases has no AUC
            try:
                rep = compute(idx)
            except ValueError:
                n_boot_skipped += 1
                continue
            for k, v in rep.items():
                draws[k].append(v)
        lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
        for k, vals in draws.items():
            if len(vals) >= 20:
                ci[k] = (float(np.percentile(vals, lo_q)),
                         float(np.percentile(vals, hi_q)))

    return EvalResult(metrics=metrics, ci=ci, n=int(y.size),
                      n_cases=int(np.sum(y == 1)) if family == "binomial"
                      else 0, family=family, level=float(level),
                      n_boot_skipped=n_boot_skipped)
