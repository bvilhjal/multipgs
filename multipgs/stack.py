"""Multi-PGS: combine many polygenic scores into one, by penalized regression.

This is the estimator of Albiñana et al., *Multi-PGS enhances polygenic
prediction by combining 937 polygenic scores*
(`Nat Commun 14, 4702, 2023 <https://doi.org/10.1038/s41467-023-40330-w>`_).
Given a training cohort with

* an ``n x K`` matrix of polygenic scores — the target trait's own score plus
  scores for however many other traits are available, and
* the target phenotype, plus the usual covariates,

it fits a penalized regression of the phenotype on the scores and returns one
combined score. Genetically correlated traits can carry information the target
trait's own GWAS is too small to see, so the combination can improve on the
single-trait score when those auxiliary signals generalize.

Model selection here is **Cross-Model Selection and Averaging** (CMSA, Privé,
Aschard & Blum, `Genetics 212:65-74, 2019
<https://doi.org/10.1534/genetics.119.302019>`_), the procedure behind
``bigstatsr::big_spLinReg``. That is this package's choice, not the paper's:
Albiñana et al. fitted their combination with ``cv.glmnet`` and assessed it by
fivefold cross-validation in iPSYCH. What is taken from the paper is the
estimator — penalized regression of the phenotype on a panel of scores, with
covariates at penalty factor 0 — not the routine that selects its penalty.

The training set is split into ``n_folds`` parts. Each part in turn is held out
while the elastic-net path is fitted on the rest and scored on the held-out
part; each fold keeps the coefficients at *its own* best
:math:`(\\alpha, \\lambda)`; the returned model is the average of those fold
coefficient vectors. Two properties matter in practice:

* No separate tuning cohort is consumed. Each held-out fold selects one
  coefficient vector, and those vectors are averaged. This ordinary CMSA loop
  tunes the returned estimator; the separate nested outer loop assesses it.
* Averaging over folds is a variance reduction. With ``K`` correlated scores
  and a lasso penalty, *which* of a set of near-duplicate scores gets picked is
  close to arbitrary; averaging spreads the weight over them instead of
  betting the model on one draw.

What this module does **not** do is protect you from sample overlap. If
individuals in the training cohort also contributed to the GWAS behind one of
the ``K`` input scores, that score is partly fitted to its own training data,
CMSA will happily reward it, and the resulting accuracy is optimistic. The bias
generally increases with overlap, but its magnitude also depends on discovery
design, score construction, relatedness and effect-size estimation. Overlap is
frequently not visible in public summary statistics at all (Wray et al.,
`Nat Rev Genet 14:507-515, 2013
<https://doi.org/10.1038/nrg3457>`_). Many PGS Catalog scores are UK
Biobank-derived — check each score's development samples in its Catalog
metadata. Overlap has to be excluded when the panel is built; no amount of
cross-validation inside the target cohort detects it, because every fold shares
the contamination.

Nor does anything here model ancestry. The combination's coefficients are
learned in the target cohort and so are appropriate to it, but the input scores'
accuracy is a property of their discovery cohorts; expect accuracy to fall in a
target ancestry unmatched to discovery (Martin et al., `Nat Genet
51:584-591, 2019 <https://doi.org/10.1038/s41588-019-0379-x>`_).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _coord


__all__ = ["multi_pgs_fit", "MultiPGSFit", "FoldFit"]

_FAMILIES = ("gaussian", "binomial")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldFit:
    """What one CMSA fold selected, and how well it did.

    ``loss`` and ``null_loss`` are on the held-out part: mean squared error for
    ``gaussian``, mean binomial deviance for ``binomial``. ``null_loss`` is the
    loss of the explicit unpenalized baseline: covariates plus any scores whose
    penalty factor is zero. ``used`` says whether this fold's coefficient vector
    entered the returned CMSA average. It is therefore true for every fold when
    the nested assessment passes the incremental-signal gate, and false for
    every fold when the returned fit is the full-data baseline.
    """

    fold: int
    alpha: float
    lam: float
    lam_index: int
    loss: float
    null_loss: float
    n_nonzero: int
    used: bool


@dataclass
class MultiPGSFit:
    """A fitted multi-PGS combination.

    ``beta`` is on the **raw score scale**: ``scores @ beta`` is the combined
    score for new individuals, in the units the input scores came in. Use
    :meth:`multi_pgs` for that. ``beta_std`` rescales the same coefficients by
    each score's training standard deviation, which is the comparable quantity
    to rank scores by — see :meth:`selected`.
    """

    beta: np.ndarray
    intercept: float
    covar_beta: np.ndarray
    beta_std: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    score_ids: np.ndarray
    covar_ids: np.ndarray
    family: str
    folds: list = field(default_factory=list)
    log: dict = field(default_factory=dict)

    # -- use -------------------------------------------------------------
    def multi_pgs(self, scores, *, score_ids=None):
        """The combined score itself: ``scores @ beta``, no covariates.

        This is what you carry into a held-out cohort and evaluate, and what
        :func:`multipgs.metrics.evaluate` expects as ``pred``. Covariates are
        deliberately excluded: an accuracy number that includes age and sex is
        not a polygenic-score accuracy.

        Columns are matched **by position**. Pass a :class:`ScorePanel` (or its
        ``score_ids``) and the order is checked instead — do that whenever the
        panel was built separately from the training one. A test panel can
        legitimately come back with the same number of columns in a different
        order: ``on_error="skip"`` may drop a different file, and a list of
        paths does not sort the way a directory does. The result is silently
        wrong rather than an error, and badly so.
        """
        if hasattr(scores, "scores") and hasattr(scores, "score_ids"):
            if score_ids is None:
                score_ids = scores.score_ids
            scores = scores.scores
        if score_ids is not None:
            got = [str(s) for s in np.asarray(score_ids, dtype=object)]
            want = [str(s) for s in np.asarray(self.score_ids, dtype=object)]
            if got != want:
                extra = [s for s in got if s not in set(want)]
                missing = [s for s in want if s not in set(got)]
                detail = (f"; not in the fit: {extra[:3]}" if extra else "") + \
                    (f"; missing from these scores: {missing[:3]}" if missing
                     else "")
                raise ValueError(
                    "these scores are not the ones this fit was trained on, or "
                    "are in a different order" + detail +
                    ". Realign with panel.select(list(fit.score_ids)).")
        s = self._prepare(scores, len(self.beta), "scores")
        return s @ self.beta

    def predict(self, scores, covar=None):
        """Full linear predictor, including intercept and covariates.

        For ``family="binomial"`` this is on the log-odds scale; pass it
        through :meth:`predict_proba` for probabilities.
        """
        out = self.multi_pgs(scores) + self.intercept
        if self.covar_beta.size:
            if covar is None:
                raise ValueError(
                    f"this fit used {self.covar_beta.size} covariate(s) "
                    f"({', '.join(map(str, self.covar_ids))}); pass covar=")
            c = self._prepare(covar, self.covar_beta.size, "covar")
            out = out + c @ self.covar_beta
        elif covar is not None:
            raise ValueError("this fit used no covariates; covar= is not used")
        return out

    def predict_proba(self, scores, covar=None):
        """Predicted probability; ``family="binomial"`` only."""
        if self.family != "binomial":
            raise ValueError("predict_proba is only defined for "
                             "family='binomial'")
        eta = self.predict(scores, covar)
        return 1.0 / (1.0 + np.exp(-eta))

    # -- inspect ---------------------------------------------------------
    @property
    def n_selected(self):
        """Number of input scores with a non-zero coefficient."""
        return int(np.count_nonzero(self.beta))

    @property
    def n_folds_used(self):
        """Fold coefficient vectors that entered the returned CMSA average."""
        return int(sum(f.used for f in self.folds))

    @property
    def cv_r2(self):
        """Nested-CV predictive R² gain from the penalized scores.

        This is ``(SSE_baseline - SSE_inner_CMSA) / SST`` over untouched outer
        folds.
        The baseline contains covariates and any explicitly unpenalized scores.
        It is a predictive loss gain, not :func:`multipgs.incremental_r2`, which
        recalibrates the supplied score by OLS in the assessment cohort.

        Every individual is scored by an inner CMSA fit built entirely inside
        that individual's outer-training set. Neither its coefficients, penalty
        choices nor penalty grid saw the outer assessment phenotype.

        It is still not a substitute for an independent cohort. Because the
        nested estimators train on fewer rows, the estimate may be conservative
        under a stable learning curve, but the direction is not guaranteed. It
        also cannot detect overlap with the GWAS behind the input scores.

        ``None`` for a binomial fit; use ``log["cv_loss"]`` there (a deviance).
        """
        return self.log.get("cv_r2")

    def selected(self, top=None):
        """Non-zero scores, ordered by ``|beta_std|`` (largest first).

        Returns a list of ``(score_id, beta_std, beta)``. The standardized
        coefficient is the comparable one — raw coefficients depend on whatever
        scale each input score happened to be on.
        """
        nz = np.flatnonzero(self.beta)
        order = nz[np.argsort(-np.abs(self.beta_std[nz]))]
        if top is not None:
            order = order[:int(top)]
        return [(self.score_ids[j], float(self.beta_std[j]),
                 float(self.beta[j])) for j in order]

    def summary(self):
        """One-paragraph human summary of the fit."""
        used = self.n_folds_used
        lines = [
            f"multi-PGS ({self.family}): {self.n_selected} of "
            f"{len(self.beta)} scores selected, averaged over {used} of "
            f"{len(self.folds)} CMSA folds.",
        ]
        if self.log.get("cv_r2") is not None:
            lines.append(f"  cross-validated R2: {self.log['cv_r2']:.4f}")
        if "null_model" in self.log:
            lines.append(
                "  Nested cross-validation did not establish incremental "
                "signal: the returned fit is the unpenalized baseline (the "
                "null incremental model). Treat "
                "that as 'the penalized scores did not add prediction here'.")
        else:
            top = self.selected(top=5)
            if top:
                lines.append("  Top scores by |beta_std|: " + ", ".join(
                    f"{sid} ({b:+.3g})" for sid, b, _ in top))
        for key in ("dropped_constant", "imputed_missing"):
            n = self.log.get(key)
            if n:
                lines.append(f"  {key.replace('_', ' ')}: {n}")
        return "\n".join(lines)

    # -- internals -------------------------------------------------------
    @staticmethod
    def _prepare(a, expect, name):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] != expect:
            raise ValueError(f"{name} must have {expect} columns, got "
                             f"{a.shape}")
        return a


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _as_2d(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {a.shape}")
    return a


def _handle_missing(X, missing, name):
    bad = ~np.isfinite(X)
    if not bad.any():
        return X, 0
    if missing == "raise":
        cols = np.flatnonzero(bad.any(axis=0))
        raise ValueError(
            f"{name} has non-finite values in {cols.size} column(s) "
            f"(first: {cols[:5].tolist()}); pass missing='mean' to fill each "
            f"column with its observed mean, or drop those rows yourself")
    if missing != "mean":
        raise ValueError("missing must be 'raise' or 'mean'")
    X = X.copy()
    for j in np.flatnonzero(bad.any(axis=0)):
        col = X[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            raise ValueError(f"{name} column {j} is entirely non-finite")
        col[~ok] = col[ok].mean()
    return X, int(bad.sum())


def _standardize(X, idx):
    """Column means and standard deviations over rows ``idx``.

    A constant column gets ``scale = 1``, so it standardizes to all-zero and
    can never enter the model. That is the intended handling: a score with no
    variance in the training set carries no information, and dividing by its
    zero standard deviation would poison the whole fit.
    """
    sub = X if idx is None else X[idx]
    center = sub.mean(axis=0)
    scale = sub.std(axis=0)
    dead = scale <= 1e-12
    scale = np.where(dead, 1.0, scale)
    return center, scale, dead


def _gaussian_stats(X, y, idx=None):
    """Stable sufficient statistics for a Gaussian fit on ``idx``.

    All moments use the first row as a fixed origin. Held-out statistics formed
    with the same origin can therefore be subtracted from their parent without
    the catastrophic cancellation of raw ``E[X**2] - E[X]**2`` moments. Row
    chunks bound the temporary allocation while BLAS still forms each Gram.
    """
    return _gaussian_stats_at_origin(X, y, idx)


def _gaussian_stats_at_origin(X, y, idx=None, *, reference=None):
    """Implementation allowing child statistics to share a parent origin."""
    rows = None if idx is None else np.asarray(idx, dtype=np.int64).ravel()
    n = int(X.shape[0] if rows is None else rows.size)
    if n == 0:
        raise ValueError("cannot form Gaussian statistics from zero rows")
    first = 0 if rows is None else int(rows[0])
    if reference is None:
        origin_x = np.array(X[first], dtype=float, copy=True)
        origin_y = float(y[first])
    else:
        origin_x = reference["origin_x"]
        origin_y = reference["origin_y"]

    D = X.shape[1]
    sum_xc = np.zeros(D)
    xtx_c = np.zeros((D, D))
    sum_yc = 0.0
    xty_c = np.zeros(D)
    step = max(1, 2_000_000 // max(D, 1))
    for start in range(0, n, step):
        stop = min(start + step, n)
        take = slice(start, stop) if rows is None else rows[start:stop]
        Xc = np.array(X[take], dtype=float, copy=True)
        Xc -= origin_x
        yc = np.asarray(y[take], dtype=float) - origin_y
        sum_xc += np.sum(Xc, axis=0)
        xtx_c += Xc.T @ Xc
        sum_yc += float(np.sum(yc))
        xty_c += Xc.T @ yc
    return {
        "n": n,
        "origin_x": origin_x,
        "sum_xc": sum_xc,
        "xtx_c": (xtx_c + xtx_c.T) * 0.5,
        "origin_y": origin_y,
        "sum_yc": sum_yc,
        "xty_c": xty_c,
    }


def _subtract_gaussian_stats(total, held_out):
    """Sufficient statistics for ``total \\ held_out``."""
    n = int(total["n"] - held_out["n"])
    if n <= 0:
        raise ValueError("held-out rows exhaust the Gaussian training set")
    if (not np.array_equal(total["origin_x"], held_out["origin_x"])
            or total["origin_y"] != held_out["origin_y"]):
        raise ValueError("Gaussian statistics must share a centering origin")
    return {
        "n": n,
        "origin_x": total["origin_x"],
        "sum_xc": total["sum_xc"] - held_out["sum_xc"],
        "xtx_c": total["xtx_c"] - held_out["xtx_c"],
        "origin_y": total["origin_y"],
        "sum_yc": float(total["sum_yc"] - held_out["sum_yc"]),
        "xty_c": total["xty_c"] - held_out["xty_c"],
    }


def _gaussian_system(stats):
    """Standardization, Gram and response covariance from raw statistics."""
    n = stats["n"]
    mean_xc = stats["sum_xc"] / n
    center = stats["origin_x"] + mean_xc
    var = np.diag(stats["xtx_c"]) / n - mean_xc * mean_xc
    # Subtraction of two large cross-products can leave tiny negative round-off.
    scale0 = np.sqrt(np.maximum(var, 0.0))
    dead = scale0 <= 1e-12
    scale = np.where(dead, 1.0, scale0)
    cov = stats["xtx_c"] / n - np.outer(mean_xc, mean_xc)
    G = cov / scale[:, None] / scale[None, :]
    mean_yc = stats["sum_yc"] / n
    ybar = stats["origin_y"] + mean_yc
    r = (stats["xty_c"] / n - mean_xc * mean_yc) / scale
    if dead.any():
        G[dead, :] = 0.0
        G[:, dead] = 0.0
        r[dead] = 0.0
    G = (G + G.T) * 0.5
    return center, scale, dead, G, r, float(ybar)


def _folds(n, n_folds, rng, stratify=None):
    order = rng.permutation(n)
    if stratify is None:
        return np.array_split(order, n_folds)
    # Keep the case fraction roughly equal across folds: interleave each
    # class's shuffled members so a rare-disease fold cannot come out with no
    # cases at all, which would make its validation deviance meaningless.
    parts = [[] for _ in range(n_folds)]
    at = 0
    for value in np.unique(stratify):
        members = order[stratify[order] == value]
        for i, m in enumerate(members):
            parts[(at + i) % n_folds].append(m)
        at = (at + len(members)) % n_folds
    return [np.array(sorted(p), dtype=np.int64) for p in parts]


def _gaussian_loss(y, pred):
    d = y - pred
    return float(np.mean(d * d))


def _binomial_loss(y, eta):
    # Mean deviance, computed through logaddexp so a confident wrong call
    # costs a large finite number rather than an inf.
    return float(2.0 * np.mean(np.logaddexp(0.0, eta) - y * eta))


def _block_losses(Xval, yval, coefs, intercepts, gaussian):
    """Held-out loss at every penalty in one block of a fitted path.

    ``intercepts`` is the single Gaussian mean or one intercept per penalty.
    The predictions come from a single ``(n_val, K) x (K, L)`` product rather
    than ``L`` separate matrix-vector calls, and the reductions match
    :func:`_gaussian_loss` and :func:`_binomial_loss` term for term.
    """
    if coefs.shape[0] == 0:
        return np.zeros(0)
    eta = Xval @ coefs.T
    eta += np.asarray(intercepts, dtype=float).reshape(1, -1) \
        if np.ndim(intercepts) else float(intercepts)
    if gaussian:
        resid = yval[:, None] - eta
        return np.mean(resid * resid, axis=0)
    return 2.0 * np.mean(np.logaddexp(0.0, eta) - yval[:, None] * eta, axis=0)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def multi_pgs_fit(scores, y, *, covar=None, family="gaussian", alpha=1.0,
                  n_folds=10, assessment_folds=5, n_lambda=100,
                  lambda_min_ratio=None, n_abort=10, penalty_factor=None,
                  unpenalized_scores=None, dfmax=None, score_ids=None,
                  covar_ids=None, missing="raise", seed=None, tol=1e-7,
                  max_iter=1000):
    """Fit a multi-PGS combination by cross-model selection and averaging.

    Parameters
    ----------
    scores : array, ``(n, K)``
        Polygenic scores for the training individuals. Column order defines
        the coefficient order in the result. Include the target trait's own
        score here — it is not special-cased.
    y : array, ``(n,)``
        Target phenotype. Coded 0/1 for ``family="binomial"``.
    covar : array, ``(n, P)``, optional
        Covariates (age, sex, genotyping batch, principal components). They are
        fitted **unpenalized inside the same regression**. For Gaussian loss
        this is algebraically equivalent to residualizing within each training
        fold, while retaining coefficients and predictions on the original
        scale; joint fitting extends the same API to binomial loss.
    family : {"gaussian", "binomial"}
        Squared-error or logistic loss.
    alpha : float or sequence of float
        Elastic-net mixing: 1.0 is the lasso, 0.0 ridge. A sequence is a grid,
        and each fold picks its own best value from it. The sparse lasso is the
        package default; the source study used its own ``cv.glmnet`` tuning.
    n_folds : int
        CMSA folds. 10 is the ``bigstatsr`` default.
    assessment_folds : int
        Outer folds for the nested assessment and its inner CMSA fits. This is
        separate from ``n_folds`` so nested assessment does not make the final
        estimator needlessly expensive; 5 is normally enough for the gate.
    n_lambda, lambda_min_ratio : int, float
        Penalty grid. The returned CMSA uses one grid computed on the full
        training set so its fold results are comparable; every nested outer
        assessment constructs a separate grid from its training rows only.
    n_abort : int
        Stop walking a fold's path after this many consecutive penalties fail
        to improve its held-out loss.
    penalty_factor : array ``(K,)``, optional
        Per-score multiplier on the penalty. ``0`` forces a score to stay in
        the model.
    unpenalized_scores : sequence, optional
        Convenience for the above: score indices or ids to leave unpenalized.
        The usual use is the target trait's own score. It then remains in the
        fitted baseline if the penalized additions fail the nested gate; this
        does not guarantee that additions improve an independent cohort.
    dfmax : int, optional
        Abandon a path once it holds more than this many non-zero scores.
    score_ids, covar_ids : sequence, optional
        Names, for reporting. Default to ``score_0 ...`` and ``covar_0 ...``.
    missing : {"raise", "mean"}
        What to do about non-finite entries in ``scores``/``covar``.
    seed : int, optional
        Fold assignment. Set it if you need the fit to be reproducible.

    Returns
    -------
    MultiPGSFit
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")

    S = _as_2d(scores, "scores")
    n, K = S.shape
    y = np.asarray(y, dtype=float).ravel()
    if y.shape != (n,):
        raise ValueError(f"y must have length {n}, got {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values; drop those individuals")
    if family == "binomial":
        vals = np.unique(y)
        if not np.all(np.isin(vals, (0.0, 1.0))):
            raise ValueError("family='binomial' needs y coded 0/1, found "
                             f"values {vals[:5]}")
        if vals.size < 2:
            raise ValueError("y has only one class")

    S_unimputed = S
    S, n_imp = _handle_missing(S, missing, "scores")
    if covar is None:
        C = np.zeros((n, 0))
        n_imp_c = 0
    else:
        C = _as_2d(covar, "covar")
        if C.shape[0] != n:
            raise ValueError(f"covar has {C.shape[0]} rows, scores has {n}")
        C_unimputed = C
        C, n_imp_c = _handle_missing(C, missing, "covar")
    if covar is None:
        C_unimputed = C
    P = C.shape[1]

    score_ids = (np.array([f"score_{j}" for j in range(K)], dtype=object)
                 if score_ids is None else np.asarray(score_ids, dtype=object))
    covar_ids = (np.array([f"covar_{j}" for j in range(P)], dtype=object)
                 if covar_ids is None else np.asarray(covar_ids, dtype=object))
    if score_ids.shape != (K,) or covar_ids.shape != (P,):
        raise ValueError("score_ids/covar_ids must match the column counts")
    for ids, name in ((score_ids, "score_ids"), (covar_ids, "covar_ids")):
        normalized = [str(value) for value in ids]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must be unique after string conversion")

    # Penalty factors: scores as asked, covariates always free.
    if penalty_factor is None:
        pf_s = np.ones(K)
    else:
        pf_s = np.asarray(penalty_factor, dtype=float).ravel()
        if pf_s.shape != (K,):
            raise ValueError(f"penalty_factor must have length {K}")
        if np.any(pf_s < 0) or not np.all(np.isfinite(pf_s)):
            raise ValueError("penalty_factor must be finite and non-negative")
    if unpenalized_scores is not None:
        pf_s = pf_s.copy()
        pf_s[_resolve_columns(unpenalized_scores, score_ids, K)] = 0.0
    if not np.any(pf_s > 0):
        raise ValueError("every score is unpenalized; there is nothing for "
                         "the penalty to select over")

    # Do not copy the dominant n x K allocation when there are no covariates.
    X = S if P == 0 else np.hstack([S, C])
    # Nested assessment must learn imputation means without looking at outer
    # validation rows. Keep the original non-finite matrix only when needed;
    # the final estimator still uses means learned from all of its training data.
    X_unimputed = None
    if n_imp + n_imp_c:
        X_unimputed = (S_unimputed if P == 0
                       else np.hstack([S_unimputed, C_unimputed]))
    pf = np.concatenate([pf_s, np.zeros(P)])

    alphas = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    if (alphas.size == 0 or not np.all(np.isfinite(alphas))
            or np.any(alphas < 0) or np.any(alphas > 1)):
        raise ValueError("alpha must be finite, non-empty and lie in [0, 1]")

    n_folds = int(n_folds)
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be in [2, n={n}], got {n_folds}")
    assessment_folds = int(assessment_folds)
    if assessment_folds < 2:
        raise ValueError("assessment_folds must be at least 2")
    if n < 3:
        raise ValueError("nested assessment needs at least 3 individuals")
    n_assess = min(assessment_folds, n)
    min_outer_train = n - (n + n_assess - 1) // n_assess
    assessment_inner = min(n_folds, n_assess, min_outer_train)
    if assessment_inner < 2:
        raise ValueError(
            "assessment_folds leaves fewer than 2 outer-training individuals")

    rng = np.random.default_rng(seed)
    parts = _folds(n, n_folds, rng, stratify=y if family == "binomial" else None)

    # One penalty grid for every returned-CMSA fold, measured on the full data at
    # the unpenalized baseline. cv.glmnet does the same: fold-specific grids would
    # not be comparable, and averaging coefficients across them would be
    # averaging across different problems.
    gaussian_stats = _gaussian_stats(X, y) if family == "gaussian" else None
    if gaussian_stats is not None:
        center, scale, dead, _, _, _ = _gaussian_system(gaussian_stats)
        lambdas = _lambda_grid_for_gaussian_stats(
            gaussian_stats, pf, alphas.min(), n_lambda, lambda_min_ratio, K)
    else:
        center, scale, dead = _standardize(X, None)
        Xs_full = (X - center) / scale
        lambdas = _lambda_grid_for(Xs_full, y, pf, family, alphas.min(),
                                   n_lambda, lambda_min_ratio, K)

    fits = []
    beta_sum = np.zeros(K + P)
    intercept_sum = 0.0
    for k, val_idx in enumerate(parts):
        tr_idx = _complement(n, val_idx)
        if tr_idx.size == 0 or val_idx.size == 0:
            continue
        if gaussian_stats is None:
            tr_stats = None
        else:
            held_stats = _gaussian_stats_at_origin(
                X, y, val_idx, reference=gaussian_stats)
            tr_stats = _subtract_gaussian_stats(gaussian_stats, held_stats)
        best = _fit_one_fold(X, y, tr_idx, val_idx, pf, alphas, lambdas,
                             family, n_abort, dfmax, tol, max_iter,
                             gaussian_stats=tr_stats)
        fits.append(FoldFit(fold=k, alpha=best["alpha"], lam=best["lam"],
                            lam_index=best["lam_index"], loss=best["loss"],
                            null_loss=best["null_loss"],
                            n_nonzero=int(np.count_nonzero(best["beta"][:K])),
                            used=False))
        # Ordinary CMSA averages every fold-selected vector. Filtering this sum
        # by whether a fold happened to beat its own null is itself selection on
        # the validation data and biases the returned model.
        beta_sum += best["beta"]
        intercept_sum += best["intercept"]

    if not fits:
        raise RuntimeError("no non-empty CMSA folds were fitted")
    cmsa_beta = beta_sum / len(fits)
    cmsa_intercept = intercept_sum / len(fits)

    cv_loss, cv_null_loss, cv_gain_se, inner_folds = _nested_cv_assessment(
        X, y, pf, alphas, family, n_assess, assessment_inner, n_lambda,
        lambda_min_ratio, n_abort, dfmax, tol, max_iter, K, seed,
        gaussian_stats=gaussian_stats, X_unimputed=X_unimputed)

    # The outer folds are untouched by grid construction, tuning and fitting.
    # A one-standard-error gate avoids turning a chance-positive null estimate
    # into a dense deployed model. The scale-relative floor prevents arithmetic
    # noise between two effectively exact baseline fits from passing the gate.
    loss_scale = (float(np.var(y)) if family == "gaussian" else 1.0)
    gain_threshold = max(cv_gain_se, 1e-12 * max(loss_scale, 1.0))
    gate_passed = (np.isfinite(cv_loss) and np.isfinite(cv_null_loss)
                   and (cv_null_loss - cv_loss) > gain_threshold)
    if gate_passed:
        beta_raw = cmsa_beta
        intercept = cmsa_intercept
        for f in fits:
            f.used = True
    else:
        beta_raw, intercept = _fit_unpenalized_baseline(
            X, y, np.arange(n), pf, family, gaussian_stats=gaussian_stats)

    n_used = sum(f.used for f in fits)

    log = {
        "n": int(n), "n_scores": int(K), "n_covar": int(P),
        "n_folds": len(fits), "n_folds_used": int(n_used),
        "assessment_folds": int(n_assess),
        "assessment_inner_folds": int(inner_folds),
        "cv_scheme": "nested_cmsa",
        "family": family, "alphas": alphas.tolist(),
        "n_lambda": int(lambdas.size),
        "lambda_max": float(lambdas[0]), "lambda_min": float(lambdas[-1]),
        "dropped_constant": int(dead[:K].sum()),
        "imputed_missing": int(n_imp + n_imp_c),
        "unpenalized_scores": int(np.count_nonzero(pf_s == 0)),
        "cv_loss": float(cv_loss) if np.isfinite(cv_loss) else None,
        "cv_null_loss": (float(cv_null_loss) if np.isfinite(cv_null_loss)
                         else None),
        "cv_gain_se": (float(cv_gain_se) if np.isfinite(cv_gain_se) else None),
        "cv_gain_threshold": (float(gain_threshold)
                              if np.isfinite(gain_threshold) else None),
    }
    if family == "gaussian" and np.isfinite(cv_loss):
        # Predictive SSE gain over the explicit unpenalized baseline. Unlike
        # incremental_r2(), this does not recalibrate predictions on assessment
        # outcomes; that is what keeps outer assessment outcomes untouched.
        var_y = float(np.mean((y - y.mean()) ** 2))
        log["cv_r2"] = (float((cv_null_loss - cv_loss) / var_y)
                        if var_y > 0 else 0.0)
    if not gate_passed:
        log["null_model"] = (
            "nested cross-validation did not establish incremental signal; "
            "returned the full-data unpenalized baseline")

    return MultiPGSFit(
        beta=beta_raw[:K], intercept=float(intercept),
        covar_beta=beta_raw[K:], beta_std=beta_raw[:K] * scale[:K],
        center=center, scale=scale, score_ids=score_ids, covar_ids=covar_ids,
        family=family, folds=fits, log=log)


def _complement(n, idx):
    keep = np.ones(n, dtype=bool)
    keep[np.asarray(idx, dtype=int)] = False
    return np.flatnonzero(keep)


def _impute_from_training(X, train):
    """Mean-impute every row using means learned only on ``train``."""
    out = np.array(X, dtype=float, copy=True)
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    observed = out[train]
    finite = np.isfinite(observed)
    counts = np.sum(finite, axis=0)
    sums = np.sum(np.where(finite, observed, 0.0), axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    # A column absent from this training split is constant for fitting purposes.
    # Zero is arbitrary but finite; its fitted coefficient is necessarily zero.
    rows, cols = np.nonzero(bad)
    out[rows, cols] = means[cols]
    return out


def _fit_unpenalized_baseline(X, y, idx, pf, family, *, gaussian_stats=None):
    """Fit intercept plus every ``pf == 0`` column on ``idx``.

    The returned coefficients are on the raw input scale. This is deliberately
    independent of the elastic-net grid: ridge has no finite lambda at which its
    penalized coefficients are exactly zero.
    """
    if family == "gaussian":
        stats = (_gaussian_stats(X, y, idx) if gaussian_stats is None
                 else gaussian_stats)
        center, scale, _, G, r, ybar = _gaussian_system(stats)
        coef, _ = _coord.unpenalized_fit(G, r, pf)
        beta = coef / scale
        return beta, float(ybar - np.dot(beta, center))

    center, scale, _ = _standardize(X, idx)
    Xs = np.ascontiguousarray((X[idx] - center) / scale)
    b0, coef = _binomial_baseline(Xs, y[idx], pf)
    beta = coef / scale
    return beta, float(b0 - np.dot(beta, center))


def _nested_cv_assessment(X, y, pf, alphas, family, n_outer, n_inner,
                          n_lambda, lambda_min_ratio, n_abort, dfmax, tol,
                          max_iter, K, seed, *, gaussian_stats=None,
                          X_unimputed=None):
    """Nested outer-fold assessment of an inner CMSA estimator.

    For each outer fold, all grid construction, fold selection and coefficient
    averaging happens inside its training rows. The untouched outer rows score
    both that inner CMSA and an explicit unpenalized baseline. The standard
    error is across outer-fold mean loss gains and powers the conservative gate.
    """
    n = y.size
    rng = np.random.default_rng(seed)
    outer_parts = _folds(
        n, n_outer, rng, stratify=y if family == "binomial" else None)
    total_loss = 0.0
    total_null = 0.0
    fold_gains = []
    inner_used = None

    for val in outer_parts:
        tr = _complement(n, val)
        if val.size == 0 or tr.size < 2:
            return np.inf, np.inf, np.inf, 0
        Xk = X
        if X_unimputed is not None:
            Xk = _impute_from_training(X_unimputed, tr)
        # Assessment preprocessing must be a function of outer-training rows
        # alone. Re-forming this parent statistic is deliberate: subtracting a
        # held-out statistic is algebraically equivalent, but its finite-
        # precision result can still depend on the outer phenotype through the
        # common numerical origin. The direct statistic makes the independence
        # contract exact, not merely true in real arithmetic.
        tr_stats = (_gaussian_stats(Xk, y, tr)
                    if gaussian_stats is not None else None)
        if tr_stats is not None:
            lambdas = _lambda_grid_for_gaussian_stats(
                tr_stats, pf, alphas.min(), n_lambda, lambda_min_ratio, K)
        else:
            c, s, _ = _standardize(Xk, tr)
            Xs = (Xk[tr] - c) / s
            lambdas = _lambda_grid_for(
                Xs, y[tr], pf, family, alphas.min(), n_lambda,
                lambda_min_ratio, K)

        baseline_beta, baseline_intercept = _fit_unpenalized_baseline(
            Xk, y, tr, pf, family, gaussian_stats=tr_stats)
        pred0 = baseline_intercept + Xk[val] @ baseline_beta
        null_loss = (_gaussian_loss(y[val], pred0) if family == "gaussian"
                     else _binomial_loss(y[val], pred0))

        ni = min(int(n_inner), tr.size)
        if ni < 2:
            return np.inf, np.inf, np.inf, 0
        inner_used = ni if inner_used is None else min(inner_used, ni)
        inner_local = _folds(
            tr.size, ni, rng,
            stratify=y[tr] if family == "binomial" else None)
        beta_sum = np.zeros(Xk.shape[1])
        intercept_sum = 0.0
        fitted = 0
        for local_val in inner_local:
            inner_val = tr[local_val]
            inner_tr = tr[_complement(tr.size, local_val)]
            if inner_val.size == 0 or inner_tr.size == 0:
                continue
            if tr_stats is None:
                train_stats = None
            else:
                held_stats = _gaussian_stats_at_origin(
                    Xk, y, inner_val, reference=tr_stats)
                train_stats = _subtract_gaussian_stats(tr_stats, held_stats)
            best = _fit_one_fold(
                Xk, y, inner_tr, inner_val, pf, alphas, lambdas, family,
                n_abort, dfmax, tol, max_iter, gaussian_stats=train_stats)
            beta_sum += best["beta"]
            intercept_sum += best["intercept"]
            fitted += 1
        if fitted < 2:
            return np.inf, np.inf, np.inf, 0

        beta = beta_sum / fitted
        intercept = intercept_sum / fitted
        pred = intercept + Xk[val] @ beta
        loss = (_gaussian_loss(y[val], pred) if family == "gaussian"
                else _binomial_loss(y[val], pred))
        total_loss += val.size * loss
        total_null += val.size * null_loss
        fold_gains.append(null_loss - loss)

    gains = np.asarray(fold_gains, dtype=float)
    gain_se = (float(np.std(gains, ddof=1) / np.sqrt(gains.size))
               if gains.size > 1 else np.inf)
    return (total_loss / n, total_null / n, gain_se,
            int(inner_used or 0))


def _resolve_columns(sel, ids, K):
    """Turn indices, ids or a boolean mask into an integer index array."""
    sel = np.asarray(sel)
    if sel.dtype == bool:
        if sel.shape != (K,):
            raise ValueError(f"boolean selection must have length {K}")
        return np.flatnonzero(sel)
    if np.issubdtype(sel.dtype, np.integer):
        if np.any(sel < 0) or np.any(sel >= K):
            raise ValueError("score index out of range")
        return sel.astype(int)
    lookup = {str(v): i for i, v in enumerate(ids)}
    try:
        return np.array([lookup[str(v)] for v in np.atleast_1d(sel)], dtype=int)
    except KeyError as exc:
        raise ValueError(f"unknown score id {exc.args[0]!r}") from None


def _lambda_grid_for(Xs, y, pf, family, alpha, n_lambda, ratio, K):
    n = Xs.shape[0]
    if family == "gaussian":
        yc = y - y.mean()
        G = Xs.T @ Xs / n
        G = (G + G.T) * 0.5
        r = Xs.T @ yc / n
        _, grad = _coord.unpenalized_fit(G, r, pf)
    else:
        # The IRLS gradient at the unpenalized baseline. Weights there are
        # p(1-p) with p the fitted baseline probability; the
        # working gradient reduces to X^T (y - p) / n.
        p = _null_probabilities(Xs, y, pf)
        grad = Xs.T @ (y - p) / n
    return _coord.lambda_grid(grad, pf, alpha, n_lambda=n_lambda,
                              lambda_min_ratio=ratio, n=n, n_penalized=K)


def _lambda_grid_for_gaussian_stats(stats, pf, alpha, n_lambda, ratio, K):
    """Gaussian penalty grid without materializing standardized rows."""
    _, _, _, G, r, _ = _gaussian_system(stats)
    _, grad = _coord.unpenalized_fit(G, r, pf)
    return _coord.lambda_grid(
        grad, pf, alpha, n_lambda=n_lambda, lambda_min_ratio=ratio,
        n=stats["n"], n_penalized=K)


def _binomial_baseline(Xs, y, pf, *, max_iter=50, tol=1e-9):
    """Intercept and standardized coefficients for the unpenalized model."""
    n = Xs.shape[0]
    free = np.flatnonzero(pf <= 0.0)
    D = np.hstack([np.ones((n, 1)), Xs[:, free]])
    b = np.zeros(D.shape[1])
    ybar = min(max(float(y.mean()), 1e-6), 1 - 1e-6)
    b[0] = np.log(ybar / (1 - ybar))
    for _ in range(max_iter):
        eta = D @ b
        p = np.empty_like(eta)
        pos = eta >= 0
        p[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        ex = np.exp(eta[~pos])
        p[~pos] = ex / (1.0 + ex)
        np.clip(p, 1e-9, 1 - 1e-9, out=p)
        w = np.maximum(p * (1 - p), 1e-5)
        H = D.T @ (D * w[:, None])
        H[np.diag_indices_from(H)] += 1e-10
        try:
            step = np.linalg.solve(H, D.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < tol:
            break
    coef = np.zeros(Xs.shape[1])
    coef[free] = b[1:]
    return float(b[0]), coef


def _null_probabilities(Xs, y, pf, *, max_iter=50, tol=1e-9):
    """Fitted probabilities of the intercept + unpenalized-columns model."""
    b0, beta = _binomial_baseline(
        Xs, y, pf, max_iter=max_iter, tol=tol)
    eta = b0 + Xs @ beta
    p = np.empty_like(eta)
    pos = eta >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ex = np.exp(eta[~pos])
    p[~pos] = ex / (1.0 + ex)
    return np.clip(p, 1e-9, 1 - 1e-9)


def _fit_one_fold(X, y, tr, val, pf, alphas, lambdas, family, n_abort, dfmax,
                  tol, max_iter, *, gaussian_stats=None):
    """Sweep the (alpha, lambda) grid on ``tr``, select on ``val``.

    Returns the raw-scale coefficients and intercept at the selected point plus
    the explicit unpenalized-baseline loss. The baseline is a candidate even for
    pure ridge, whose finite-lambda path never contains an exact null model.
    """
    gaussian = family == "gaussian"
    if gaussian:
        stats = (_gaussian_stats(X, y, tr) if gaussian_stats is None
                 else gaussian_stats)
        center, scale, _, G, r, ybar = _gaussian_system(stats)
        base_coef, _ = _coord.unpenalized_fit(G, r, pf)
        Xtr = ytr = None
        base_b0 = ybar
    else:
        center, scale, _ = _standardize(X, tr)
        # The solver wants columns contiguous. Preparing that layout here once
        # avoids recopying the whole training matrix for every path block.
        Xtr = np.asfortranarray((X[tr] - center) / scale)
        ytr = y[tr]
        base_b0, base_coef = _binomial_baseline(Xtr, ytr, pf)
    Xval = (X[val] - center) / scale
    yval = y[val]

    pred0 = base_b0 + Xval @ base_coef
    null_loss = (_gaussian_loss(yval, pred0) if gaussian
                 else _binomial_loss(yval, pred0))
    best = {"loss": null_loss, "alpha": float(alphas[0]),
            "alpha_index": 0, "lam": float("inf"), "lam_index": -1,
            "coef": base_coef.copy(), "b0": float(base_b0)}
    for a_idx, a in enumerate(alphas):
        # Warm-start state carried down the grid, and across blocks of it.
        if gaussian:
            beta_w, grad_w = _coord.unpenalized_fit(G, r, pf)
        else:
            beta_w, b0_w = None, None

        # Walk the grid in blocks, stopping as soon as `n_abort` consecutive
        # penalties fail to improve the held-out loss. Warm starts carry
        # across blocks, so this costs nothing over fitting the path at once.
        block = max(int(n_abort), 1)
        start = 0
        since_best = 0
        while start < lambdas.size:
            lams = lambdas[start:start + block]
            if gaussian:
                coefs, nf = _coord.enet_path_gaussian(
                    G, r, pf=pf, alpha=a, lambdas=lams, beta_init=beta_w,
                    grad_init=grad_w, tol=tol, max_iter=max_iter, dfmax=dfmax)
                b0s = None
            else:
                b0s, coefs, nf = _coord.enet_path_binomial(
                    Xtr, ytr, pf=pf, alpha=a, lambdas=lams, beta_init=beta_w,
                    b0_init=b0_w, tol=tol, max_iter=max(20, max_iter // 10),
                    dfmax=dfmax)
            n_ok = min(nf, lams.size)

            # One BLAS-3 product for the whole block's held-out predictions.
            # Scoring each penalty separately issued n_ok matrix-vector calls
            # over the same Xval; batching them is the same arithmetic at about
            # six times the throughput. Selection stays sequential below, so
            # the abort counter and tie-breaking are unchanged.
            block_losses = _block_losses(
                Xval, yval, coefs[:n_ok], ybar if gaussian else b0s[:n_ok],
                gaussian)
            for i in range(n_ok):
                coef = coefs[i]
                b0 = ybar if gaussian else float(b0s[i])
                lo = float(block_losses[i])
                if lo < best["loss"] - 1e-15:
                    best = {"loss": lo, "alpha": float(a),
                            "alpha_index": a_idx, "lam": float(lams[i]),
                            "lam_index": start + i, "coef": coef.copy(),
                            "b0": b0}
                    since_best = 0
                else:
                    since_best += 1

            if since_best >= n_abort or n_ok < lams.size:
                break
            beta_w = coefs[n_ok - 1].copy()
            if gaussian:
                grad_w = r - G @ beta_w
            else:
                b0_w = float(b0s[n_ok - 1])
            start += block

    beta_raw = best["coef"] / scale
    intercept = best["b0"] - float(np.dot(beta_raw, center))
    return {"beta": beta_raw, "intercept": intercept, "loss": best["loss"],
            "null_loss": float(null_loss), "alpha": best["alpha"],
            "alpha_index": best["alpha_index"], "lam": best["lam"],
            "lam_index": best["lam_index"], "n_val": int(val.size)}
