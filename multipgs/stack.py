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

from ._cmsa import (
    _complement,
    _empty_solver_info,
    _fit_one_fold,
    _fit_unpenalized_baseline,
    _folds,
    _lambda_grid_for,
    _lambda_grid_for_gaussian_stats,
    _merge_solver_info,
    _nested_cv_assessment,
)
from ._stats import (
    _gaussian_stats,
    _gaussian_stats_at_origin,
    _gaussian_system,
    _standardize,
    _subtract_gaussian_stats,
)
from ._validate import _nonnegative_integer, _positive_integer

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
    ``converged`` is false when the fold's baseline or any fitted path point
    exhausted its numerical iterations; ``n_iteration_exhausted`` counts those
    path points.
    """

    fold: int
    alpha: float
    lam: float
    lam_index: int
    loss: float
    null_loss: float
    n_nonzero: int
    used: bool
    converged: bool = True
    n_iteration_exhausted: int = 0

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
        if "convergence_warning" in self.log:
            lines.append("  Numerical warning: " + self.log["convergence_warning"])
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
        Penalty grid. The returned CMSA uses one grid per ``alpha``, computed
        on the full training set and shared by its folds. Every nested outer
        assessment constructs its alpha-specific grids from training rows only.
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
    tol : float
        Strictly positive coordinate-descent convergence tolerance.
    max_iter : int
        Positive maximum coordinate-descent sweeps per fitted penalty.

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

    n_folds = _positive_integer(n_folds, "n_folds")
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be in [2, n={n}], got {n_folds}")
    assessment_folds = _positive_integer(assessment_folds, "assessment_folds")
    if assessment_folds < 2:
        raise ValueError("assessment_folds must be at least 2")
    n_lambda = _positive_integer(n_lambda, "n_lambda")
    n_abort = _positive_integer(n_abort, "n_abort")
    max_iter = _positive_integer(max_iter, "max_iter")
    if dfmax is not None:
        dfmax = _nonnegative_integer(dfmax, "dfmax")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    if lambda_min_ratio is not None:
        lambda_min_ratio = float(lambda_min_ratio)
        if (not np.isfinite(lambda_min_ratio)
                or not 0.0 < lambda_min_ratio <= 1.0):
            raise ValueError("lambda_min_ratio must be finite and lie in (0, 1]")
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

    # One full-data penalty grid per alpha, shared by every returned-CMSA fold.
    # Lambda_max scales as 1 / alpha, so anchoring a mixed-alpha search at its
    # smallest alpha would leave the lasso path over-penalized (catastrophically
    # so when ridge is included). Fold-specific grids within an alpha would still
    # make its fold selections incomparable, hence the full-data anchors here.
    gaussian_stats = _gaussian_stats(X, y) if family == "gaussian" else None
    if gaussian_stats is not None:
        center, scale, dead, _, _, _ = _gaussian_system(gaussian_stats)
        lambda_grids = np.vstack([
            _lambda_grid_for_gaussian_stats(
                gaussian_stats, pf, a, n_lambda, lambda_min_ratio, K)
            for a in alphas])
    else:
        center, scale, dead = _standardize(X, None)
        Xs_full = (X - center) / scale
        lambda_grids = np.vstack([
            _lambda_grid_for(Xs_full, y, pf, family, a, n_lambda,
                             lambda_min_ratio, K)
            for a in alphas])

    fits = []
    cmsa_solver_info = _empty_solver_info()
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
        best = _fit_one_fold(X, y, tr_idx, val_idx, pf, alphas, lambda_grids,
                             family, n_abort, dfmax, tol, max_iter,
                             gaussian_stats=tr_stats)
        best_info = best.get("solver_info", _empty_solver_info())
        _merge_solver_info(cmsa_solver_info, best_info)
        fits.append(FoldFit(fold=k, alpha=best["alpha"], lam=best["lam"],
                            lam_index=best["lam_index"], loss=best["loss"],
                             null_loss=best["null_loss"],
                             n_nonzero=int(np.count_nonzero(best["beta"][:K])),
                             used=False,
                             converged=bool(best_info.get("converged", True)),
                             n_iteration_exhausted=int(best_info.get(
                                 "n_iteration_exhausted", 0))))
        # Ordinary CMSA averages every fold-selected vector. Filtering this sum
        # by whether a fold happened to beat its own null is itself selection on
        # the validation data and biases the returned model.
        beta_sum += best["beta"]
        intercept_sum += best["intercept"]

    if not fits:
        raise RuntimeError("no non-empty CMSA folds were fitted")
    cmsa_beta = beta_sum / len(fits)
    cmsa_intercept = intercept_sum / len(fits)

    assessment = _nested_cv_assessment(
        X, y, pf, alphas, family, n_assess, assessment_inner, n_lambda,
        lambda_min_ratio, n_abort, dfmax, tol, max_iter, K, seed,
        gaussian_stats=gaussian_stats, X_unimputed=X_unimputed,
        return_info=True)
    (cv_loss, cv_null_loss, cv_gain_se, inner_folds,
     assessment_solver_info) = assessment

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
        deployed_baseline_info = {"converged": True}
        for f in fits:
            f.used = True
    else:
        beta_raw, intercept, deployed_baseline_info = \
            _fit_unpenalized_baseline(
                X, y, np.arange(n), pf, family,
                gaussian_stats=gaussian_stats, return_info=True)

    n_used = sum(f.used for f in fits)
    solver_info = _empty_solver_info()
    _merge_solver_info(solver_info, cmsa_solver_info)
    _merge_solver_info(solver_info, assessment_solver_info)
    if not deployed_baseline_info["converged"]:
        solver_info["converged"] = False
        solver_info["n_baseline_not_converged"] += 1

    log = {
        "n": int(n), "n_scores": int(K), "n_covar": int(P),
        "n_folds": len(fits), "n_folds_used": int(n_used),
        "assessment_folds": int(n_assess),
        "assessment_inner_folds": int(inner_folds),
        "cv_scheme": "nested_cmsa",
        "family": family, "alphas": alphas.tolist(),
        "n_lambda": int(lambda_grids.shape[1]),
        "lambda_max": float(np.max(lambda_grids[:, 0])),
        "lambda_min": float(np.min(lambda_grids[:, -1])),
        "lambda_max_by_alpha": lambda_grids[:, 0].tolist(),
        "lambda_min_by_alpha": lambda_grids[:, -1].tolist(),
        "solver_converged": bool(solver_info["converged"]),
        "n_path_points_fitted": int(solver_info["n_path_points_fitted"]),
        "n_iteration_exhausted": int(solver_info["n_iteration_exhausted"]),
        "n_coordinate_descent_exhausted": int(
            solver_info["n_coordinate_descent_exhausted"]),
        "n_irls_exhausted": int(solver_info["n_irls_exhausted"]),
        "n_baseline_not_converged": int(
            solver_info["n_baseline_not_converged"]),
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
    if not solver_info["converged"]:
        log["convergence_warning"] = (
            "one or more numerical fits exhausted their iteration limit; "
            "inspect the solver counters and consider increasing max_iter")

    return MultiPGSFit(
        beta=beta_raw[:K], intercept=float(intercept),
        covar_beta=beta_raw[K:], beta_std=beta_raw[:K] * scale[:K],
        center=center, scale=scale, score_ids=score_ids, covar_ids=covar_ids,
        family=family, folds=fits, log=log)

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

