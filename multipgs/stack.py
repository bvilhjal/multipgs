"""Multi-PGS: combine many polygenic scores into one, by penalized regression.

This is the estimator of Albiñana et al., *Multi-PGS enhances polygenic
prediction by combining 937 polygenic scores*
(`Nat Commun 14, 4702, 2023 <https://doi.org/10.1038/s41467-023-40330-w>`_).
Given a training cohort with

* an ``n x K`` matrix of polygenic scores — the target trait's own score plus
  scores for however many other traits are available, and
* the target phenotype, plus the usual covariates,

it fits a penalized regression of the phenotype on the scores and returns one
combined score. Genetically correlated traits carry information the target
trait's own GWAS is too small to see, so the combination beats the single-trait
score, and beats it by most where the target GWAS is smallest.

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

* No separate tuning cohort is consumed — the whole training set is used, and
  every individual's phenotype is out-of-sample for the fold that selected the
  model it contributed to.
* Averaging over folds is a variance reduction. With ``K`` correlated scores
  and a lasso penalty, *which* of a set of near-duplicate scores gets picked is
  close to arbitrary; averaging spreads the weight over them instead of
  betting the model on one draw.

What this module does **not** do is protect you from sample overlap. If
individuals in the training cohort also contributed to the GWAS behind one of
the ``K`` input scores, that score is partly fitted to its own training data,
CMSA will happily reward it, and the resulting accuracy is optimistic. The bias
is proportional to the fraction of the assessment sample that was also in the
discovery sample, and overlap is frequently not visible in public summary
statistics at all (Wray et al., `Nat Rev Genet 14:507-515, 2013
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
    loss of the covariate-only model (the solution at the largest
    :math:`\\lambda`), so ``loss < null_loss`` is exactly the statement that the
    scores helped in this fold. A fold that fails that test is recorded with
    ``used=False`` and left out of the average.
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
        """Folds that beat their covariate-only model and entered the average."""
        return int(sum(f.used for f in self.folds))

    @property
    def cv_r2(self):
        """Cross-validated R² *of the scores*, pooled over the held-out folds.

        Incremental over the covariate-only model, so it is the same quantity
        :func:`multipgs.incremental_r2` reports in a held-out cohort, and
        directly comparable to it. Without covariates it reduces to the plain
        R².

        Every training individual is scored by the fold that held them out, at
        a penalty chosen by the *other* folds — so neither the coefficients nor
        the penalty saw the individual being scored. That makes this an
        out-of-sample number without a separate held-out cohort, which is the
        practical reason to use CMSA at all.

        It is still not a substitute for an independent cohort. It is slightly
        conservative about the returned model (which averages the folds, and is
        usually a little better than any single one), and it cannot see the
        failure that matters most: if this cohort overlaps the GWAS behind the
        input scores, every number here is inflated and no amount of internal
        cross-validation will say so.

        ``None`` for a binomial fit (use ``log["cv_loss"]``, a deviance) or
        when there were too few folds to hold one out.
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
                "  Cross-validation did not beat the covariate-only model: "
                "the returned model is null (beta == 0). Treat that as 'the "
                "scores did not predict here', not as a small effect.")
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
    sub = X[idx]
    center = sub.mean(axis=0)
    scale = sub.std(axis=0)
    dead = scale <= 1e-12
    scale = np.where(dead, 1.0, scale)
    return center, scale, dead


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


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def multi_pgs_fit(scores, y, *, covar=None, family="gaussian", alpha=1.0,
                  n_folds=10, n_lambda=100, lambda_min_ratio=None, n_abort=10,
                  penalty_factor=None, unpenalized_scores=None, dfmax=None,
                  score_ids=None, covar_ids=None, missing="raise", seed=None,
                  tol=1e-7, max_iter=1000):
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
        fitted **unpenalized inside the same regression**, which is not the
        same as regressing them out first — the scores are selected against
        what the covariates cannot already explain.
    family : {"gaussian", "binomial"}
        Squared-error or logistic loss.
    alpha : float or sequence of float
        Elastic-net mixing: 1.0 is the lasso, 0.0 ridge. A sequence is a grid,
        and each fold picks its own best value from it. The lasso default
        follows the paper.
    n_folds : int
        CMSA folds. 10 is the ``bigstatsr`` default.
    n_lambda, lambda_min_ratio : int, float
        Penalty grid, computed once on the full training set so that fold
        results are comparable.
    n_abort : int
        Stop walking a fold's path after this many consecutive penalties fail
        to improve its held-out loss.
    penalty_factor : array ``(K,)``, optional
        Per-score multiplier on the penalty. ``0`` forces a score to stay in
        the model.
    unpenalized_scores : sequence, optional
        Convenience for the above: score indices or ids to leave unpenalized.
        The usual use is the target trait's own score, so the combination can
        only add to it.
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

    S, n_imp = _handle_missing(S, missing, "scores")
    if covar is None:
        C = np.zeros((n, 0))
        n_imp_c = 0
    else:
        C = _as_2d(covar, "covar")
        if C.shape[0] != n:
            raise ValueError(f"covar has {C.shape[0]} rows, scores has {n}")
        C, n_imp_c = _handle_missing(C, missing, "covar")
    P = C.shape[1]

    score_ids = (np.array([f"score_{j}" for j in range(K)], dtype=object)
                 if score_ids is None else np.asarray(score_ids, dtype=object))
    covar_ids = (np.array([f"covar_{j}" for j in range(P)], dtype=object)
                 if covar_ids is None else np.asarray(covar_ids, dtype=object))
    if score_ids.shape != (K,) or covar_ids.shape != (P,):
        raise ValueError("score_ids/covar_ids must match the column counts")

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

    X = np.hstack([S, C])
    pf = np.concatenate([pf_s, np.zeros(P)])

    alphas = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    if np.any(alphas < 0) or np.any(alphas > 1):
        raise ValueError("alpha must lie in [0, 1]")

    n_folds = int(n_folds)
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be in [2, n={n}], got {n_folds}")

    rng = np.random.default_rng(seed)
    parts = _folds(n, n_folds, rng, stratify=y if family == "binomial" else None)

    # One penalty grid for every fold, measured on the full training set at the
    # covariate-only fit. cv.glmnet does the same: fold-specific grids would
    # not be comparable, and averaging coefficients across them would be
    # averaging across different problems.
    center, scale, dead = _standardize(X, np.arange(n))
    Xs_full = (X - center) / scale
    lambdas = _lambda_grid_for(Xs_full, y, pf, family, alphas.min(), n_lambda,
                               lambda_min_ratio, K)

    fits = []
    results = []
    beta_sum = np.zeros(K + P)
    intercept_sum = 0.0
    for k, val_idx in enumerate(parts):
        tr_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
        if tr_idx.size == 0 or val_idx.size == 0:
            continue
        best = _fit_one_fold(X, y, tr_idx, val_idx, pf, alphas, lambdas,
                             family, n_abort, dfmax, tol, max_iter)
        used = best["loss"] < best["null_loss"]
        fits.append(FoldFit(fold=k, alpha=best["alpha"], lam=best["lam"],
                            lam_index=best["lam_index"], loss=best["loss"],
                            null_loss=best["null_loss"],
                            n_nonzero=int(np.count_nonzero(best["beta"][:K])),
                            used=bool(used)))
        results.append(best)
        if used:
            beta_sum += best["beta"]
            intercept_sum += best["intercept"]

    n_used = sum(f.used for f in fits)
    cv_loss, cv_null_loss = _honest_cv_loss(results, n)

    # Two gates, and the pooled one decides. A fold selects on a few dozen
    # individuals, so under pure noise roughly half of them beat their own
    # null by chance, and the per-fold gate alone lets a dense model of
    # nothing through. The pooled statistic uses every individual once, at a
    # penalty chosen by the *other* folds, so nothing in it was tuned on the
    # data it scores.
    if n_used and cv_loss < cv_null_loss:
        beta_raw = beta_sum / n_used
        intercept = intercept_sum / n_used
    else:
        beta_raw = np.zeros(K + P)
        ybar = float(np.mean(y))
        if family == "gaussian":
            intercept = ybar
        else:
            ybar = min(max(ybar, 1e-6), 1 - 1e-6)
            intercept = float(np.log(ybar / (1 - ybar)))

    log = {
        "n": int(n), "n_scores": int(K), "n_covar": int(P),
        "n_folds": len(fits), "n_folds_used": int(n_used),
        "family": family, "alphas": alphas.tolist(),
        "n_lambda": int(lambdas.size),
        "lambda_max": float(lambdas[0]), "lambda_min": float(lambdas[-1]),
        "dropped_constant": int(dead[:K].sum()),
        "imputed_missing": int(n_imp + n_imp_c),
        "unpenalized_scores": int(np.count_nonzero(pf_s == 0)),
        "cv_loss": float(cv_loss) if np.isfinite(cv_loss) else None,
        "cv_null_loss": (float(cv_null_loss) if np.isfinite(cv_null_loss)
                         else None),
    }
    if family == "gaussian" and np.isfinite(cv_loss):
        # Incremental over the covariate-only model, not the full model's R².
        # The scores are what is being assessed; with covariates in the fit,
        # 1 - cv_loss/var(y) would credit them with age and sex as well, and
        # would not be comparable to the held-out number
        # :func:`multipgs.incremental_r2` reports for the same fit.
        var_y = float(np.mean((y - y.mean()) ** 2))
        log["cv_r2"] = (float((cv_null_loss - cv_loss) / var_y)
                        if var_y > 0 else 0.0)
    if np.all(beta_raw == 0):
        log["null_model"] = (
            "the cross-validated fit did not beat its covariate-only model; "
            "coefficients are all zero")

    return MultiPGSFit(
        beta=beta_raw[:K], intercept=float(intercept),
        covar_beta=beta_raw[K:], beta_std=beta_raw[:K] * scale[:K],
        center=center, scale=scale, score_ids=score_ids, covar_ids=covar_ids,
        family=family, folds=fits, log=log)


def _honest_cv_loss(results, n):
    """Pooled held-out loss at a penalty each fold did not choose for itself.

    Evaluating a fold at the ``(alpha, lambda)`` *it* selected on *its own*
    validation individuals is selection on the assessment set, and reports a
    positive R² for pure noise. Taking the penalty from the other folds
    removes that: for fold ``k`` the operating point is the median lambda index
    and modal alpha index chosen by folds ``j != k``, none of which saw fold
    ``k``'s individuals at any stage. Both losses are means over individuals,
    so pooling is the size-weighted mean of the per-fold losses.

    Slightly conservative as an estimate of the returned model, which averages
    the folds and is normally a little better than any one of them. Returns
    ``(inf, inf)`` when there are too few folds to hold one out.
    """
    if len(results) < 2:
        return np.inf, np.inf
    a_sel = np.array([r["alpha_index"] for r in results])
    l_sel = np.array([r["lam_index"] for r in results])
    sizes = np.array([r["n_val"] for r in results], dtype=float)

    total = 0.0
    total_null = 0.0
    for k, r in enumerate(results):
        others = np.ones(len(results), dtype=bool)
        others[k] = False
        counts = np.bincount(a_sel[others])
        a_idx = int(np.argmax(counts))
        l_idx = int(np.median(l_sel[others]))
        table = r["loss_table"]
        row = table[a_idx]
        if not np.isfinite(row[l_idx]):
            # Early stopping never reached that penalty for this alpha; take
            # the nearest one that was evaluated.
            evaluated = np.flatnonzero(np.isfinite(row))
            if evaluated.size == 0:
                return np.inf, np.inf
            l_idx = int(evaluated[np.argmin(np.abs(evaluated - l_idx))])
        total += sizes[k] * row[l_idx]
        total_null += sizes[k] * r["null_loss"]
    return total / n, total_null / n


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
        # The IRLS gradient at the covariate-only fit. Weights at the null
        # model are p(1-p) with p the fitted covariate-only probability; the
        # working gradient reduces to X^T (y - p) / n.
        p = _null_probabilities(Xs, y, pf)
        grad = Xs.T @ (y - p) / n
    return _coord.lambda_grid(grad, pf, alpha, n_lambda=n_lambda,
                              lambda_min_ratio=ratio, n=n, n_penalized=K)


def _null_probabilities(Xs, y, pf, *, max_iter=50, tol=1e-9):
    """Fitted probabilities of the intercept + unpenalized-columns model."""
    n = Xs.shape[0]
    free = np.flatnonzero(pf <= 0.0)
    D = np.hstack([np.ones((n, 1)), Xs[:, free]])
    b = np.zeros(D.shape[1])
    b[0] = np.log(max(min(y.mean(), 1 - 1e-6), 1e-6)
                  / (1 - max(min(y.mean(), 1 - 1e-6), 1e-6)))
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(D @ b)))
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
    p = 1.0 / (1.0 + np.exp(-(D @ b)))
    return np.clip(p, 1e-9, 1 - 1e-9)


def _fit_one_fold(X, y, tr, val, pf, alphas, lambdas, family, n_abort, dfmax,
                  tol, max_iter):
    """Sweep the (alpha, lambda) grid on ``tr``, select on ``val``.

    Returns the raw-scale coefficients and intercept at the selected point,
    plus the loss of the covariate-only model, so the caller can decide whether
    this fold earned its place in the average.
    """
    center, scale, _ = _standardize(X, tr)
    Xtr = np.ascontiguousarray((X[tr] - center) / scale)
    Xval = (X[val] - center) / scale
    ytr, yval = y[tr], y[val]
    ntr = tr.size
    gaussian = family == "gaussian"

    if gaussian:
        ybar = float(ytr.mean())
        G = Xtr.T @ Xtr / ntr
        G = (G + G.T) * 0.5
        r = Xtr.T @ (ytr - ybar) / ntr

    best = None
    null_loss = np.inf
    # Held-out loss at every (alpha, lambda) actually evaluated; inf elsewhere.
    # Two numbers per grid point, so this stays negligible whatever n is.
    loss_table = np.full((alphas.size, lambdas.size), np.inf)
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

            for i in range(n_ok):
                coef = coefs[i]
                b0 = ybar if gaussian else float(b0s[i])
                pred = b0 + Xval @ coef
                lo = _gaussian_loss(yval, pred) if gaussian \
                    else _binomial_loss(yval, pred)
                loss_table[a_idx, start + i] = lo
                if start + i == 0:
                    null_loss = lo
                if best is None or lo < best["loss"] - 1e-15:
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
            "lam_index": best["lam_index"], "loss_table": loss_table,
            "n_val": int(val.size)}
