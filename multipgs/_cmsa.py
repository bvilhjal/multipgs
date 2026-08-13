"""CMSA fold fitting and nested assessment."""

from __future__ import annotations

import numpy as np

from . import _coord
from ._stats import (
    _gaussian_stats,
    _gaussian_stats_at_origin,
    _gaussian_system,
    _replace_gaussian_y_moments,
    _standardize,
    _subtract_gaussian_stats,
)

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


def _empty_solver_info():
    return {"converged": True, "n_path_points_fitted": 0,
            "n_iteration_exhausted": 0,
            "n_coordinate_descent_exhausted": 0,
            "n_irls_exhausted": 0,
            "n_baseline_not_converged": 0}


def _merge_solver_info(target, source):
    """Accumulate additive solver diagnostics in place."""
    target["converged"] = (bool(target["converged"])
                           and bool(source.get("converged", True)))
    for key in ("n_path_points_fitted", "n_iteration_exhausted",
                "n_coordinate_descent_exhausted", "n_irls_exhausted",
                "n_baseline_not_converged"):
        target[key] += int(source.get(key, 0))


def _fit_unpenalized_baseline(X, y, idx, pf, family, *, gaussian_stats=None,
                              return_info=False):
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
        result = (beta, float(ybar - np.dot(beta, center)))
        if return_info:
            return result + ({"converged": True, "n_iter": 0,
                              "linear_solve_failed": False},)
        return result

    center, scale, _ = _standardize(X, idx)
    Xs = np.ascontiguousarray((X[idx] - center) / scale)
    b0, coef, info = _binomial_baseline(Xs, y[idx], pf, return_info=True)
    beta = coef / scale
    result = (beta, float(b0 - np.dot(beta, center)))
    return result + (info,) if return_info else result


def _nested_cv_assessment(X, y, pf, alphas, family, n_outer, n_inner,
                          n_lambda, lambda_min_ratio, n_abort, dfmax, tol,
                          max_iter, K, seed, *, gaussian_stats=None,
                          X_unimputed=None, return_info=False):
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
    solver_info = _empty_solver_info()

    def result(loss, null_loss, gain_se, inner):
        values = (loss, null_loss, gain_se, inner)
        return values + (solver_info,) if return_info else values

    for val in outer_parts:
        tr = _complement(n, val)
        if val.size == 0 or tr.size < 2:
            return result(np.inf, np.inf, np.inf, 0)
        Xk = X
        if X_unimputed is not None:
            Xk = _impute_from_training(X_unimputed, tr)
        # The parent Gram is a function of X only. Subtracting the held-out
        # rows reuses it; X'y is rebuilt on the training rows so a large
        # outer phenotype cannot cancel into r. Fold-local mean imputation
        # changes X, so that path still forms training statistics directly.
        if gaussian_stats is not None and X_unimputed is None:
            held_stats = _gaussian_stats_at_origin(
                Xk, y, val, reference=gaussian_stats)
            tr_stats = _subtract_gaussian_stats(gaussian_stats, held_stats)
            # Reuse the Gram; rebuild X'y on training rows so a large
            # outer phenotype cannot cancel into the path.
            _replace_gaussian_y_moments(tr_stats, Xk, y, tr)
        else:
            tr_stats = (_gaussian_stats(Xk, y, tr)
                        if family == "gaussian" else None)
        if tr_stats is not None:
            lambda_grids = np.vstack([
                _lambda_grid_for_gaussian_stats(
                    tr_stats, pf, a, n_lambda, lambda_min_ratio, K)
                for a in alphas])
        else:
            c, s, _ = _standardize(Xk, tr)
            Xs = (Xk[tr] - c) / s
            lambda_grids = np.vstack([
                _lambda_grid_for(
                    Xs, y[tr], pf, family, a, n_lambda,
                    lambda_min_ratio, K)
                for a in alphas])

        baseline_beta, baseline_intercept, baseline_info = \
            _fit_unpenalized_baseline(
                Xk, y, tr, pf, family, gaussian_stats=tr_stats,
                return_info=True)
        if not baseline_info["converged"]:
            solver_info["converged"] = False
            solver_info["n_baseline_not_converged"] += 1
        pred0 = baseline_intercept + Xk[val] @ baseline_beta
        null_loss = (_gaussian_loss(y[val], pred0) if family == "gaussian"
                     else _binomial_loss(y[val], pred0))

        ni = min(int(n_inner), tr.size)
        if ni < 2:
            return result(np.inf, np.inf, np.inf, 0)
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
                Xk, y, inner_tr, inner_val, pf, alphas, lambda_grids, family,
                n_abort, dfmax, tol, max_iter, gaussian_stats=train_stats)
            _merge_solver_info(
                solver_info, best.get("solver_info", _empty_solver_info()))
            beta_sum += best["beta"]
            intercept_sum += best["intercept"]
            fitted += 1
        if fitted < 2:
            return result(np.inf, np.inf, np.inf, 0)

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
    return result(total_loss / n, total_null / n, gain_se,
                  int(inner_used or 0))


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


def _binomial_baseline(Xs, y, pf, *, max_iter=50, tol=1e-9,
                       return_info=False):
    """Intercept and standardized coefficients for the unpenalized model."""
    n = Xs.shape[0]
    free = np.flatnonzero(pf <= 0.0)
    D = np.hstack([np.ones((n, 1)), Xs[:, free]])
    b = np.zeros(D.shape[1])
    ybar = min(max(float(y.mean()), 1e-6), 1 - 1e-6)
    b[0] = np.log(ybar / (1 - ybar))
    converged = False
    solve_failed = False
    n_iter = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1
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
            solve_failed = True
            break
        b += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    coef = np.zeros(Xs.shape[1])
    coef[free] = b[1:]
    result = (float(b[0]), coef)
    if not return_info:
        return result
    info = {"converged": converged, "n_iter": int(n_iter),
            "linear_solve_failed": solve_failed}
    return result + (info,)


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
    solver_info = _empty_solver_info()
    if gaussian:
        stats = (_gaussian_stats(X, y, tr) if gaussian_stats is None
                 else gaussian_stats)
        center, scale, _, G, r, ybar = _gaussian_system(stats)
        base_coef, _ = _coord.unpenalized_fit(G, r, pf)
        Xtr = ytr = None
        base_b0 = ybar
        baseline_info = {"converged": True}
    else:
        center, scale, _ = _standardize(X, tr)
        # The solver wants columns contiguous. Preparing that layout here once
        # avoids recopying the whole training matrix for every path block.
        Xtr = np.asfortranarray((X[tr] - center) / scale)
        ytr = y[tr]
        base_b0, base_coef, baseline_info = _binomial_baseline(
            Xtr, ytr, pf, return_info=True)
    if not baseline_info["converged"]:
        solver_info["converged"] = False
        solver_info["n_baseline_not_converged"] = 1
    Xval = (X[val] - center) / scale
    yval = y[val]

    pred0 = base_b0 + Xval @ base_coef
    null_loss = (_gaussian_loss(yval, pred0) if gaussian
                 else _binomial_loss(yval, pred0))
    best = {"loss": null_loss, "alpha": float(alphas[0]),
            "alpha_index": 0, "lam": float("inf"), "lam_index": -1,
            "coef": base_coef.copy(), "b0": float(base_b0)}
    for a_idx, a in enumerate(alphas):
        alpha_lambdas = (np.asarray(lambdas[a_idx], dtype=float)
                         if np.ndim(lambdas) == 2
                         else np.asarray(lambdas, dtype=float))
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
        while start < alpha_lambdas.size:
            lams = alpha_lambdas[start:start + block]
            if gaussian:
                coefs, nf, path_info = _coord.enet_path_gaussian(
                    G, r, pf=pf, alpha=a, lambdas=lams, beta_init=beta_w,
                    grad_init=grad_w, tol=tol, max_iter=max_iter, dfmax=dfmax,
                    return_info=True)
                b0s = None
            else:
                b0s, coefs, nf, path_info = _coord.enet_path_binomial(
                    Xtr, ytr, pf=pf, alpha=a, lambdas=lams, beta_init=beta_w,
                    b0_init=b0_w, tol=tol, max_iter=max_iter, dfmax=dfmax,
                    return_info=True)
            n_ok = min(nf, lams.size)
            path_info["n_path_points_fitted"] = n_ok
            _merge_solver_info(solver_info, path_info)

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
            "lam_index": best["lam_index"], "n_val": int(val.size),
            "solver_info": solver_info}

