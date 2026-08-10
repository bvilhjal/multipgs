"""Elastic-net coordinate descent — the numerical core of the multi-PGS stack.

Two solvers, one update rule. Both minimise

.. math::

    \\tfrac{1}{2n}\\, \\ell(y, \\beta_0 + X\\beta)
      + \\lambda \\sum_j w_j \\left[ \\alpha |\\beta_j|
        + \\tfrac{1-\\alpha}{2} \\beta_j^2 \\right]

over a decreasing sequence of :math:`\\lambda`, with warm starts, where
:math:`w_j` is a per-column *penalty factor*. Columns with ``w_j == 0`` are
never shrunk; that is how covariates (age, sex, principal components) ride
along inside the same fit as the polygenic scores instead of being partialled
out beforehand.

**Gaussian** (:func:`enet_path_gaussian`) uses *covariance updates*: the solver
touches only the :math:`K \\times K` Gram matrix :math:`G = X^\\top X / n` and
:math:`r = X^\\top y / n`, so one sweep costs :math:`O(K^2)` no matter how many
individuals there are. That is the right trade for multi-PGS, where a few
hundred to a few thousand scores are stacked in tens of thousands of people:
the Gram is formed once by BLAS and the path is then independent of ``n``.

**Binomial** (:func:`enet_path_binomial`) cannot reuse a fixed Gram, because the
IRLS weights change every outer iteration. It runs the textbook nested loop —
quadratic approximation outside, weighted coordinate descent inside — with
*naive updates* against a column-major ``X``, at :math:`O(nK)` per sweep.

Kernels are written twice in the style ldpred3 uses: an explicit-loop version
compiled by Numba, and a NumPy fallback that vectorises the same arithmetic in
the same order, so the two agree to floating-point round-off. The dispatch
happens once, at import.

Nothing in this module is public API; see :mod:`multipgs.stack`.
"""

from __future__ import annotations

import numpy as np

from ._ldpred3_compat import HAVE_NUMBA, _jit_nogil


__all__ = ["enet_path_gaussian", "enet_path_binomial", "lambda_grid",
           "unpenalized_fit"]

# Element budget for the row-chunked weighted column-sum-of-squares. Bounds the
# temporary at roughly 16 MB of float64 whatever n and K are.
_CHUNK_ELEMS = 2_000_000


# ---------------------------------------------------------------------------
# Coordinate sweeps
# ---------------------------------------------------------------------------

def _sweep_gram_py(G, beta, grad, pf, l1, l2, idx, nidx):
    """One coordinate sweep; returns the largest scaled squared step.

    ``grad`` holds :math:`X^\\top (y - X\\beta) / n` and is kept current by the
    rank-one correction after each accepted step. ``G`` is symmetric, so the
    contiguous row ``G[j]`` stands in for the strided column.
    """
    maxd = 0.0
    for t in range(nidx):
        j = idx[t]
        gjj = G[j, j]
        if gjj <= 0.0:
            continue
        bj = beta[j]
        z = grad[j] + gjj * bj
        p = pf[j]
        if p <= 0.0:
            new = z / gjj
        else:
            thr = l1 * p
            if z > thr:
                new = (z - thr) / (gjj + l2 * p)
            elif z < -thr:
                new = (z + thr) / (gjj + l2 * p)
            else:
                new = 0.0
        d = new - bj
        if d != 0.0:
            beta[j] = new
            grad -= G[j] * d
            dd = gjj * d * d
            if dd > maxd:
                maxd = dd
    return maxd


def _sweep_gram_nb(G, beta, grad, pf, l1, l2, idx, nidx):
    K = G.shape[0]
    maxd = 0.0
    for t in range(nidx):
        j = idx[t]
        gjj = G[j, j]
        if gjj <= 0.0:
            continue
        bj = beta[j]
        z = grad[j] + gjj * bj
        p = pf[j]
        if p <= 0.0:
            new = z / gjj
        else:
            thr = l1 * p
            if z > thr:
                new = (z - thr) / (gjj + l2 * p)
            elif z < -thr:
                new = (z + thr) / (gjj + l2 * p)
            else:
                new = 0.0
        d = new - bj
        if d != 0.0:
            beta[j] = new
            for k in range(K):
                grad[k] -= G[j, k] * d
            dd = gjj * d * d
            if dd > maxd:
                maxd = dd
    return maxd


def _sweep_wls_py(X, w, res, beta, xtwx, pf, l1, l2, idx, nidx, n):
    maxd = 0.0
    for t in range(nidx):
        j = idx[t]
        d0 = xtwx[j]
        if d0 <= 0.0:
            continue
        xj = X[:, j]
        bj = beta[j]
        z = float(np.dot(w * xj, res)) / n + d0 * bj
        p = pf[j]
        if p <= 0.0:
            new = z / d0
        else:
            thr = l1 * p
            if z > thr:
                new = (z - thr) / (d0 + l2 * p)
            elif z < -thr:
                new = (z + thr) / (d0 + l2 * p)
            else:
                new = 0.0
        d = new - bj
        if d != 0.0:
            beta[j] = new
            res -= xj * d
            dd = d0 * d * d
            if dd > maxd:
                maxd = dd
    return maxd


def _sweep_wls_nb(X, w, res, beta, xtwx, pf, l1, l2, idx, nidx, n):
    maxd = 0.0
    for t in range(nidx):
        j = idx[t]
        d0 = xtwx[j]
        if d0 <= 0.0:
            continue
        bj = beta[j]
        s = 0.0
        for i in range(n):
            s += w[i] * X[i, j] * res[i]
        z = s / n + d0 * bj
        p = pf[j]
        if p <= 0.0:
            new = z / d0
        else:
            thr = l1 * p
            if z > thr:
                new = (z - thr) / (d0 + l2 * p)
            elif z < -thr:
                new = (z + thr) / (d0 + l2 * p)
            else:
                new = 0.0
        d = new - bj
        if d != 0.0:
            beta[j] = new
            for i in range(n):
                res[i] -= X[i, j] * d
            dd = d0 * d * d
            if dd > maxd:
                maxd = dd
    return maxd


if HAVE_NUMBA:  # pragma: no cover - selected at import; both paths are tested
    _sweep_gram = _jit_nogil(_sweep_gram_nb)
    _sweep_wls = _jit_nogil(_sweep_wls_nb)
else:
    _sweep_gram = _sweep_gram_py
    _sweep_wls = _sweep_wls_py


# ---------------------------------------------------------------------------
# Gaussian path
# ---------------------------------------------------------------------------

def _path_gram(G, pf, beta, grad, lambdas, alpha, tol, max_iter, dfmax, out):
    """Warm-started path. Returns how many ``lambdas`` were actually fitted."""
    K = G.shape[0]
    allidx = np.arange(K)
    act = np.empty(K, dtype=np.int64)
    nlam = lambdas.shape[0]
    fitted = nlam
    for li in range(nlam):
        l1 = lambdas[li] * alpha
        l2 = lambdas[li] * (1.0 - alpha)
        for _outer in range(max_iter):
            if _sweep_gram(G, beta, grad, pf, l1, l2, allidx, K) < tol:
                break
            n_act = 0
            for j in range(K):
                if beta[j] != 0.0 or pf[j] <= 0.0:
                    act[n_act] = j
                    n_act += 1
            for _inner in range(max_iter):
                if _sweep_gram(G, beta, grad, pf, l1, l2, act, n_act) < tol:
                    break
        nnz = 0
        for j in range(K):
            out[li, j] = beta[j]
            if beta[j] != 0.0 and pf[j] > 0.0:
                nnz += 1
        if nnz > dfmax:
            # Past here the path is denser than the caller allows. Report the
            # truncation rather than return a full-length path that was not
            # fitted.
            fitted = li + 1
            break
    return fitted


_path_gram_jit = _jit_nogil(_path_gram) if HAVE_NUMBA else _path_gram


def unpenalized_fit(G, r, pf, *, ridge=1e-10):
    """Least-squares fit restricted to the unpenalized columns.

    This is the model at :math:`\\lambda_{\\max}`, and the point at which
    :func:`lambda_grid` measures the gradient, so the grid starts where the
    first *score* enters rather than where the first covariate would. Returns
    ``(beta, grad)`` with ``beta`` zero outside the unpenalized set.
    """
    G = np.asarray(G, dtype=float)
    r = np.asarray(r, dtype=float)
    beta = np.zeros(G.shape[0])
    free = np.flatnonzero(np.asarray(pf, dtype=float) <= 0.0)
    if free.size:
        A = G[np.ix_(free, free)].copy()
        A[np.diag_indices_from(A)] += ridge
        try:
            beta[free] = np.linalg.solve(A, r[free])
        except np.linalg.LinAlgError:
            # Collinear covariates (a dummy-coded set carrying no reference
            # level, say). The minimum-norm solution keeps the fit going and
            # leaves the fitted values — all the score coefficients ever see —
            # unchanged.
            beta[free] = np.linalg.lstsq(A, r[free], rcond=None)[0]
    return beta, r - G @ beta


def lambda_grid(grad, pf, alpha, *, n_lambda=100, lambda_min_ratio=None,
                n=None, n_penalized=None):
    """Geometric grid from the largest useful ``lambda`` down.

    ``grad`` must be the gradient at the unpenalized-only fit (see
    :func:`unpenalized_fit`). For ridge (``alpha == 0``) no finite
    :math:`\\lambda` zeroes the solution, so the starting point is taken at
    ``alpha = 1e-3``, exactly as glmnet does.
    """
    pf = np.asarray(pf, dtype=float)
    a = max(float(alpha), 1e-3)
    pen = pf > 0.0
    if not np.any(pen):
        raise ValueError("every column is unpenalized; nothing to select over")
    lmax = float(np.max(np.abs(np.asarray(grad, dtype=float)[pen])
                        / (a * pf[pen])))
    if not np.isfinite(lmax) or lmax <= 0.0:
        lmax = 1e-3
    if lambda_min_ratio is None:
        if n is None or n_penalized is None:
            lambda_min_ratio = 1e-3
        else:
            lambda_min_ratio = 1e-4 if n > n_penalized else 1e-2
    n_lambda = int(n_lambda)
    if n_lambda < 2:
        return np.array([lmax], dtype=float)
    return np.geomspace(lmax, lmax * float(lambda_min_ratio), n_lambda)


def enet_path_gaussian(G, r, *, pf, alpha, lambdas, beta_init=None,
                       grad_init=None, tol=1e-7, max_iter=1000, dfmax=None):
    """Elastic-net path for a squared-error loss, via covariance updates.

    ``G`` is :math:`X^\\top X / n` (symmetric; unit diagonal for standardized
    columns) and ``r`` is :math:`X^\\top y / n` for a centred ``y``. Returns
    ``(coefs, n_fitted)``; rows of ``coefs`` at or past ``n_fitted`` were not
    fitted (``dfmax`` truncation).
    """
    G = np.ascontiguousarray(G, dtype=np.float64)
    r = np.ascontiguousarray(r, dtype=np.float64)
    pf = np.ascontiguousarray(pf, dtype=np.float64)
    lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)
    K = G.shape[0]
    if G.shape != (K, K) or r.shape != (K,) or pf.shape != (K,):
        raise ValueError("G, r and pf must be (K, K), (K,) and (K,)")
    if beta_init is None or grad_init is None:
        beta, grad = unpenalized_fit(G, r, pf)
    else:
        beta = np.array(beta_init, dtype=np.float64)
        grad = np.array(grad_init, dtype=np.float64)
    out = np.zeros((lambdas.shape[0], K), dtype=np.float64)
    dfmax = K if dfmax is None else int(dfmax)
    n_fitted = _path_gram_jit(G, pf, beta, grad, lambdas, float(alpha),
                              float(tol), int(max_iter), dfmax, out)
    return out, int(n_fitted)


# ---------------------------------------------------------------------------
# Binomial path
# ---------------------------------------------------------------------------

def _weighted_col_sumsq(X, w, n):
    """``sum_i w_i X_ij^2 / n`` without an ``n x K`` temporary."""
    K = X.shape[1]
    out = np.zeros(K)
    step = max(1, _CHUNK_ELEMS // max(K, 1))
    for s in range(0, n, step):
        Xs = np.asarray(X[s:s + step])
        out += (Xs * Xs).T @ w[s:s + step]
    return out / n


def enet_path_binomial(X, y, *, pf, alpha, lambdas, beta_init=None,
                       b0_init=None, tol=1e-7, max_iter=100, irls_max=25,
                       irls_tol=1e-7, dfmax=None, w_min=1e-5):
    """Elastic-net path for a binomial log-likelihood (IRLS + naive updates).

    ``X`` should already be standardized and ``y`` coded 0/1. ``beta_init`` and
    ``b0_init`` warm-start the first penalty, which is what lets a caller walk
    a long grid in blocks without paying to re-descend it. Returns
    ``(intercepts, coefs, n_fitted)``.
    """
    X = np.asfortranarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    pf = np.ascontiguousarray(pf, dtype=np.float64)
    lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)
    n, K = X.shape
    if y.shape != (n,) or pf.shape != (K,):
        raise ValueError("y must be (n,) and pf must be (K,)")
    dfmax = K if dfmax is None else int(dfmax)

    ybar = min(max(float(np.mean(y)), 1e-6), 1.0 - 1e-6)
    b0 = float(np.log(ybar / (1.0 - ybar))) if b0_init is None \
        else float(b0_init)
    beta = np.zeros(K) if beta_init is None \
        else np.array(beta_init, dtype=np.float64)
    if beta.shape != (K,):
        raise ValueError(f"beta_init must have length {K}")
    coefs = np.zeros((lambdas.shape[0], K))
    b0s = np.full(lambdas.shape[0], b0)

    allidx = np.arange(K, dtype=np.int64)
    act = np.empty(K, dtype=np.int64)
    n_fitted = lambdas.shape[0]

    for li in range(lambdas.shape[0]):
        l1 = lambdas[li] * alpha
        l2 = lambdas[li] * (1.0 - alpha)
        eta = b0 + X @ beta
        for _irls in range(irls_max):
            p = 1.0 / (1.0 + np.exp(-eta))
            np.clip(p, 1e-9, 1.0 - 1e-9, out=p)
            w = np.maximum(p * (1.0 - p), w_min)
            # Working residual z - eta with z = eta + (y - p) / w, so the
            # working response never has to be formed.
            res = (y - p) / w
            xtwx = _weighted_col_sumsq(X, w, n)
            wsum = float(np.sum(w))
            eta_old = eta.copy()
            for _outer in range(max_iter):
                shift = float(np.dot(w, res)) / wsum
                b0 += shift
                res -= shift
                m = _sweep_wls(X, w, res, beta, xtwx, pf, l1, l2,
                               allidx, K, n)
                if max(m, (wsum / n) * shift * shift) < tol:
                    break
                n_act = 0
                for j in range(K):
                    if beta[j] != 0.0 or pf[j] <= 0.0:
                        act[n_act] = j
                        n_act += 1
                for _inner in range(max_iter):
                    shift = float(np.dot(w, res)) / wsum
                    b0 += shift
                    res -= shift
                    m = _sweep_wls(X, w, res, beta, xtwx, pf, l1, l2,
                                   act, n_act, n)
                    if max(m, (wsum / n) * shift * shift) < tol:
                        break
            eta = b0 + X @ beta
            if float(np.max(np.abs(eta - eta_old))) < irls_tol:
                break
        b0s[li] = b0
        coefs[li] = beta
        if int(np.count_nonzero(beta[pf > 0.0])) > dfmax:
            n_fitted = li + 1
            break
    return b0s, coefs, int(n_fitted)
