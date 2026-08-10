"""The coordinate-descent core, against an independent implementation."""

import numpy as np
import pytest

from multipgs import _coord


def _problem(n=300, K=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, K))
    X = (X - X.mean(0)) / X.std(0)
    b = np.zeros(K)
    b[:5] = rng.normal(size=5)
    y = X @ b + rng.normal(size=n) * 0.8
    y = y - y.mean()
    G = X.T @ X / n
    G = (G + G.T) * 0.5
    r = X.T @ y / n
    return X, y, G, r


@pytest.mark.parametrize("lam,alpha", [(0.05, 1.0), (0.02, 0.5), (0.1, 0.2)])
def test_gaussian_matches_sklearn(lam, alpha):
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y, G, r = _problem()
    pf = np.ones(G.shape[0])
    coefs, _ = _coord.enet_path_gaussian(G, r, pf=pf, alpha=alpha,
                                         lambdas=np.array([lam]), tol=1e-14,
                                         max_iter=5000)
    ref = sklearn.ElasticNet(alpha=lam, l1_ratio=alpha, fit_intercept=False,
                             tol=1e-14, max_iter=500_000).fit(X, y)
    assert np.allclose(coefs[0], ref.coef_, atol=1e-6)
    assert np.count_nonzero(coefs[0]) == np.count_nonzero(ref.coef_)


def test_unpenalized_columns_equal_frisch_waugh_lovell():
    """pf == 0 columns must behave exactly as if they were partialled out."""
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y, G, r = _problem()
    pf = np.ones(G.shape[0])
    pf[:3] = 0.0
    coefs, _ = _coord.enet_path_gaussian(G, r, pf=pf, alpha=1.0,
                                         lambdas=np.array([0.05]), tol=1e-14,
                                         max_iter=5000)
    Q, _ = np.linalg.qr(X[:, :3])
    yr = y - Q @ (Q.T @ y)
    Xr = X[:, 3:] - Q @ (Q.T @ X[:, 3:])
    ref = sklearn.Lasso(alpha=0.05, fit_intercept=False, tol=1e-14,
                        max_iter=500_000).fit(Xr, yr)
    assert np.allclose(coefs[0][3:], ref.coef_, atol=1e-7)


def test_numba_and_numpy_kernels_agree():
    """The compiled sweep and the NumPy fallback must give the same answer."""
    X, y, G, r = _problem(seed=7)
    pf = np.ones(G.shape[0])
    lambdas = np.geomspace(0.3, 0.01, 25)

    def run(sweep):
        beta, grad = _coord.unpenalized_fit(G, r, pf)
        out = np.zeros((lambdas.size, G.shape[0]))
        saved = _coord._sweep_gram
        try:
            _coord._sweep_gram = sweep
            _coord._path_gram(G, pf, beta, grad, lambdas, 1.0, 1e-12, 1000,
                              G.shape[0], out)
        finally:
            _coord._sweep_gram = saved
        return out

    a = run(_coord._sweep_gram_py)
    b = run(_coord._sweep_gram_nb)
    assert np.allclose(a, b, atol=1e-10)


def test_lambda_max_zeroes_every_penalized_coefficient():
    _, _, G, r = _problem()
    pf = np.ones(G.shape[0])
    _, grad = _coord.unpenalized_fit(G, r, pf)
    lambdas = _coord.lambda_grid(grad, pf, 1.0, n_lambda=5)
    coefs, _ = _coord.enet_path_gaussian(G, r, pf=pf, alpha=1.0,
                                         lambdas=lambdas)
    assert np.count_nonzero(coefs[0]) == 0
    assert np.count_nonzero(coefs[-1]) > 0
    assert lambdas[0] > lambdas[-1]


def test_lambda_grid_rejects_all_unpenalized():
    with pytest.raises(ValueError, match="nothing to select over"):
        _coord.lambda_grid(np.ones(4), np.zeros(4), 1.0)


def test_dfmax_truncates_the_path():
    _, _, G, r = _problem()
    pf = np.ones(G.shape[0])
    _, grad = _coord.unpenalized_fit(G, r, pf)
    lambdas = _coord.lambda_grid(grad, pf, 1.0, n_lambda=60)
    coefs, n_fitted = _coord.enet_path_gaussian(G, r, pf=pf, alpha=1.0,
                                                lambdas=lambdas, dfmax=4)
    assert n_fitted < lambdas.size
    assert np.count_nonzero(coefs[n_fitted - 1]) > 4


def test_binomial_path_reduces_deviance_monotonically():
    rng = np.random.default_rng(3)
    n, K = 400, 20
    X = rng.normal(size=(n, K))
    X = (X - X.mean(0)) / X.std(0)
    b = np.zeros(K)
    b[:4] = rng.normal(size=4) * 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ b)))).astype(float)
    pf = np.ones(K)
    # Start the grid at the true lambda_max: the gradient of the binomial
    # log-likelihood at the intercept-only fit.
    grad = X.T @ (y - y.mean()) / n
    lambdas = _coord.lambda_grid(grad, pf, 1.0, n_lambda=20,
                                 lambda_min_ratio=5e-3)
    b0s, coefs, n_fitted = _coord.enet_path_binomial(X, y, pf=pf, alpha=1.0,
                                                     lambdas=lambdas)
    assert n_fitted == lambdas.size
    dev = [2 * np.mean(np.logaddexp(0.0, b0s[i] + X @ coefs[i])
                       - y * (b0s[i] + X @ coefs[i]))
           for i in range(n_fitted)]
    # Training deviance is non-increasing as the penalty relaxes.
    assert np.all(np.diff(dev) < 1e-8)
    # And nothing enters at lambda_max.
    assert np.count_nonzero(coefs[0]) == 0


def test_binomial_warm_start_matches_a_single_pass():
    """Fitting the grid in two blocks must equal fitting it in one."""
    rng = np.random.default_rng(11)
    n, K = 250, 12
    X = rng.normal(size=(n, K))
    X = (X - X.mean(0)) / X.std(0)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] * 1.2)))).astype(float)
    pf = np.ones(K)
    lambdas = np.geomspace(0.2, 0.005, 12)
    b0_all, c_all, _ = _coord.enet_path_binomial(X, y, pf=pf, alpha=1.0,
                                                 lambdas=lambdas, tol=1e-11)
    b0_a, c_a, _ = _coord.enet_path_binomial(X, y, pf=pf, alpha=1.0,
                                             lambdas=lambdas[:6], tol=1e-11)
    b0_b, c_b, _ = _coord.enet_path_binomial(
        X, y, pf=pf, alpha=1.0, lambdas=lambdas[6:], beta_init=c_a[-1],
        b0_init=b0_a[-1], tol=1e-11)
    assert np.allclose(c_all[6:], c_b, atol=1e-5)
    assert np.allclose(b0_all[6:], b0_b, atol=1e-5)


def test_shape_validation():
    G = np.eye(3)
    with pytest.raises(ValueError, match=r"G, r and pf"):
        _coord.enet_path_gaussian(G, np.ones(2), pf=np.ones(3), alpha=1.0,
                                  lambdas=np.array([0.1]))
    with pytest.raises(ValueError, match="y must be"):
        _coord.enet_path_binomial(np.zeros((5, 3)), np.zeros(4), pf=np.ones(3),
                                  alpha=1.0, lambdas=np.array([0.1]))
