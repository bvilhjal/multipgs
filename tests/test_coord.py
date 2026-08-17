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
        converged = np.zeros(lambdas.size, dtype=bool)
        sweeps = np.zeros(lambdas.size, dtype=np.int64)
        saved = _coord._sweep_gram
        try:
            _coord._sweep_gram = sweep
            _coord._path_gram(G, pf, beta, grad, lambdas, 1.0, 1e-12, 1000,
                              G.shape[0], out, converged, sweeps)
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


@pytest.mark.parametrize(
    "kwargs, message",
    [({"n_lambda": 0}, "positive integer"),
     ({"n_lambda": 2.5}, "positive integer"),
     ({"lambda_min_ratio": 0.0}, r"\(0, 1\]"),
     ({"lambda_min_ratio": 1.1}, r"\(0, 1\]"),
     ({"alpha": np.nan}, r"\[0, 1\]")])
def test_lambda_grid_validates_numerical_controls(kwargs, message):
    alpha = kwargs.pop("alpha", 1.0)
    with pytest.raises(ValueError, match=message):
        _coord.lambda_grid(np.ones(4), np.ones(4), alpha, **kwargs)


def test_dfmax_truncates_the_path():
    _, _, G, r = _problem()
    pf = np.ones(G.shape[0])
    _, grad = _coord.unpenalized_fit(G, r, pf)
    lambdas = _coord.lambda_grid(grad, pf, 1.0, n_lambda=60)
    coefs, n_fitted = _coord.enet_path_gaussian(G, r, pf=pf, alpha=1.0,
                                                lambdas=lambdas, dfmax=4)
    assert n_fitted < lambdas.size
    assert np.count_nonzero(coefs[n_fitted - 1]) > 4


def test_dfmax_is_a_path_stop_not_a_hard_sparsity_cap():
    """The first row crossing dfmax is retained for stable path semantics."""
    _, _, G, r = _problem()
    pf = np.ones(G.shape[0])
    _, grad = _coord.unpenalized_fit(G, r, pf)
    lambdas = _coord.lambda_grid(grad, pf, 1.0, n_lambda=60)
    coefs, n_fitted = _coord.enet_path_gaussian(
        G, r, pf=pf, alpha=1.0, lambdas=lambdas, dfmax=0)

    assert n_fitted >= 2
    assert np.count_nonzero(coefs[n_fitted - 1]) > 0
    assert np.all(coefs[n_fitted:] == 0.0)


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


def test_gaussian_batch_matches_sequential_paths():
    """PUMAS refits share a Gram; the batch must match one path per right-hand side."""
    _, _, G, r = _problem(n=200, K=12, seed=21)
    rng = np.random.default_rng(21)
    R = r + 0.05 * rng.normal(size=(6, r.size))
    pf = np.ones(G.shape[0])
    lambdas = np.geomspace(0.2, 0.01, 15)
    batched, n_fitted = _coord.enet_path_gaussian_batch(
        G, R, pf=pf, alpha=1.0, lambdas=lambdas, tol=1e-12)
    for i, rhs in enumerate(R):
        coefs, n_one = _coord.enet_path_gaussian(
            G, rhs, pf=pf, alpha=1.0, lambdas=lambdas, tol=1e-12)
        assert int(n_fitted[i]) == n_one
        assert np.array_equal(batched[i], coefs)


def test_gaussian_max_iter_is_a_total_sweep_budget(monkeypatch):
    calls = 0

    def never_converges(*args):
        nonlocal calls
        calls += 1
        return 1.0

    monkeypatch.setattr(_coord, "_sweep_gram", never_converges)
    G = np.eye(2)
    out = np.zeros((1, 2))
    converged = np.zeros(1, dtype=bool)
    sweeps = np.zeros(1, dtype=np.int64)
    fitted, exhausted = _coord._path_gram(
        G, np.ones(2), np.zeros(2), np.ones(2), np.array([0.1]),
        1.0, 1e-12, 3, 2, out, converged, sweeps)

    assert fitted == 1 and exhausted == 1
    assert calls == 3
    assert sweeps[0] == 3
    assert not converged[0]


def test_shape_validation():
    G = np.eye(3)
    with pytest.raises(ValueError, match=r"G, r and pf"):
        _coord.enet_path_gaussian(G, np.ones(2), pf=np.ones(3), alpha=1.0,
                                  lambdas=np.array([0.1]))
    with pytest.raises(ValueError, match="y must be"):
        _coord.enet_path_binomial(np.zeros((5, 3)), np.zeros(4), pf=np.ones(3),
                                  alpha=1.0, lambdas=np.array([0.1]))


def test_gaussian_path_optionally_reports_iteration_exhaustion():
    G = np.array([[1.0, 0.8], [0.8, 1.0]])
    r = np.array([1.0, -0.5])
    result = _coord.enet_path_gaussian(
        G, r, pf=np.ones(2), alpha=1.0, lambdas=np.array([0.0]),
        tol=1e-30, max_iter=1, return_info=True)
    coefs, n_fitted, info = result
    assert coefs.shape == (1, 2) and n_fitted == 1
    assert info["converged"] is False
    assert info["n_iteration_exhausted"] == 1
    assert np.array_equal(info["converged_path"], [False])
    assert np.array_equal(info["n_sweeps_path"], [1])


def test_gaussian_batch_reports_per_repeat_iteration_exhaustion():
    G = np.array([[1.0, 0.8], [0.8, 1.0]])
    R = np.array([[1.0, -0.5], [0.5, -1.0]])
    coefs, n_fitted, info = _coord.enet_path_gaussian_batch(
        G, R, pf=np.ones(2), alpha=1.0, lambdas=np.array([0.0]),
        tol=1e-30, max_iter=1, return_info=True)

    assert coefs.shape == (2, 1, 2)
    assert np.array_equal(n_fitted, [1, 1])
    assert info["converged"] is False
    assert info["n_iteration_exhausted"] == 2
    assert np.array_equal(info["n_iteration_exhausted_by_repeat"], [1, 1])
    assert not np.any(info["converged_path"])


def test_binomial_max_iter_is_a_total_sweep_budget(monkeypatch):
    calls = 0

    def never_converges(*args):
        nonlocal calls
        calls += 1
        return 1.0

    monkeypatch.setattr(_coord, "_sweep_wls", never_converges)
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    _, _, n_fitted, info = _coord.enet_path_binomial(
        X, y, pf=np.ones(1), alpha=1.0, lambdas=np.array([0.0]),
        tol=1e-30, irls_tol=1e-30, max_iter=3, irls_max=1,
        return_info=True)

    assert n_fitted == 1
    assert calls == 3
    assert info["converged"] is False
    assert info["n_coordinate_descent_exhausted"] == 1


def test_binomial_path_optionally_reports_iteration_exhaustion():
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    result = _coord.enet_path_binomial(
        X, y, pf=np.ones(1), alpha=1.0, lambdas=np.array([0.0]),
        tol=1e-30, irls_tol=1e-30, max_iter=1, irls_max=1,
        return_info=True)
    _, _, n_fitted, info = result
    assert n_fitted == 1
    assert info["converged"] is False
    assert info["n_iteration_exhausted"] == 1
    assert info["n_irls_exhausted"] == 1
