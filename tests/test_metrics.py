"""Accuracy metrics, checked against closed forms rather than themselves."""

import numpy as np
import pytest

from multipgs import (auc, evaluate, incremental_r2, liability_r2,
                      nagelkerke_r2, r2)


def test_r2_is_the_squared_correlation_and_scale_free():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    pred = 0.6 * y + rng.normal(size=500)
    assert r2(y, pred) == pytest.approx(np.corrcoef(y, pred)[0, 1] ** 2)
    assert r2(y, 3.0 * pred + 7.0) == pytest.approx(r2(y, pred))
    assert r2(y, np.zeros(500)) == 0.0


def test_incremental_r2_is_the_gain_over_covariates_alone():
    rng = np.random.default_rng(1)
    n = 800
    c = rng.normal(size=(n, 2))
    g = rng.normal(size=n)
    y = 2.0 * c[:, 0] + 0.5 * g + rng.normal(size=n) * 0.5
    # A score orthogonal to the covariates adds about its own R2.
    delta = incremental_r2(y, g, c)
    assert 0.0 < delta < r2(y, g) + 0.05
    # A score that is a covariate adds nothing.
    assert incremental_r2(y, c[:, 0], c) == pytest.approx(0.0, abs=1e-9)
    # Without covariates it reduces to r2.
    assert incremental_r2(y, g) == pytest.approx(r2(y, g), abs=1e-9)


def test_auc_endpoints_and_ties():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(y, np.array([0.0, 1.0, 2.0, 3.0])) == 1.0
    assert auc(y, np.array([3.0, 2.0, 1.0, 0.0])) == 0.0
    assert auc(y, np.ones(4)) == 0.5           # no resolution at all
    # One tied case/control pair out of four contributes 0.5.
    assert auc(y, np.array([0.0, 1.0, 1.0, 2.0])) == pytest.approx(0.875)


def test_auc_needs_both_classes():
    with pytest.raises(ValueError, match="both cases and controls"):
        auc(np.zeros(5), np.arange(5.0))
    with pytest.raises(ValueError, match="0/1"):
        auc(np.array([0.0, 1.0, 2.0, 1.0]), np.arange(4.0))


def test_nagelkerke_is_zero_for_a_useless_score_and_high_for_a_perfect_one():
    rng = np.random.default_rng(2)
    y = (rng.random(600) < 0.3).astype(float)
    assert nagelkerke_r2(y, rng.normal(size=600)) < 0.02
    assert nagelkerke_r2(y, y + rng.normal(size=600) * 0.05) > 0.9


def test_liability_r2_matches_the_published_transformation():
    # Lee et al. 2012. With P == K the theta term vanishes and the factor is
    # [K(1-K)]^2 / (z^2 P(1-P)) -- check against that closed form.
    from statistics import NormalDist
    K = 0.1
    nd = NormalDist()
    t = nd.inv_cdf(1 - K)
    z = nd.pdf(t)
    c = (K * (1 - K)) ** 2 / (z * z * K * (1 - K))
    assert liability_r2(0.05, K, K) == pytest.approx(c * 0.05 / (1 + 0.0),
                                                     rel=1e-12)
    # Ascertainment (P > K) shrinks the observed-scale R2 more.
    ascertained = liability_r2(0.05, K, 0.5)
    assert ascertained != pytest.approx(c * 0.05)
    assert 0 < ascertained < 1


def test_liability_r2_recovers_a_simulated_liability_scale_r2():
    """End-to-end: simulate a threshold trait with known liability R2."""
    rng = np.random.default_rng(3)
    n, K = 200_000, 0.15
    g = rng.normal(size=n)
    rho = 0.6                                  # corr(g, liability)
    liab = rho * g + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    y = (liab > np.quantile(liab, 1 - K)).astype(float)
    observed = r2(y, g)
    est = liability_r2(observed, K, float(y.mean()))
    assert est == pytest.approx(rho ** 2, abs=0.02)


def test_liability_r2_validates_its_inputs():
    with pytest.raises(ValueError, match="prevalence"):
        liability_r2(0.1, 0.0, 0.5)
    with pytest.raises(ValueError, match="prop_cases"):
        liability_r2(0.1, 0.1, 1.5)
    with pytest.raises(ValueError, match="finite"):
        liability_r2(np.nan, 0.1, 0.5)
    with pytest.raises(ValueError, match="finite"):
        liability_r2([0.1, np.inf], 0.1, 0.5)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            liability_r2(bad, 0.1, 0.5)


def test_evaluate_reports_the_right_metrics_per_family():
    rng = np.random.default_rng(4)
    n = 1000
    g = rng.normal(size=n)
    c = rng.normal(size=(n, 2))
    y = g + c[:, 0] + rng.normal(size=n)
    res = evaluate(y, g, covar=c, n_boot=100, seed=0)
    assert set(res.metrics) == {"r2", "incremental_r2"}
    assert set(res.ci) == set(res.metrics)
    for name, (lo, hi) in res.ci.items():
        assert lo <= res.metrics[name] <= hi
    assert "n = 1000" in str(res)

    yb = (y > np.median(y)).astype(float)
    resb = evaluate(yb, g, family="binomial", prevalence=0.5, n_boot=50,
                    seed=0)
    assert set(resb.metrics) == {"r2", "auc", "nagelkerke_r2", "liability_r2"}
    assert resb.n_cases == int(yb.sum())
    assert "cases" in str(resb)


def test_evaluate_without_bootstrap_has_no_intervals():
    rng = np.random.default_rng(5)
    y = rng.normal(size=100)
    res = evaluate(y, rng.normal(size=100), n_boot=0)
    assert res.ci == {}
    assert res.n_boot_skipped == 0


def test_evaluate_counts_bootstrap_replicates_it_could_not_use(monkeypatch):
    """A resample whose metric raises must be counted, not silently dropped."""
    rng = np.random.default_rng(13)
    n = 200
    y = rng.normal(size=n)
    y[0] = -5.0                          # the full-data metric must not fail
    pred = 0.6 * y + rng.normal(size=n)
    real_r2 = r2

    def flaky(yy, pp):
        if yy[0] > 0.0:
            raise ValueError("this resample cannot be evaluated")
        return real_r2(yy, pp)

    monkeypatch.setattr("multipgs.metrics.r2", flaky)
    res = evaluate(y, pred, n_boot=100, seed=0)
    assert 0 < res.n_boot_skipped < 100
    assert "skipped" in str(res)
    again = evaluate(y, pred, n_boot=100, seed=0)
    assert again.n_boot_skipped == res.n_boot_skipped


def test_evaluate_validates_shapes_and_family():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="same length"):
        evaluate(rng.normal(size=10), rng.normal(size=9))
    with pytest.raises(ValueError, match="family"):
        evaluate(rng.normal(size=10), rng.normal(size=10), family="poisson")
    with pytest.raises(ValueError, match="at least 3"):
        evaluate(np.zeros(2), np.zeros(2))


@pytest.mark.parametrize("n_boot", [-1, 1.5, np.nan, True])
def test_evaluate_validates_bootstrap_count(n_boot):
    y = np.arange(5.0)
    with pytest.raises(ValueError, match="n_boot.*non-negative integer"):
        evaluate(y, y, n_boot=n_boot)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_evaluate_validates_interval_level(level):
    y = np.arange(5.0)
    with pytest.raises(ValueError, match=r"level.*\(0, 1\)"):
        evaluate(y, y, n_boot=0, level=level)


def test_public_metrics_reject_non_finite_observations_and_covariates():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    pred = np.arange(4.0)
    bad_pred = pred.copy()
    bad_pred[1] = np.nan
    for metric in (r2, incremental_r2, auc, nagelkerke_r2):
        with pytest.raises(ValueError, match="pred.*non-finite"):
            metric(y, bad_pred)

    bad_y = y.copy()
    bad_y[0] = np.inf
    with pytest.raises(ValueError, match="y.*non-finite"):
        r2(bad_y, pred)

    covar = np.ones((4, 2))
    covar[0, 0] = np.nan
    for metric in (incremental_r2, nagelkerke_r2):
        with pytest.raises(ValueError, match="covar.*non-finite"):
            metric(y, pred, covar)
    with pytest.raises(ValueError, match="covar.*non-finite"):
        evaluate(y, pred, covar=covar, n_boot=0)
    with pytest.raises(ValueError, match="pred.*non-finite"):
        evaluate(y, bad_pred, n_boot=0)
