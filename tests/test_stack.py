"""The CMSA multi-PGS estimator."""

import numpy as np
import pytest

from multipgs import multi_pgs_fit, r2, simulate_panel
from multipgs import stack as stack_mod


def test_beats_the_best_single_score():
    """The point of the method: the combination must beat any one input."""
    sim = simulate_panel(n=3000, n_scores=40, n_causal=5, h2=0.5, seed=1)
    tr, te = slice(0, 2000), slice(2000, None)
    fit = multi_pgs_fit(sim.scores[tr], sim.y[tr], n_folds=5, n_lambda=40,
                        seed=0, score_ids=sim.score_ids)
    combined = r2(sim.y[te], fit.multi_pgs(sim.scores[te]))
    best_single = max(r2(sim.y[te], sim.scores[te, k])
                      for k in range(sim.scores.shape[1]))
    oracle = r2(sim.y[te], sim.genetic_value[te])
    assert combined > best_single * 1.2
    assert combined <= oracle + 0.05
    assert fit.n_folds_used == 5


def test_selects_mostly_the_causal_scores():
    sim = simulate_panel(n=4000, n_scores=60, n_causal=4, h2=0.6, seed=2)
    fit = multi_pgs_fit(sim.scores, sim.y, n_folds=5, n_lambda=40, seed=0,
                        score_ids=sim.score_ids)
    top = [sid for sid, _, _ in fit.selected(top=4)]
    causal = set(sim.score_ids[:4])
    assert len(causal.intersection(top)) >= 3


def test_covariates_are_not_penalized_and_not_reported_as_scores():
    sim = simulate_panel(n=2000, n_scores=20, n_causal=3, h2=0.4, n_covar=3,
                         seed=4)
    # A covariate with a big effect that the scores cannot explain.
    y = sim.y + 4.0 * sim.covar[:, 0]
    fit = multi_pgs_fit(sim.scores, y, covar=sim.covar, n_folds=5,
                        n_lambda=30, seed=0)
    assert fit.covar_beta.shape == (3,)
    assert abs(fit.covar_beta[0]) > 1.0        # unpenalized, so it survives
    assert fit.beta.shape == (20,)
    # multi_pgs is the score alone; predict adds the covariates.
    assert not np.allclose(fit.multi_pgs(sim.scores),
                           fit.predict(sim.scores, sim.covar))


def test_failed_incremental_gate_preserves_the_covariate_baseline():
    """A null *increment* must not turn the full predictor into an intercept."""
    rng = np.random.default_rng(401)
    n = 300
    scores = rng.normal(size=(n, 8))
    covar = rng.normal(size=(n, 1))
    y = 4.0 * covar[:, 0]
    fit = multi_pgs_fit(scores, y, covar=covar, n_folds=4,
                        assessment_folds=4, n_lambda=20, seed=1)
    assert "null_model" in fit.log
    assert fit.n_folds_used == 0
    assert np.all(fit.beta == 0.0)
    assert fit.covar_beta[0] == pytest.approx(4.0, abs=1e-8)
    assert np.max(np.abs(fit.predict(scores, covar) - y)) < 2e-8


def test_failed_incremental_gate_preserves_an_unpenalized_score():
    """A forced target-trait score is the baseline, not an optional addition."""
    rng = np.random.default_rng(402)
    scores = rng.normal(size=(320, 7))
    y = 3.0 * scores[:, 0]
    fit = multi_pgs_fit(scores, y, unpenalized_scores=[0], n_folds=4,
                        assessment_folds=4, n_lambda=20, seed=2)
    assert "null_model" in fit.log
    assert fit.n_folds_used == 0
    assert fit.beta[0] == pytest.approx(3.0, abs=1e-8)
    assert np.all(fit.beta[1:] == 0.0)
    assert np.max(np.abs(fit.multi_pgs(scores) + fit.intercept - y)) < 2e-8


def test_predict_requires_the_covariates_it_was_fitted_with():
    sim = simulate_panel(n=500, n_scores=10, n_covar=2, seed=5)
    fit = multi_pgs_fit(sim.scores, sim.y, covar=sim.covar, n_folds=3,
                        n_lambda=15, seed=0)
    with pytest.raises(ValueError, match="covariate"):
        fit.predict(sim.scores)
    fit_nc = multi_pgs_fit(sim.scores, sim.y, n_folds=3, n_lambda=15, seed=0)
    with pytest.raises(ValueError, match="no covariates"):
        fit_nc.predict(sim.scores, sim.covar)


def test_binomial_recovers_signal():
    sim = simulate_panel(n=4000, n_scores=30, n_causal=4, h2=0.6,
                         family="binomial", prevalence=0.2, n_covar=0, seed=6)
    tr, te = slice(0, 3000), slice(3000, None)
    fit = multi_pgs_fit(sim.scores[tr], sim.y[tr], family="binomial",
                        n_folds=5, n_lambda=30, seed=0)
    from multipgs import auc
    assert auc(sim.y[te], fit.multi_pgs(sim.scores[te])) > 0.75
    p = fit.predict_proba(sim.scores[te])
    assert np.all((p > 0) & (p < 1))


def test_binomial_rejects_non_binary_y():
    sim = simulate_panel(n=300, n_scores=5, seed=7)
    with pytest.raises(ValueError, match="0/1"):
        multi_pgs_fit(sim.scores, sim.y, family="binomial", n_folds=3)


def test_pure_noise_usually_gives_a_null_model():
    """The gate must not dress up selection noise as a score."""
    n_null = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        fit = multi_pgs_fit(rng.normal(size=(300, 40)), rng.normal(size=300),
                            n_folds=5, n_lambda=30, seed=seed)
        n_null += fit.n_selected == 0
        if fit.n_selected == 0:
            assert np.all(fit.beta == 0)
            assert "null_model" in fit.log
            assert "null" in fit.summary()
    assert n_null >= 6


def test_cross_validated_r2_is_not_positive_on_pure_noise():
    """cv_r2 must nest tuning: choosing and scoring a penalty on the same
    fold reports a positive R2 for noise. It must not."""
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        fit = multi_pgs_fit(rng.normal(size=(400, 50)), rng.normal(size=400),
                            n_folds=5, n_lambda=30, seed=seed)
        assert fit.cv_r2 < 0.01


def test_nested_gate_rejects_fixed_phenotype_permutations():
    """Outer assessment should normally reject deliberately broken signal."""
    sim = simulate_panel(n=600, n_scores=20, n_causal=4, h2=0.5,
                         n_covar=0, seed=88)
    n_null = 0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        y = sim.y[rng.permutation(sim.y.size)]
        fit = multi_pgs_fit(sim.scores, y, n_folds=4, assessment_folds=4,
                            n_lambda=20, seed=seed)
        n_null += "null_model" in fit.log
        assert fit.log["cv_scheme"] == "nested_cmsa"
    assert n_null >= 3


def test_outer_assessment_outcomes_do_not_change_the_inner_fit(monkeypatch):
    """Changing outer-fold y may change its loss, never its grid or model."""
    rng = np.random.default_rng(409)
    X = rng.normal(size=(90, 4))
    y = rng.normal(size=90)
    seed = 7
    outer0 = stack_mod._folds(
        len(y), 3, np.random.default_rng(seed))[0]
    original = stack_mod._fit_one_fold
    seen = []

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        # The first two calls are the two inner folds for outer fold zero.
        seen.append((np.asarray(args[6]).copy(), result["beta"].copy(),
                     result["intercept"]))
        return result

    monkeypatch.setattr(stack_mod, "_fit_one_fold", recorded)

    def first_outer_models(outcome):
        seen.clear()
        stats = stack_mod._gaussian_stats(X, outcome)
        stack_mod._nested_cv_assessment(
            X, outcome, np.ones(4), np.array([1.0]), "gaussian",
            3, 2, 8, None, 3, None, 1e-8, 200, 4, seed,
            gaussian_stats=stats)
        return [(lam.copy(), beta.copy(), intercept)
                for lam, beta, intercept in seen[:2]]

    before = first_outer_models(y)
    changed = y.copy()
    changed[outer0] += np.linspace(100.0, 1000.0, outer0.size)
    after = first_outer_models(changed)
    for left, right in zip(before, after):
        assert np.array_equal(left[0], right[0])
        assert np.array_equal(left[1], right[1])
        assert left[2] == right[2]


def test_cross_validated_r2_tracks_held_out_accuracy():
    sim = simulate_panel(n=4000, n_scores=30, n_causal=4, h2=0.5, seed=20)
    tr, te = slice(0, 3000), slice(3000, None)
    fit = multi_pgs_fit(sim.scores[tr], sim.y[tr], n_folds=5, n_lambda=40,
                        seed=0)
    pred = fit.predict(sim.scores[te])
    held_out = 1.0 - (np.sum((sim.y[te] - pred) ** 2)
                      / np.sum((sim.y[te] - sim.y[te].mean()) ** 2))
    assert fit.cv_r2 == pytest.approx(held_out, abs=0.06)
    assert fit.log["cv_loss"] < fit.log["cv_null_loss"]


def test_cross_validated_r2_is_gain_over_covariate_baseline():
    """It must be nested predictive gain, not full-model or recalibrated R²."""
    sim = simulate_panel(n=4000, n_scores=25, n_causal=3, h2=0.5, n_covar=2,
                         seed=21)
    y = sim.y + 3.0 * sim.covar[:, 0]           # a covariate that dominates
    tr, te = slice(0, 3000), slice(3000, None)
    fit = multi_pgs_fit(sim.scores[tr], y[tr], covar=sim.covar[tr], n_folds=5,
                        n_lambda=40, seed=0)
    design_tr = np.column_stack([np.ones(3000), sim.covar[tr]])
    baseline_coef = np.linalg.lstsq(design_tr, y[tr], rcond=None)[0]
    baseline_pred = np.column_stack(
        [np.ones(1000), sim.covar[te]]) @ baseline_coef
    full_pred = fit.predict(sim.scores[te], sim.covar[te])
    held_out = (np.sum((y[te] - baseline_pred) ** 2)
                - np.sum((y[te] - full_pred) ** 2)) \
        / np.sum((y[te] - y[te].mean()) ** 2)
    assert fit.cv_r2 == pytest.approx(held_out, abs=0.06)
    # The full-model R2 would be far larger; cv_r2 must not be that.
    assert fit.cv_r2 < 0.5


def test_unpenalized_score_always_survives():
    sim = simulate_panel(n=1500, n_scores=25, n_causal=3, h2=0.4, seed=10)
    fit = multi_pgs_fit(sim.scores, sim.y, n_folds=4, n_lambda=30, seed=0,
                        score_ids=sim.score_ids,
                        unpenalized_scores=[sim.score_ids[20]])
    assert fit.beta[20] != 0.0
    assert fit.log["unpenalized_scores"] == 1


def test_penalty_factor_shape_and_sign_are_validated():
    sim = simulate_panel(n=300, n_scores=10, seed=11)
    with pytest.raises(ValueError, match="length 10"):
        multi_pgs_fit(sim.scores, sim.y, penalty_factor=np.ones(3), n_folds=3)
    with pytest.raises(ValueError, match="non-negative"):
        multi_pgs_fit(sim.scores, sim.y, penalty_factor=-np.ones(10),
                      n_folds=3)
    with pytest.raises(ValueError, match="every score is unpenalized"):
        multi_pgs_fit(sim.scores, sim.y, penalty_factor=np.zeros(10),
                      n_folds=3)


def test_unknown_score_id_is_named_in_the_error():
    sim = simulate_panel(n=300, n_scores=6, seed=12)
    with pytest.raises(ValueError, match="unknown score id 'nope'"):
        multi_pgs_fit(sim.scores, sim.y, n_folds=3, score_ids=sim.score_ids,
                      unpenalized_scores=["nope"])


def test_normalized_score_and_covar_ids_must_be_unique():
    rng = np.random.default_rng(407)
    scores = rng.normal(size=(80, 3))
    y = rng.normal(size=80)
    with pytest.raises(ValueError, match="score_ids must be unique"):
        multi_pgs_fit(scores, y, score_ids=[1, "1", 2], n_folds=3)
    with pytest.raises(ValueError, match="covar_ids must be unique"):
        multi_pgs_fit(scores, y, covar=rng.normal(size=(80, 2)),
                      covar_ids=["age", "age"], n_folds=3)


def test_missing_values_raise_by_default_and_can_be_imputed():
    sim = simulate_panel(n=600, n_scores=12, seed=13)
    scores = sim.scores.copy()
    scores[0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        multi_pgs_fit(scores, sim.y, n_folds=3, n_lambda=15)
    fit = multi_pgs_fit(scores, sim.y, n_folds=3, n_lambda=15, missing="mean",
                        seed=0)
    assert fit.log["imputed_missing"] == 1


def test_nested_mean_imputation_uses_each_outer_training_set(monkeypatch):
    """Outer validation feature values must not set training imputation means."""
    original = stack_mod._impute_from_training
    imputed = []

    def checked(X, train):
        assert np.isnan(X[0, 0])  # nested assessment received the raw matrix
        out = original(X, train)
        imputed.append(out[0, 0])
        observed = X[train, 0]
        expected = np.mean(observed[np.isfinite(observed)])
        assert out[0, 0] == pytest.approx(expected)
        return out

    monkeypatch.setattr(stack_mod, "_impute_from_training", checked)
    rng = np.random.default_rng(406)
    scores = rng.normal(size=(120, 5))
    scores[0, 0] = np.nan
    multi_pgs_fit(scores, rng.normal(size=120), missing="mean", n_folds=3,
                  assessment_folds=3, n_lambda=5, seed=0)
    assert len(imputed) == 3
    assert len(set(np.round(imputed, 12))) > 1


def test_constant_score_is_dropped_not_divided_by_zero():
    sim = simulate_panel(n=600, n_scores=12, seed=14)
    scores = sim.scores.copy()
    scores[:, 5] = 3.0
    fit = multi_pgs_fit(scores, sim.y, n_folds=3, n_lambda=20, seed=0)
    assert fit.beta[5] == 0.0
    assert fit.log["dropped_constant"] == 1
    assert np.all(np.isfinite(fit.beta))


def test_seed_makes_the_fit_reproducible():
    sim = simulate_panel(n=800, n_scores=15, seed=15)
    a = multi_pgs_fit(sim.scores, sim.y, n_folds=4, n_lambda=20, seed=42)
    b = multi_pgs_fit(sim.scores, sim.y, n_folds=4, n_lambda=20, seed=42)
    assert np.array_equal(a.beta, b.beta)


def test_alpha_grid_is_searched_per_fold():
    sim = simulate_panel(n=1200, n_scores=20, n_causal=3, seed=16)
    fit = multi_pgs_fit(sim.scores, sim.y, alpha=[1.0, 0.5, 0.1], n_folds=4,
                        n_lambda=25, seed=0)
    assert set(fit.log["alphas"]) == {1.0, 0.5, 0.1}
    assert {f.alpha for f in fit.folds} <= {1.0, 0.5, 0.1}


def test_mixed_alpha_search_uses_an_independent_lambda_anchor_per_alpha():
    rng = np.random.default_rng(410)
    scores = rng.normal(size=(90, 5))
    y = scores[:, 0] + rng.normal(size=90)
    fit = multi_pgs_fit(
        scores, y, alpha=[1.0, 0.01], n_folds=3, assessment_folds=2,
        n_lambda=4, lambda_min_ratio=0.1, seed=0)
    maxima = np.asarray(fit.log["lambda_max_by_alpha"])
    minima = np.asarray(fit.log["lambda_min_by_alpha"])
    assert maxima[1] / maxima[0] == pytest.approx(100.0)
    assert minima == pytest.approx(maxima * 0.1)
    # In particular, the lasso reaches its own low-penalty endpoint rather than
    # inheriting the much larger endpoint anchored by alpha=0.01.
    assert minima[0] < maxima[1] * 0.01


def test_ridge_has_an_explicit_exact_unpenalized_baseline():
    """No finite ridge lambda is the null, so it must be fitted separately."""
    rng = np.random.default_rng(403)
    scores = rng.normal(size=(300, 8))
    covar = rng.normal(size=(300, 1))
    y = 4.0 * covar[:, 0]
    fit = multi_pgs_fit(scores, y, covar=covar, alpha=0.0, n_folds=4,
                        assessment_folds=4, n_lambda=20, seed=1)
    assert "null_model" in fit.log
    assert fit.covar_beta[0] == pytest.approx(4.0, abs=1e-8)
    assert all(f.lam_index == -1 and np.isinf(f.lam) for f in fit.folds)
    assert all(f.loss == pytest.approx(f.null_loss) for f in fit.folds)


def test_cmsa_average_does_not_filter_unfavorable_folds(monkeypatch):
    """Every fold-selected vector enters once the outer gate passes."""
    values = iter([1.0, 2.0, 3.0, 4.0])

    def fake_fold(X, y, tr, val, pf, alphas, lambdas, family, n_abort,
                  dfmax, tol, max_iter, *, gaussian_stats=None):
        value = next(values)
        return {"beta": np.array([value, 0.0]), "intercept": value,
                "loss": 0.5 if value in (1.0, 3.0) else 1.5,
                "null_loss": 1.0, "alpha": 1.0, "alpha_index": 0,
                "lam": 0.1, "lam_index": 1, "n_val": int(val.size)}

    def fake_assessment(*args, **kwargs):
        return 0.0, 1.0, 0.0, 2

    monkeypatch.setattr(stack_mod, "_fit_one_fold", fake_fold)
    monkeypatch.setattr(stack_mod, "_nested_cv_assessment", fake_assessment)
    rng = np.random.default_rng(404)
    scores = rng.normal(size=(40, 2))
    fit = multi_pgs_fit(scores, rng.normal(size=40), n_folds=4,
                        assessment_folds=2, n_lambda=3, seed=0)
    assert fit.beta.tolist() == pytest.approx([2.5, 0.0])
    assert fit.intercept == pytest.approx(2.5)
    assert fit.n_folds_used == 4
    assert all(f.used for f in fit.folds)
    assert any(f.loss > f.null_loss for f in fit.folds)


def test_gaussian_subtracted_statistics_match_direct_standardization():
    """Fold reuse must be algebraically equivalent to forming its Gram anew."""
    rng = np.random.default_rng(405)
    X = rng.normal(size=(180, 9))
    y = rng.normal(size=180)
    val = np.arange(0, 180, 4)
    tr = stack_mod._complement(180, val)
    total = stack_mod._gaussian_stats(X, y)
    stats = stack_mod._subtract_gaussian_stats(
        total,
        stack_mod._gaussian_stats_at_origin(X, y, val, reference=total))
    center, scale, dead, G, r, ybar = stack_mod._gaussian_system(stats)
    Xs = (X[tr] - X[tr].mean(axis=0)) / X[tr].std(axis=0)
    direct_G = Xs.T @ Xs / tr.size
    direct_r = Xs.T @ (y[tr] - y[tr].mean()) / tr.size
    assert not dead.any()
    assert np.allclose(center, X[tr].mean(axis=0), atol=1e-14)
    assert np.allclose(scale, X[tr].std(axis=0), atol=1e-14)
    assert ybar == pytest.approx(y[tr].mean(), abs=1e-14)
    assert np.allclose(G, direct_G, atol=2e-14)
    assert np.allclose(r, direct_r, atol=2e-14)


def test_gaussian_statistics_are_stable_for_large_offsets():
    """Moment reuse must not erase ordinary variance beside a large mean."""
    rng = np.random.default_rng(408)
    X = 1e12 + rng.normal(size=(240, 4))
    y = rng.normal(size=240)
    val = np.arange(0, 240, 4)
    tr = stack_mod._complement(240, val)
    total = stack_mod._gaussian_stats(X, y)
    held = stack_mod._gaussian_stats_at_origin(X, y, val, reference=total)
    stats = stack_mod._subtract_gaussian_stats(total, held)
    _, scale, dead, G, r, _ = stack_mod._gaussian_system(stats)
    Xs = (X[tr] - X[tr].mean(axis=0)) / X[tr].std(axis=0)
    direct_G = Xs.T @ Xs / tr.size
    direct_r = Xs.T @ (y[tr] - y[tr].mean()) / tr.size
    assert not dead.any()
    assert np.allclose(scale, X[tr].std(axis=0), rtol=2e-5)
    assert np.allclose(G, direct_G, atol=3e-5)
    assert np.allclose(r, direct_r, atol=3e-5)


def test_predict_accepts_a_single_individual():
    sim = simulate_panel(n=400, n_scores=8, seed=17)
    fit = multi_pgs_fit(sim.scores, sim.y, n_folds=3, n_lambda=15, seed=0)
    assert fit.multi_pgs(sim.scores[0]).shape == (1,)
    with pytest.raises(ValueError, match="8 columns"):
        fit.multi_pgs(sim.scores[:, :4])


def test_input_validation():
    sim = simulate_panel(n=200, n_scores=6, seed=18)
    with pytest.raises(ValueError, match="family must be"):
        multi_pgs_fit(sim.scores, sim.y, family="poisson")
    with pytest.raises(ValueError, match="y must have length"):
        multi_pgs_fit(sim.scores, sim.y[:10])
    with pytest.raises(ValueError, match="n_folds"):
        multi_pgs_fit(sim.scores, sim.y, n_folds=1)
    with pytest.raises(ValueError, match="alpha"):
        multi_pgs_fit(sim.scores, sim.y, alpha=1.5, n_folds=3)


@pytest.mark.parametrize(
    "kwargs, message",
    [({"n_folds": 2.5}, "positive integer"),
     ({"assessment_folds": True}, "positive integer"),
     ({"n_lambda": 0}, "positive integer"),
     ({"n_abort": 0}, "positive integer"),
     ({"max_iter": 0}, "positive integer"),
     ({"dfmax": -1}, "non-negative integer"),
     ({"dfmax": 1.5}, "non-negative integer"),
     ({"tol": 0.0}, "strictly positive"),
     ({"tol": np.nan}, "strictly positive"),
     ({"lambda_min_ratio": 0.0}, r"\(0, 1\]"),
     ({"lambda_min_ratio": 1.1}, r"\(0, 1\]")])
def test_individual_fit_validates_numerical_controls(kwargs, message):
    rng = np.random.default_rng(411)
    scores = rng.normal(size=(40, 4))
    with pytest.raises(ValueError, match=message):
        multi_pgs_fit(scores, rng.normal(size=40), **kwargs)


def test_fit_log_and_fold_results_expose_iteration_exhaustion():
    rng = np.random.default_rng(412)
    scores = rng.normal(size=(70, 4))
    y = scores[:, 0] - scores[:, 1] + rng.normal(size=70)
    fit = multi_pgs_fit(
        scores, y, n_folds=3, assessment_folds=2, n_lambda=4, n_abort=4,
        tol=1e-30, max_iter=1, seed=0)
    assert fit.log["solver_converged"] is False
    assert fit.log["n_iteration_exhausted"] > 0
    assert "convergence_warning" in fit.log
    assert any(not fold.converged for fold in fit.folds)
    assert any(fold.n_iteration_exhausted > 0 for fold in fit.folds)


def test_multi_pgs_rejects_a_reordered_panel():
    """Positional matching silently halves accuracy; ids must be checked."""
    sim = simulate_panel(n=1200, n_scores=10, n_causal=3, h2=0.5, seed=30)
    fit = multi_pgs_fit(sim.scores, sim.y, n_folds=4, n_lambda=25, seed=0,
                        score_ids=sim.score_ids)
    swapped = sim.score_ids.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    with pytest.raises(ValueError, match="different order"):
        fit.multi_pgs(sim.scores, score_ids=swapped)
    with pytest.raises(ValueError, match="not in the fit"):
        fit.multi_pgs(sim.scores, score_ids=[f"other_{i}" for i in range(10)])
    # Matching ids, and no ids at all, both still work.
    assert fit.multi_pgs(sim.scores, score_ids=sim.score_ids).shape == (1200,)
    assert fit.multi_pgs(sim.scores).shape == (1200,)


def test_multi_pgs_accepts_a_panel_and_checks_it(tmp_path):
    from multipgs import panel_from_catalog, simulate_target
    target = simulate_target(str(tmp_path / "sim"), n=150, n_variants=200,
                             n_scores=4, seed=31)
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    y = target["true_scores"][:, 0] + np.random.default_rng(0).normal(size=150)
    fit = multi_pgs_fit(panel.scores, y, n_folds=3, n_lambda=20, seed=0,
                        score_ids=panel.score_ids)
    assert np.allclose(fit.multi_pgs(panel), fit.multi_pgs(panel.scores))
    with pytest.raises(ValueError, match="different order"):
        fit.multi_pgs(panel.select([3, 2, 1, 0]))
