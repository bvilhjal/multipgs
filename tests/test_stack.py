"""The CMSA multi-PGS estimator."""

import numpy as np
import pytest

from multipgs import multi_pgs_fit, r2, simulate_panel


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
    """cv_r2 must be honest: selecting the penalty per fold and scoring on
    that same fold reports a positive R2 for noise. It must not."""
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        fit = multi_pgs_fit(rng.normal(size=(400, 50)), rng.normal(size=400),
                            n_folds=5, n_lambda=30, seed=seed)
        assert fit.cv_r2 < 0.01


def test_cross_validated_r2_tracks_held_out_accuracy():
    sim = simulate_panel(n=4000, n_scores=30, n_causal=4, h2=0.5, seed=20)
    tr, te = slice(0, 3000), slice(3000, None)
    fit = multi_pgs_fit(sim.scores[tr], sim.y[tr], n_folds=5, n_lambda=40,
                        seed=0)
    held_out = r2(sim.y[te], fit.multi_pgs(sim.scores[te]))
    assert fit.cv_r2 == pytest.approx(held_out, abs=0.06)
    assert fit.log["cv_loss"] < fit.log["cv_null_loss"]


def test_cross_validated_r2_is_incremental_over_covariates():
    """It must describe the scores, not the covariates riding along."""
    from multipgs import incremental_r2
    sim = simulate_panel(n=4000, n_scores=25, n_causal=3, h2=0.5, n_covar=2,
                         seed=21)
    y = sim.y + 3.0 * sim.covar[:, 0]           # a covariate that dominates
    tr, te = slice(0, 3000), slice(3000, None)
    fit = multi_pgs_fit(sim.scores[tr], y[tr], covar=sim.covar[tr], n_folds=5,
                        n_lambda=40, seed=0)
    held_out = incremental_r2(y[te], fit.multi_pgs(sim.scores[te]),
                              sim.covar[te])
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


def test_missing_values_raise_by_default_and_can_be_imputed():
    sim = simulate_panel(n=600, n_scores=12, seed=13)
    scores = sim.scores.copy()
    scores[0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        multi_pgs_fit(scores, sim.y, n_folds=3, n_lambda=15)
    fit = multi_pgs_fit(scores, sim.y, n_folds=3, n_lambda=15, missing="mean",
                        seed=0)
    assert fit.log["imputed_missing"] == 1


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
