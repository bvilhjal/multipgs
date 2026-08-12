"""Safety and scale contracts for summary-statistic fitting."""

import numpy as np
import pytest

from multipgs.sumstat import (
    _range_basis,
    evaluate_sumstat,
    multi_pgs_sumstats,
    pseudo_r2,
    score_gram,
    score_moments,
    subsample_score_moments,
)


@pytest.mark.parametrize("gram", [
    np.array([[4.0, 3.0], [3.0, 9.0]]),
    np.ones((2, 2)),
])
def test_range_basis_defines_an_orthogonal_projection(gram):
    vectors = _range_basis(gram)[0]
    projection = vectors @ vectors.T

    assert np.allclose(vectors.T @ vectors, np.eye(vectors.shape[1]))
    assert np.allclose(projection, projection.T)
    assert np.allclose(projection @ projection, projection)


def test_fit_exposes_raw_coefficients_and_raw_moments():
    weights = np.diag([2.0, 0.5])
    z = np.array([0.4, 0.1])
    fit = multi_pgs_sumstats(weights, z, np.eye(2), weights_gwas=weights,
                             n_lambda=12,
                             tune="none")

    assert np.allclose(fit.gram, weights.T @ weights)
    assert np.allclose(fit.r, weights.T @ z)
    assert np.allclose(fit.beta_std, fit.beta * fit.score_sd)
    assert np.allclose(fit.path_std, fit.path * fit.score_sd[None, :])
    assert np.allclose(fit.variant_weights(weights), weights @ fit.beta)
    scores = np.array([[1.0, 2.0], [-1.0, 0.5]])
    assert np.allclose(fit.multi_pgs(scores), scores @ fit.beta)


def test_moment_validation_is_invariant_to_positive_score_rescaling():
    weights = np.eye(2)
    scaled_weights = weights.copy()
    scaled_weights[:, 1] *= 1e-6
    z = np.array([0.1, 0.5])
    kwargs = {"tune": "none", "ld_shrinkage": 0.2, "n_lambda": 20}

    fit = multi_pgs_sumstats(
        weights, z, np.eye(2), weights_gwas=weights, **kwargs)
    scaled = multi_pgs_sumstats(
        scaled_weights, z, np.eye(2), weights_gwas=scaled_weights, **kwargs)

    assert fit.log["gram_rank"] == scaled.log["gram_rank"] == 2
    assert np.allclose(fit.beta_std, scaled.beta_std)
    assert np.allclose(fit.variant_weights(weights),
                       scaled.variant_weights(scaled_weights))
    raw_scores = np.array([[1.0, 2.0], [-0.5, 0.25]])
    scaled_scores = raw_scores.copy()
    scaled_scores[:, 1] *= 1e-6
    assert np.allclose(fit.multi_pgs(raw_scores),
                       scaled.multi_pgs(scaled_scores))
    assert fit.selection_mse == pytest.approx(scaled.selection_mse)


def test_rank_deficient_projection_is_invariant_to_score_rescaling():
    scale = 1e-6
    weights = np.eye(2)
    scaled_weights = weights.copy()
    scaled_weights[:, 1] *= scale
    c = np.array([0.1, 0.05])
    scaled_c = np.array([c[0], scale * c[1]])
    ld = np.ones((2, 2))

    fit = multi_pgs_sumstats(
        weights, c, ld, weights_gwas=weights, tune="none",
        ld_shrinkage=0.2, n_lambda=20)
    scaled_fit = multi_pgs_sumstats(
        scaled_weights, c, ld, weights_gwas=scaled_weights, tune="none",
        ld_shrinkage=0.2, n_lambda=20)
    assert np.allclose(fit.beta_std, scaled_fit.beta_std)
    assert np.allclose(fit.frozen_variant_weights(weights),
                       scaled_fit.frozen_variant_weights(scaled_weights))
    assert fit.log["discarded_ld_null_c_fraction"] == pytest.approx(
        scaled_fit.log["discarded_ld_null_c_fraction"])

    gram = np.ones((2, 2))
    scaled_gram = np.diag([1.0, scale]) @ gram @ np.diag([1.0, scale])
    beta = np.array([0.3, 0.2])
    scaled_beta = np.array([beta[0], beta[1] / scale])
    assert pseudo_r2(beta, gram, c) == pytest.approx(
        pseudo_r2(scaled_beta, scaled_gram, scaled_c))
    evaluated = evaluate_sumstat(beta, c, gram, regime="A")
    scaled_evaluated = evaluate_sumstat(
        scaled_beta, scaled_c, scaled_gram, regime="A")
    assert evaluated.r2 == pytest.approx(scaled_evaluated.r2)
    assert evaluated.mse == pytest.approx(scaled_evaluated.mse)
    assert evaluated.log["discarded_beta_c_null"] == pytest.approx(
        scaled_evaluated.log["discarded_beta_c_null"])


def test_mse_selection_rejects_a_sign_reversed_tuning_association():
    z = np.array([0.4, 0.2])
    weights = np.eye(2)
    fit = multi_pgs_sumstats(
        weights, z, np.eye(2), weights_gwas=weights, z_valid=-z,
        ld_valid=np.eye(2), weights_gwas_valid=weights,
        weights_ld_valid=weights, n_lambda=20)

    # Squared correlation would prefer the least-penalized, wrong-sign model.
    # MSE correctly keeps the null model when every fitted coefficient points
    # opposite to the independent tuning association.
    assert np.array_equal(fit.beta, np.zeros(2))
    assert fit.lambda_index == 0
    assert fit.selection_mse == pytest.approx(1.0)
    assert np.isnan(fit.pseudo_r2)
    assert fit.log["selection_metric"] == "MSE"
    assert fit.log["regime"] == "B"


def test_ld_shrinkage_zero_is_identity_and_positive_delta_shrinks():
    weights = np.array([[1.0, 1.0], [0.0, 0.1], [0.1, 0.0]])
    z = np.array([0.3, 0.03, 0.03])
    plain = multi_pgs_sumstats(weights, z, np.eye(3), weights_gwas=weights,
                               n_lambda=30,
                               tune="none")
    zero = multi_pgs_sumstats(weights, z, np.eye(3), weights_gwas=weights,
                              n_lambda=30,
                              tune="none", ld_shrinkage=0.0)
    shrunk = multi_pgs_sumstats(weights, z, np.eye(3), weights_gwas=weights,
                                n_lambda=30,
                                tune="none", ld_shrinkage=1.0)

    assert np.array_equal(plain.lambdas, zero.lambdas)
    assert np.array_equal(plain.path, zero.path)
    assert np.array_equal(plain.beta, zero.beta)
    assert not np.allclose(shrunk.beta, plain.beta)
    assert np.linalg.norm(shrunk.beta) < np.linalg.norm(plain.beta)


def test_ld_shrinkage_grid_is_selected_by_unshrunk_mse():
    weights = np.array([[1.0, 1.0], [0.0, 0.1], [0.1, 0.0]])
    z = np.array([0.3, 0.03, 0.03])
    grid = np.array([0.0, 0.1, 1.0])
    fit = multi_pgs_sumstats(weights, z, np.eye(3), weights_gwas=weights,
                             n_lambda=30,
                             tune="none", ld_shrinkage=grid)

    assert fit.log["ld_shrinkage"] in grid
    assert fit.log["n_shrinkage"] == grid.size
    assert len(fit.log["delta_audit"]) == grid.size
    for row, delta in zip(fit.log["delta_audit"], grid):
        assert row["delta"] == pytest.approx(delta)
        assert row["n_fitted"] > 0
        assert row["best_index"] is not None
        assert row["best_lambda"] > 0.0
        assert np.isfinite(row["selection_mse"])
    unshrunk_mse = (1.0 - 2.0 * float(fit.beta @ fit.r)
                    + float(fit.beta @ fit.gram @ fit.beta))
    assert fit.selection_mse == pytest.approx(unshrunk_mse)


@pytest.mark.parametrize("bad", [[], [-0.1], [0.0, np.nan]])
def test_invalid_ld_shrinkage_grids_are_rejected(bad):
    with pytest.raises(ValueError, match="ld_shrinkage must be"):
        multi_pgs_sumstats(np.eye(2), np.array([0.1, 0.05]), np.eye(2),
                           weights_gwas=np.eye(2), tune="none",
                           ld_shrinkage=bad)


def test_same_gram_fit_projects_noisy_c_out_of_the_ld_nullspace():
    fit = multi_pgs_sumstats(
        np.eye(2), np.array([0.2, -0.2]), np.ones((2, 2)),
        weights_gwas=np.eye(2), tune="none", ld_shrinkage=0.1)
    assert np.allclose(fit.beta, 0.0)
    assert fit.log["discarded_ld_null_c_fraction"] == pytest.approx(1.0)
    assert "outside the LD Gram range" in fit.log["moment_warning"]


def test_materially_indefinite_fitting_moments_are_refused():
    indefinite = np.array([[1.0, 1.1], [1.1, 1.0]])
    with pytest.raises(ValueError, match="materially indefinite"):
        multi_pgs_sumstats(np.eye(2), np.array([0.1, 0.1]), indefinite,
                           weights_gwas=np.eye(2), tune="none")


def test_quantized_ld_indefiniteness_is_refused_globally():
    # Rounded from a rank-deficient empirical correlation. Every diagonal is
    # valid and many individual directions are positive, but D8 rounding makes
    # the complete covariance materially indefinite (min eigenvalue -0.0058).
    ld = np.array([
        [127, -63, 19, -77, -21, -121],
        [-63, 127, -118, -50, 119, 94],
        [19, -118, 127, 89, -127, -57],
        [-77, -50, 89, 127, -87, 41],
        [-21, 119, -127, -87, 127, 59],
        [-121, 94, -57, 41, 59, 127],
    ], dtype=np.int8)
    with pytest.raises(ValueError, match="materially indefinite"):
        multi_pgs_sumstats(np.eye(6), np.zeros(6), ld,
                           weights_gwas=np.eye(6), tune="none")


def test_standalone_pumas_reports_noisy_signal_outside_the_gram_range():
    _, _, log = subsample_score_moments(
        np.array([0.2, -0.2]), np.ones((2, 2)), 10_000, 7_500,
        var_y=1.0)
    assert "outside the LD Gram range" in log["moment_warning"]


def test_safe_tuning_and_input_contracts_are_explicit():
    weights, z, ld = np.eye(2), np.array([0.1, 0.05]), np.eye(2)
    with pytest.raises(ValueError, match="tune='auto' has no tuning data"):
        multi_pgs_sumstats(weights, z, ld, weights_gwas=weights)
    with pytest.raises(ValueError, match=r"alpha.*\[0, 1\]"):
        multi_pgs_sumstats(weights, z, ld, weights_gwas=weights, alpha=1.1,
                           tune="none")
    with pytest.raises(ValueError, match="var_y must be"):
        multi_pgs_sumstats(weights, z, ld, weights_gwas=weights, var_y=0.0,
                           tune="none")
    with pytest.raises(ValueError, match="does not use ld_valid"):
        multi_pgs_sumstats(weights, z, ld, weights_gwas=weights, ld_valid=ld,
                           tune="none")
    with pytest.raises(ValueError, match="z_valid contains non-finite"):
        multi_pgs_sumstats(
            weights, z, ld, weights_gwas=weights, z_valid=[np.nan, 0.0],
            ld_valid=ld, weights_gwas_valid=weights,
            weights_ld_valid=weights)
    with pytest.raises(ValueError, match="weights_independent_of_z=True"):
        multi_pgs_sumstats(weights, z, ld, weights_gwas=weights,
                           tune="pumas", n_eff=1000)


def test_same_moments_cannot_be_relabelled_as_external_or_tuning():
    beta = np.array([0.2, 0.0])
    c = np.array([0.1, 0.0])
    for regime in ("A", "B"):
        with pytest.raises(ValueError, match="equal to the fitting moments"):
            evaluate_sumstat(beta, c, np.eye(2), regime=regime, fitted_on=c)


def test_pumas_reuses_one_gram_factor_across_repeats(monkeypatch):
    calls = 0
    eigh = np.linalg.eigh

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return eigh(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    multi_pgs_sumstats(np.eye(3), np.array([0.15, 0.08, 0.03]), np.eye(3),
                       weights_gwas=np.eye(3), tune="pumas", n_eff=10_000,
                       n_repeats=5, rng=0,
                       weights_independent_of_z=True)
    assert calls == 1


def test_weight_digest_and_sparse_trailing_variants_guard_deployment():
    weights = [(np.array([0]), np.array([1.0]))]
    z = np.array([0.1, 0.0, 0.0])
    c, gram, _ = score_moments(
        weights, z, np.eye(3), weights_gwas=weights)
    assert c.shape == (1,) and gram.shape == (1, 1)

    fit = multi_pgs_sumstats(
        weights, z, np.eye(3), weights_gwas=weights, tune="none")
    assert fit.n_variants == 3
    assert fit.variant_weights(weights).shape == (3,)
    changed = [(np.array([0]), np.array([2.0]))]
    with pytest.raises(ValueError, match="weights differ"):
        fit.variant_weights(changed)


def test_gwas_and_ld_weights_use_dataset_specific_genotype_scales():
    weights_ld = np.diag([1.0, 3.0])
    weights_gwas = np.diag([2.0, 0.5])
    z = np.array([0.2, 0.1])

    c, gram, _ = score_moments(
        weights_ld, z, np.eye(2), weights_gwas=weights_gwas)
    assert np.allclose(c, [0.4, 0.05])
    assert np.allclose(gram, np.diag([1.0, 9.0]))

    fit = multi_pgs_sumstats(
        weights_ld, z, np.eye(2), weights_gwas=weights_gwas, tune="none",
        ld_shrinkage=0.1)
    assert np.allclose(fit.r, c)
    assert np.allclose(fit.gram, gram)
    assert np.allclose(fit.frozen_variant_weights(weights_ld),
                       weights_ld @ fit.beta)
    with pytest.raises(ValueError, match="weights differ"):
        fit.frozen_variant_weights(weights_gwas)


def test_train_and_tuning_sources_may_have_distinct_variants_and_orders():
    weights_ld = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.2]])
    weights_gwas = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([0.1, 0.2])
    weights_ld_valid = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.2, 0.1], [0.1, 0.3]])
    weights_gwas_valid = np.array([
        [0.0, 1.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.1, 0.1]])
    z_valid = np.array([0.1, 0.2, 0.0, 0.0, 0.0])

    fit = multi_pgs_sumstats(
        weights_ld, z, np.eye(3), weights_gwas=weights_gwas,
        z_valid=z_valid, ld_valid=np.eye(4),
        weights_gwas_valid=weights_gwas_valid,
        weights_ld_valid=weights_ld_valid, n_lambda=12)
    assert fit.log["n_variants_ld"] == 3
    assert fit.log["n_variants_gwas"] == 2
    assert fit.log["n_variants_ld_valid"] == 4
    assert fit.log["n_variants_gwas_valid"] == 5
    assert fit.log["selection"] == "independent GWAS"


def test_full_rank_independent_tuning_preserves_unequal_scale_moments():
    weights_train = np.diag([1.0, 10.0])
    z_train = np.array([0.2, 0.01])
    gram_valid = np.array([[4.0, 3.0], [3.0, 9.0]])
    weights_valid = np.linalg.cholesky(gram_valid).T
    c_valid = np.array([0.4, 0.7])
    z_valid = np.linalg.solve(weights_valid.T, c_valid)

    fit = multi_pgs_sumstats(
        weights_train, z_train, np.eye(2), weights_gwas=weights_train,
        z_valid=z_valid, ld_valid=np.eye(2),
        weights_gwas_valid=weights_valid,
        weights_ld_valid=weights_valid, tune="independent",
        ld_shrinkage=0.1, n_lambda=20)

    assert fit.log["gram_rank"] == fit.log["tuning_gram_rank"] == 2
    assert np.allclose(fit.r, weights_train.T @ z_train)
    assert fit.log["tuning_discarded_ld_null_c_norm"] == pytest.approx(
        0.0, abs=1e-14)
    assert fit.log[
        "training_discarded_by_tuning_ld_c_norm"] == pytest.approx(
            0.0, abs=1e-14)


def test_independent_tuning_is_invariant_to_positive_score_rescaling():
    weights_train = np.diag([1.0, 10.0])
    z_train = np.array([0.2, 0.01])
    gram_valid = np.array([[4.0, 3.0], [3.0, 9.0]])
    weights_valid = np.linalg.cholesky(gram_valid).T
    z_valid = np.linalg.solve(weights_valid.T, np.array([0.4, 0.7]))
    kwargs = {"z_valid": z_valid, "ld_valid": np.eye(2),
              "tune": "independent", "ld_shrinkage": [0.0, 0.1],
              "n_lambda": 20}

    fit = multi_pgs_sumstats(
        weights_train, z_train, np.eye(2), weights_gwas=weights_train,
        weights_gwas_valid=weights_valid, weights_ld_valid=weights_valid,
        **kwargs)
    score_rescaling = np.array([3.0, 1e-6])
    scaled_train = weights_train * score_rescaling
    scaled_valid = weights_valid * score_rescaling
    scaled_fit = multi_pgs_sumstats(
        scaled_train, z_train, np.eye(2), weights_gwas=scaled_train,
        weights_gwas_valid=scaled_valid, weights_ld_valid=scaled_valid,
        **kwargs)

    assert np.allclose(fit.beta_std, scaled_fit.beta_std)
    assert np.allclose(fit.frozen_variant_weights(weights_train),
                       scaled_fit.frozen_variant_weights(scaled_train))
    assert fit.selection_mse == pytest.approx(scaled_fit.selection_mse)
    assert fit.log["ld_shrinkage"] == scaled_fit.log["ld_shrinkage"]
    assert fit.log[
        "tuning_discarded_ld_null_c_fraction"] == pytest.approx(
            scaled_fit.log["tuning_discarded_ld_null_c_fraction"])
    assert fit.log[
        "training_discarded_by_tuning_ld_c_fraction"] == pytest.approx(
            scaled_fit.log[
                "training_discarded_by_tuning_ld_c_fraction"])


def test_independent_selection_projects_its_own_gram_null_signal():
    weights = np.eye(2)
    c_train = np.array([0.11921661, -0.06710897])
    c_valid = np.array([0.10002694, 0.01363211])
    fit = multi_pgs_sumstats(
        weights, c_train, np.ones((2, 2)), weights_gwas=weights,
        z_valid=c_valid, ld_valid=np.ones((2, 2)),
        weights_gwas_valid=weights, weights_ld_valid=weights,
        ld_shrinkage=[0.01, 0.1, 1.0], n_lambda=20)
    projected_valid = np.full(2, c_valid.mean())
    expected_mse = (1.0 - 2.0 * float(fit.beta @ projected_valid)
                    + float(fit.beta @ np.ones((2, 2)) @ fit.beta))
    assert fit.selection_mse == pytest.approx(expected_mse)
    assert fit.log["tuning_discarded_ld_null_c_fraction"] > 0.0
    assert fit.log["training_discarded_by_tuning_ld_c_fraction"] > 0.0
    assert np.linalg.norm(fit.beta) < 1.0
    assert np.array_equal(fit.beta, fit.path[fit.lambda_index])


def test_fixed_vector_checks_only_the_direction_beta_uses():
    beta = np.array([1.0, 0.0])
    c = np.array([0.1, 0.2])
    singular = np.diag([1.0, 0.0])
    result = evaluate_sumstat(beta, c, singular, regime="A")
    assert result.r2 == pytest.approx(0.01)
    assert result.mse == pytest.approx(1.8)

    unused_indefinite = np.diag([1.0, -1.0])
    assert pseudo_r2(beta, unused_indefinite, c) == pytest.approx(0.01)
    assert evaluate_sumstat(
        beta, c, unused_indefinite, regime="A").r2 == pytest.approx(0.01)
    unresolved = evaluate_sumstat(
        np.array([0.0, 1.0]), c, singular, regime="A")
    assert np.isnan(unresolved.r2) and unresolved.mse == pytest.approx(1.0)
    assert unresolved.log["discarded_beta_c_null"] == pytest.approx(0.2)


def test_rank_deficient_null_noise_is_projected_before_same_gram_fitting():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(20, 40))
    gram = x.T @ x / x.shape[0]
    z = rng.normal(scale=1.0 / np.sqrt(x.shape[0]), size=40)
    weights = np.eye(40)

    fit = multi_pgs_sumstats(
        weights, z, gram, weights_gwas=weights, tune="none",
        ld_shrinkage=[0.0, 0.1], n_lambda=20)
    assert fit.log["gram_rank"] <= 20
    assert fit.log["discarded_ld_null_c_fraction"] > 0.0
    assert np.all(np.isfinite(fit.beta))
    assert np.isnan(fit.pseudo_r2) or fit.pseudo_r2 <= 1.0 + 1e-8
    assert fit.evaluate(fit.r, fit.gram).regime == "C"
    assert fit.evaluate(fit.c_raw, fit.gram).regime == "C"
    assert evaluate_sumstat(fit, fit.r, fit.gram).regime == "C"
    assert evaluate_sumstat(fit, fit.c_raw, fit.gram).regime == "C"


def test_pumas_projects_full_and_pseudo_split_null_signal():
    weights = np.eye(2)
    fit = multi_pgs_sumstats(
        weights, np.array([0.2, -0.2]), np.ones((2, 2)),
        weights_gwas=weights, tune="pumas", n_eff=1_000, n_repeats=2,
        n_lambda=20, max_iter=100, rng=0, ld_shrinkage=[0.0, 0.1],
        weights_independent_of_z=True)
    assert np.allclose(fit.beta, 0.0)
    assert fit.selection_mse == pytest.approx(1.0, abs=0.01)
    assert fit.log["discarded_ld_null_c_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_lambda_min_ratio_is_validated(bad):
    with pytest.raises(ValueError, match="lambda_min_ratio"):
        multi_pgs_sumstats(
            np.eye(2), np.array([0.1, 0.05]), np.eye(2),
            weights_gwas=np.eye(2), tune="none", lambda_min_ratio=bad)


@pytest.mark.parametrize("name,bad", [
    ("n_lambda", 0), ("n_lambda", 1.5), ("n_lambda", True),
    ("max_iter", 0), ("max_iter", 2.5), ("max_iter", np.nan),
])
def test_positive_integer_solver_controls_are_validated(name, bad):
    kwargs = {name: bad}
    with pytest.raises(ValueError, match=name):
        multi_pgs_sumstats(
            np.eye(2), np.array([0.1, 0.05]), np.eye(2),
            weights_gwas=np.eye(2), tune="none", **kwargs)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_positive_finite_tolerance_is_validated(bad):
    with pytest.raises(ValueError, match="tol"):
        multi_pgs_sumstats(
            np.eye(2), np.array([0.1, 0.05]), np.eye(2),
            weights_gwas=np.eye(2), tune="none", tol=bad)


def test_unused_tuning_inputs_are_rejected():
    weights = np.eye(2)
    z = np.array([0.1, 0.05])
    with pytest.raises(ValueError, match="does not use n_eff"):
        multi_pgs_sumstats(
            weights, z, np.eye(2), weights_gwas=weights, tune="none",
            n_eff=1_000)
    with pytest.raises(ValueError, match="does not use z_valid"):
        multi_pgs_sumstats(
            weights, z, np.eye(2), weights_gwas=weights, tune="pumas",
            n_eff=1_000, weights_independent_of_z=True, z_valid=z)


def test_unpenalized_score_ids_define_an_unshrunk_baseline():
    weights = np.eye(2)
    fit = multi_pgs_sumstats(
        weights, np.array([0.3, 0.1]), np.eye(2), weights_gwas=weights,
        score_ids=["baseline", "extra"], unpenalized_scores=["baseline"],
        ld_shrinkage=10.0, tune="none", n_lambda=12)
    assert fit.log["unpenalized_scores"] == 1
    assert fit.beta[0] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="unknown unpenalized score id"):
        multi_pgs_sumstats(
            weights, np.array([0.3, 0.1]), np.eye(2), weights_gwas=weights,
            score_ids=["baseline", "extra"],
            unpenalized_scores=["missing"], tune="none")
    with pytest.raises(ValueError, match="duplicate scores"):
        multi_pgs_sumstats(
            weights, np.array([0.3, 0.1]), np.eye(2), weights_gwas=weights,
            unpenalized_scores=[0, 0], tune="none")


def test_explicit_dense_ld_variant_count_must_match_both_inputs():
    with pytest.raises(ValueError, match="dense LD weights have 2 rows"):
        score_moments(
            np.eye(2), np.array([0.1, 0.05]), np.eye(2),
            weights_gwas=np.eye(2), n_variants_ld=3)
    with pytest.raises(ValueError, match="dense weights have 2 rows"):
        score_gram(np.eye(2), np.eye(2), n_variants=3)
    fit = multi_pgs_sumstats(
        np.eye(2), np.array([0.1, 0.05]), np.eye(2),
        weights_gwas=np.eye(2), tune="none")
    with pytest.raises(ValueError, match="dense weights have 2 rows"):
        fit.frozen_variant_weights(np.eye(2), n_variants_ld=3)


def test_weights_gwas_is_mandatory_even_for_identical_scales():
    with pytest.raises(ValueError, match="weights_gwas is required separately"):
        score_moments(np.eye(2), np.array([0.1, 0.05]), np.eye(2))
    with pytest.raises(ValueError, match="weights_gwas is required separately"):
        multi_pgs_sumstats(
            np.eye(2), np.array([0.1, 0.05]), np.eye(2), tune="none")
