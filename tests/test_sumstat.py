"""Fitting the combination from summary statistics alone."""

import numpy as np
import pytest

from multipgs import _coord
from multipgs.sumstat import (
    REGIMES,
    align_to_reference,
    evaluate_sumstat,
    multi_pgs_sumstats,
    pseudo_r2,
    score_gram,
    score_moments,
    subsample_score_moments,
)


def _setup(seed=0, n=3000, m=200, k=10):
    """Standardized genotypes, sparse score weights, and a phenotype.

    In-sample LD makes the summary-statistic identity exact, which is what lets
    these tests assert equality rather than approximate agreement.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m))
    x = (x - x.mean(0)) / x.std(0)
    w = np.zeros((m, k))
    for j in range(k):
        idx = rng.choice(m, size=int(rng.integers(5, 30)), replace=False)
        w[idx, j] = rng.normal(size=idx.size)
    scores = x @ w
    y = (1.2 * scores[:, 0] / scores[:, 0].std()
         + 0.8 * scores[:, 3] / scores[:, 3].std()
         + rng.normal(size=n) * 2.0)
    y -= y.mean()
    return x, w, scores, y, x.T @ x / n, x.T @ y / n


def test_gram_is_the_score_covariance():
    """W'DW is cov(scores) and W'z is scores'y/n — the whole premise."""
    x, w, scores, y, ld, z = _setup()
    gram, var = score_gram(w, ld)
    n = x.shape[0]
    assert np.allclose(gram, scores.T @ scores / n, atol=1e-10)
    assert np.allclose(var, np.diag(scores.T @ scores / n), atol=1e-10)
    wz = w.T @ z
    assert np.allclose(wz, scores.T @ y / n, atol=1e-10)


def test_sparse_and_dense_weights_agree():
    _, w, _, _, ld, _ = _setup(seed=1)
    pairs = [(np.flatnonzero(w[:, j]), w[np.flatnonzero(w[:, j]), j])
             for j in range(w.shape[1])]
    dense, _ = score_gram(w, ld)
    sparse, _ = score_gram(pairs, ld, n_variants=w.shape[0])
    assert np.allclose(dense, sparse)


@pytest.mark.parametrize("bad", [np.array([0.9]), np.array([np.nan]), ["0"]])
def test_sparse_variant_indices_are_validated_before_integer_cast(bad):
    with pytest.raises(ValueError, match="integer variant indices"):
        score_gram([(bad, np.array([1.0]))], np.eye(1), n_variants=1)


@pytest.mark.parametrize("bad", [2.9, np.nan, True, "2"])
def test_sparse_n_variants_is_validated_before_integer_cast(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        score_gram([(np.array([0]), np.array([1.0]))], np.eye(2),
                   n_variants=bad)


def test_ld_block_indices_are_validated_before_integer_cast():
    with pytest.raises(ValueError, match="integer variant indices"):
        score_gram(
            [(np.array([0]), np.array([1.0]))],
            [(np.eye(1), np.array([0.5]))], n_variants=1)


def test_duplicate_sparse_entries_are_coalesced_consistently():
    weights = [(np.array([0, 0]), np.array([1.0, 2.0]))]
    gram, var = score_gram(weights, np.eye(1), n_variants=1)
    c, gram_again, _ = score_moments(weights, np.array([1.0]), np.eye(1),
                                     weights_gwas=weights,
                                     n_variants_ld=1)
    assert gram[0, 0] == pytest.approx(9.0)
    assert var[0] == pytest.approx(9.0)
    assert c[0] == pytest.approx(3.0)
    assert np.array_equal(gram_again, gram)


def test_block_streaming_matches_one_dense_matrix():
    """The block path must give the same Gram as the whole matrix at once."""
    _, w, _, _, ld, _ = _setup(seed=2, m=120, k=6)
    size = 30
    blocks = [(ld[i:i + size, i:i + size], np.arange(i, i + size))
              for i in range(0, ld.shape[0], size)]
    # A block-diagonal reference is not the full matrix, so compare against the
    # same blocks assembled densely rather than against `ld`.
    block_dense = np.zeros_like(ld)
    for corr, idx in blocks:
        block_dense[np.ix_(idx, idx)] = corr
    from_blocks, _ = score_gram(w, blocks)
    from_dense, _ = score_gram(w, block_dense)
    assert np.allclose(from_blocks, from_dense)


def test_sparse_blocks_multiply_only_the_active_score_columns(monkeypatch):
    import ldpred3

    seen = []

    def tracked(corr, rhs):
        seen.append(rhs.shape)
        return np.asarray(corr) @ rhs

    monkeypatch.setattr(ldpred3, "ld_matmul", tracked)
    weights = []
    for score in range(40):
        variant = score % 4
        weights.append((np.array([variant]), np.array([score + 1.0])))
    blocks = ((np.eye(2), np.arange(start, start + 2))
              for start in (0, 2))
    gram, _ = score_gram(weights, blocks, n_variants=4)
    assert gram.shape == (40, 40)
    assert seen == [(2, 20), (2, 20)]


def test_it_reproduces_the_individual_level_fit_exactly():
    """Same unadjusted centred data gives the exact stacked Gaussian fit."""
    x, w, scores, y, ld, z = _setup(seed=3)
    n, k = x.shape[0], w.shape[1]
    sd = scores.std(0)
    standardized = scores / sd
    gram = standardized.T @ standardized / n
    r = standardized.T @ y / n
    pf = np.ones(k)
    _, grad = _coord.unpenalized_fit(gram, r, pf)
    lambdas = _coord.lambda_grid(grad, pf, 1.0, n_lambda=30)
    expected, _ = _coord.enet_path_gaussian(gram, r, pf=pf, alpha=1.0,
                                            lambdas=lambdas)

    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=30, alpha=1.0,
                             var_y=float(y.var()), tune="none")
    assert np.allclose(fit.score_sd, sd, atol=1e-10)
    assert np.allclose(fit.lambdas, lambdas[:fit.lambdas.size], atol=1e-10)
    assert np.allclose(fit.path_std, expected[:fit.path.shape[0]], atol=1e-10)


def test_variant_weights_reproduce_the_combined_score():
    x, w, scores, y, ld, z = _setup(seed=4)
    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=20,
                             var_y=float(y.var()), tune="none")
    assert np.allclose(x @ fit.variant_weights(w), fit.multi_pgs(scores),
                       atol=1e-10)


def test_pseudo_r2_is_the_realized_r2_when_ld_is_in_sample():
    x, w, scores, y, ld, z = _setup(seed=5)
    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=20,
                             var_y=float(y.var()),
                             tune="none")
    predicted = x @ fit.variant_weights(w)
    realized = np.corrcoef(predicted, y)[0, 1] ** 2
    assert fit.pseudo_r2 == pytest.approx(realized, abs=1e-8)


def test_in_sample_selection_is_labelled_and_an_independent_gwas_is_too():
    _, w, _, y, ld, z = _setup(seed=6)
    plain = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=15,
                               var_y=float(y.var()), tune="none")
    assert plain.log["selection"] == "in-sample"
    assert "optimistic" in plain.log["warning"]
    assert "optimistic" in plain.summary()

    honest = multi_pgs_sumstats(
        w, z, ld, weights_gwas=w, n_lambda=15, z_valid=z * 0.9,
        ld_valid=ld, weights_gwas_valid=w, weights_ld_valid=w,
        var_y=float(y.var()))
    assert honest.log["selection"] == "independent GWAS"
    assert honest.log["regime"] == "B"
    assert "warning" not in honest.log


def test_a_score_with_no_variance_is_dropped_not_nan():
    _, w, _, y, ld, z = _setup(seed=7)
    w = w.copy()
    w[:, 2] = 0.0
    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=15,
                             var_y=float(y.var()), tune="none")
    assert fit.log["n_dead"] == 1
    assert fit.beta[2] == 0.0
    assert np.all(np.isfinite(fit.beta))
    assert np.all(np.isfinite(fit.variant_weights(w)))


def test_an_r2_above_one_is_reported_as_a_scale_diagnostic():
    """var_y left at 1 for an unstandardized y is the mistake this flags."""
    _, w, _, y, ld, z = _setup(seed=8)
    assert y.var() > 2.0
    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=15,
                             tune="none")
    assert "exceeds var_y" in fit.log["moment_warning"]


def test_penalty_factors_can_force_a_score_in():
    _, w, _, y, ld, z = _setup(seed=9)
    pf = np.ones(w.shape[1])
    pf[7] = 0.0
    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=5,
                             penalty_factor=pf,
                             var_y=float(y.var()), tune="none")
    assert fit.beta[7] != 0.0


def test_pseudo_r2_rejects_a_non_psd_reference():
    gram = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="indefinite"):
        pseudo_r2(np.array([0.0, 1.0]), gram, np.array([1.0, 1.0]))
    assert np.isnan(pseudo_r2(np.zeros(2), np.eye(2), np.full(2, 0.1)))


def test_multi_pgs_checks_the_score_order():
    _, w, scores, y, ld, z = _setup(seed=10)
    ids = [f"PGS{i:06d}" for i in range(w.shape[1])]
    fit = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=10,
                             score_ids=ids,
                             var_y=float(y.var()), tune="none")
    assert np.allclose(fit.multi_pgs(scores, score_ids=ids),
                       fit.multi_pgs(scores))
    with pytest.raises(ValueError, match="different order"):
        fit.multi_pgs(scores, score_ids=ids[::-1])
    with pytest.raises(ValueError, match="columns"):
        fit.multi_pgs(scores[:, :3])


def test_input_validation():
    _, w, _, _, ld, z = _setup(seed=11, m=60, k=4)
    with pytest.raises(ValueError, match="must be square"):
        multi_pgs_sumstats(w, z, ld[:, :5], weights_gwas=w, tune="none")
    with pytest.raises(ValueError, match="dense weights have|z covers"):
        multi_pgs_sumstats(w, z[:10], ld, weights_gwas=w, tune="none")
    with pytest.raises(ValueError, match="non-finite"):
        multi_pgs_sumstats(w, np.full(z.size, np.nan), ld, weights_gwas=w,
                           tune="none")
    with pytest.raises(ValueError, match="score_ids has"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w,
                           score_ids=["a", "b"], tune="none",
                           var_y=10.0)
    with pytest.raises(ValueError, match="unique"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w,
                           score_ids=["a", "a", "b", "c"], tune="none",
                           var_y=10.0)
    with pytest.raises(ValueError, match="penalty_factor has"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w,
                           penalty_factor=[1.0, 1.0], tune="none",
                           var_y=10.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w,
                           penalty_factor=[1.0, -1.0, 1.0, 1.0],
                           tune="none", var_y=10.0)
    with pytest.raises(ValueError, match="contiguous"):
        score_gram(w, [(ld[:2, :2], np.array([0, 5]))])
    with pytest.raises(ValueError, match="cover"):
        score_gram(w, [(ld[:2, :2], np.array([0, 1]))])
    with pytest.raises(ValueError, match="overlaps"):
        score_gram(np.eye(4), [(np.eye(2), np.array([0, 1])),
                               (np.eye(2), np.array([0, 1]))])
    with pytest.raises(ValueError, match="gap"):
        score_gram(np.eye(4), [(np.eye(2), np.array([0, 1])),
                               (np.eye(1), np.array([3]))])
    with pytest.raises(ValueError, match="non-finite"):
        score_gram(np.array([[np.nan]]), np.eye(1))


def test_align_to_reference_converts_catalog_weights_to_the_standardized_scale(
        tmp_path):
    from multipgs import simulate_target
    from multipgs.catalog import read_scoring_file

    target = simulate_target(str(tmp_path / "sim"), n=80, n_variants=150,
                             n_scores=3, seed=12)
    from ldpred3.genotype_io import read_bim
    variants = read_bim(target["prefix"] + ".bim")
    n_variants = len(np.asarray(variants.id).ravel())
    empirical_sd = np.full(n_variants, 0.7)

    pairs, ids, log = align_to_reference(target["scoring_files"], variants,
                                         sd=empirical_sd)
    assert len(pairs) == 3 and len(ids) == 3
    assert log["standardized"] is True
    assert log["scale_source"] == "empirical_sd"
    assert "warning" not in log

    raw, _, raw_log = align_to_reference(target["scoring_files"], variants)
    assert raw_log["standardized"] is False
    assert "warning" in raw_log
    # The empirical dosage SD is the exact multiplier used by the LD reference.
    first_scoring = read_scoring_file(target["scoring_files"][0])
    assert len(first_scoring) > 0
    assert np.allclose(pairs[0][1], raw[0][1] * 0.7)

    af = np.full(n_variants, 0.3)
    with pytest.raises(ValueError, match="HWE approximation explicitly"):
        align_to_reference(target["scoring_files"], variants, af=af)
    hwe, _, hwe_log = align_to_reference(
        target["scoring_files"], variants, af=af, hwe_genotype_sd=True)
    hwe_sd = float(np.sqrt(2 * 0.3 * 0.7))
    assert np.allclose(hwe[0][1], raw[0][1] * hwe_sd)
    assert hwe_log["scale_source"] == "hwe_from_af"
    assert "approximation" in hwe_log["warning"]

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        align_to_reference(target["scoring_files"], variants,
                           af=np.full(n_variants, 1.1),
                           hwe_genotype_sd=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        bad_sd = empirical_sd.copy()
        bad_sd[0] = np.nan
        align_to_reference(target["scoring_files"], variants, sd=bad_sd)


# ---------------------------------------------------------------------------
# Evaluation from an independent GWAS
# ---------------------------------------------------------------------------

def _two_samples(seed, n_fit=3000, n_eval=4000, m=200, k=12):
    """A fitting sample and an independent evaluation sample, same architecture."""
    rng = np.random.default_rng(seed)

    def geno(n):
        g = rng.normal(size=(n, m))
        return (g - g.mean(0)) / g.std(0)

    w = np.zeros((m, k))
    for j in range(k):
        idx = rng.choice(m, size=int(rng.integers(5, 30)), replace=False)
        w[idx, j] = rng.normal(size=idx.size)
    truth = np.zeros(k)
    truth[[1, 4]] = [1.0, -0.6]
    w_true = w @ truth

    def sample(n):
        x = geno(n)
        g = x @ w_true
        y = g / g.std() * np.sqrt(0.4) + rng.normal(size=n) * np.sqrt(0.6)
        return x, y - y.mean()

    return w, sample(n_fit), sample(n_eval)


def test_evaluation_identities_are_exact_against_individual_level_truth():
    """R2 and MSE from moments must equal the realized values, not approximate."""
    w, (x_f, y_f), (x_e, y_e) = _two_samples(20)
    z_f = x_f.T @ y_f / x_f.shape[0]
    z_e = x_e.T @ y_e / x_e.shape[0]
    ld_f = x_f.T @ x_f / x_f.shape[0]
    ld_e = x_e.T @ x_e / x_e.shape[0]

    fit = multi_pgs_sumstats(w, z_f, ld_f, weights_gwas=w, n_lambda=30,
                             var_y=float(y_f.var()), tune="none")
    c_e, gram_e, _ = score_moments(w, z_e, ld_e, weights_gwas=w)
    result = fit.evaluate(c_e, gram_e, var_y=float(y_e.var()), regime="A")

    predicted = x_e @ fit.variant_weights(w)
    assert result.r2 == pytest.approx(
        np.corrcoef(predicted, y_e)[0, 1] ** 2, abs=1e-10)
    assert result.mse == pytest.approx(
        float(np.mean((y_e - predicted) ** 2)), abs=1e-10)


def test_regime_c_is_detected_and_a_versus_b_must_be_declared():
    """C has an observable signature; A and B do not, so they must be stated.

    Defaulting "not C" to A is the bug this pins: a PUMAS pseudo-validation
    split also differs from its pseudo-training split, so that default would
    stamp "clean external validation" on a regime B number.
    """
    w, (x_f, y_f), (x_e, y_e) = _two_samples(21)
    z_f = x_f.T @ y_f / x_f.shape[0]
    z_e = x_e.T @ y_e / x_e.shape[0]
    ld_f = x_f.T @ x_f / x_f.shape[0]
    ld_e = x_e.T @ x_e / x_e.shape[0]
    fit = multi_pgs_sumstats(w, z_f, ld_f, weights_gwas=w, n_lambda=20,
                             var_y=float(y_f.var()), tune="none")

    internal = fit.evaluate(*score_moments(
                                w, z_f, ld_f, weights_gwas=w)[:2],
                            var_y=float(y_f.var()))
    assert internal.regime == "C" and not internal.is_validation
    assert "upper bound" in internal.log["warning"]
    assert "not a validation" in internal.summary()

    with pytest.raises(ValueError, match="cannot infer the evaluation regime"):
        fit.evaluate(*score_moments(
                         w, z_e, ld_e, weights_gwas=w)[:2],
                     var_y=float(y_e.var()))

    external = fit.evaluate(*score_moments(
                                w, z_e, ld_e, weights_gwas=w)[:2],
                            var_y=float(y_e.var()), regime="A")
    assert external.regime == "A" and external.is_validation
    pseudo = fit.evaluate(*score_moments(
                              w, z_e, ld_e, weights_gwas=w)[:2],
                          var_y=float(y_e.var()), regime="B")
    assert pseudo.regime == "B" and not pseudo.is_validation
    assert not pseudo.is_assessment
    assert pseudo.r2 == external.r2      # same number, different provenance


def test_every_regime_is_documented():
    assert set(REGIMES) == {"A", "B", "C"}
    for text in REGIMES.values():
        assert text and isinstance(text, str)


def test_evaluate_rejects_mismatched_shapes_and_bad_var_y():
    beta = np.ones(3)
    c = np.full(3, 0.2)
    gram = np.eye(3)
    assert evaluate_sumstat(beta, c, gram, regime="A").r2 == pytest.approx(0.12)
    with pytest.raises(ValueError, match="beta is length 3"):
        evaluate_sumstat(beta, c[:2], gram, regime="A")
    with pytest.raises(ValueError, match="beta is length 3"):
        evaluate_sumstat(beta, c, np.eye(2), regime="A")
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="var_y must be"):
            evaluate_sumstat(beta, c, gram, var_y=bad, regime="A")
    with pytest.raises(ValueError, match="regime must be one of"):
        evaluate_sumstat(beta, c, gram, regime="D")
    with pytest.raises(ValueError, match="indefinite"):
        evaluate_sumstat(np.array([0.0, 1.0]), np.ones(2),
                         np.array([[1.0, 0.0], [0.0, -1.0]]), regime="A")


def test_an_r2_above_one_is_reported_and_a_useless_model_is_flagged():
    noisy = evaluate_sumstat(np.ones(2), np.array([5.0, 5.0]), np.eye(2),
                             regime="A")
    assert noisy.r2 > 1.0 and "above 1" in noisy.log["moment_warning"]
    useless = evaluate_sumstat(np.array([1.0, 0.0]), np.array([0.01, 0.0]),
                               np.eye(2), var_y=1.0, regime="A")
    assert "worse than the mean" in useless.log["mse_warning"]


def test_score_moments_agrees_with_the_fit_it_feeds():
    w, (x_f, y_f), _ = _two_samples(22)
    z_f = x_f.T @ y_f / x_f.shape[0]
    ld_f = x_f.T @ x_f / x_f.shape[0]
    c, gram, var = score_moments(w, z_f, ld_f, weights_gwas=w)
    fit = multi_pgs_sumstats(w, z_f, ld_f, weights_gwas=w, n_lambda=10,
                             var_y=float(y_f.var()), tune="none")
    assert np.allclose(c, fit.c_raw)
    assert np.allclose(np.sqrt(var), fit.score_sd)
    assert np.array_equal(fit.raw_beta, fit.beta)
    assert np.allclose(fit.beta_std, fit.beta * fit.score_sd)


# ---------------------------------------------------------------------------
# Score-space PUMAS
# ---------------------------------------------------------------------------

def test_the_split_reproduces_the_conditional_moments_of_a_real_split():
    """Cov(c_tr|c) = (1/n_tr - 1/n) V_S, and the cross-term is -V_S/n.

    These are the two factors the derivation warns are easy to get wrong; the
    mis-scalings are checked below to make sure the test can tell the
    difference.
    """
    rng = np.random.default_rng(0)
    k, n, var_y = 4, 50_000.0, 1.7
    a = rng.normal(size=(k, k))
    gram = a @ a.T + np.eye(k)
    # A realistic signal level. The rank-one term of V_S contributes in
    # proportion to each score's implied R2, so at c ~ 0 it is genuinely
    # negligible and a test built there could not tell whether it was kept.
    c = np.sqrt(0.3 * var_y * np.diag(gram)) * rng.choice([-1.0, 1.0], size=k)
    v_s = var_y * gram + np.outer(c, c)
    n_train = 0.7 * n
    kappa = 1.0 / n_train - 1.0 / n

    draws = np.array([subsample_score_moments(c, gram, n, n_train,
                                              var_y=var_y, rng=r, check=False)
                      [:2] for r in range(6000)])
    train, val = draws[:, 0, :], draws[:, 1, :]

    def rel(x, y):
        return np.abs(x - y).max() / np.abs(y).max()

    assert rel(np.cov(train.T), kappa * v_s) < 0.08
    assert rel(np.cov(val.T), (1 / (n - n_train) - 1 / n) * v_s) < 0.08
    cross = np.cov(np.hstack([train, val]).T)[:k, k:]
    assert rel(cross, -v_s / n) < 0.08
    assert np.abs(train.mean(0) - c).max() < 0.02 * np.abs(c).max() + 1e-4

    # The two mis-scalings the derivation rules out must be distinguishable.
    assert rel(np.cov(train.T), (1.0 / n_train) * v_s) > 0.3
    assert rel(np.cov(train.T), kappa * var_y * gram) > 0.1


def test_the_validation_half_is_the_exact_deterministic_complement():
    """n_tr c_tr + n_val c_val = n c, exactly — c_val is never drawn."""
    rng = np.random.default_rng(1)
    k, n, n_train = 6, 20_000.0, 15_000.0
    a = rng.normal(size=(k, k))
    gram = a @ a.T + np.eye(k)
    c = rng.normal(size=k) * 0.02
    for seed in range(5):
        train, val, _ = subsample_score_moments(c, gram, n, n_train, var_y=1.0,
                                                rng=seed, check=False)
        assert np.allclose(n_train * train + (n - n_train) * val, n * c)


def test_pumas_selects_a_more_penalized_model_than_in_sample_selection():
    """The point of the whole exercise: in-sample selection does not select."""
    rng = np.random.default_rng(4)
    m, k, n_ref, n_gwas = 400, 60, 2000, 20_000

    def geno(n):
        g = rng.normal(size=(n, m))
        return (g - g.mean(0)) / g.std(0)

    w = np.zeros((m, k))
    for j in range(k):
        idx = rng.choice(m, size=int(rng.integers(10, 50)), replace=False)
        w[idx, j] = rng.normal(size=idx.size)
    truth = np.zeros(k)
    truth[[0, 3]] = [1.0, 0.6]
    w_true = w @ truth

    ld = geno(n_ref)
    ld = ld.T @ ld / n_ref
    x = geno(n_gwas)
    g = x @ w_true
    y = g / g.std() * np.sqrt(0.3) + rng.normal(size=n_gwas) * np.sqrt(0.7)
    y -= y.mean()
    y /= y.std()
    z = x.T @ y / n_gwas

    plain = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=40,
                               var_y=1.0,
                               tune="none")
    pumas = multi_pgs_sumstats(w, z, ld, weights_gwas=w, n_lambda=40,
                               var_y=1.0, tune="pumas",
                               n_eff=n_gwas, n_repeats=6, rng=0,
                               weights_independent_of_z=True)

    assert plain.lambda_index == plain.lambdas.size - 1   # degenerate
    assert pumas.lambda_index < plain.lambda_index
    assert pumas.n_selected < plain.n_selected
    assert pumas.log["selection"] == "PUMAS pseudo-split"
    assert pumas.log["regime"] == "B"
    assert pumas.log["n_repeats"] == 6
    assert "warning" not in pumas.log


def test_pumas_requires_the_effective_sample_size():
    _, w, _, _, ld, z = _setup(seed=12)
    with pytest.raises(ValueError, match="needs n_eff"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w, tune="pumas")
    with pytest.raises(ValueError, match="tune must be"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w, tune="nonsense")
    with pytest.raises(ValueError, match="train_fraction"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w, tune="pumas", n_eff=1000,
                           train_fraction=1.5, weights_independent_of_z=True)
    with pytest.raises(ValueError, match="n_repeats"):
        multi_pgs_sumstats(w, z, ld, weights_gwas=w, tune="pumas", n_eff=1000,
                           n_repeats=0,
                           weights_independent_of_z=True)


def test_an_impossible_single_score_r2_is_logged_as_a_diagnostic():
    """A noisy estimate may exceed one, but the scale warning remains useful."""
    gram = np.eye(3)
    c = np.array([0.1, 2.0, 0.1])          # score 1 implies R2 = 4
    _, _, noisy_log = subsample_score_moments(
        c, gram, 10_000.0, 7_000.0, var_y=1.0)
    assert "single-score R2 above 1" in noisy_log["warning"]
    # The same c is fine once var_y matches the scale it is on.
    _, _, log = subsample_score_moments(c, gram, 10_000.0, 7_000.0, var_y=9.0)
    assert log["max_implied_r2"] < 1.0


def test_a_rank_deficient_panel_is_reported_not_silently_accepted():
    """A duplicated score leaves a direction the split cannot penalize."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(5, 5))
    gram = base @ base.T
    gram = np.block([[gram, gram], [gram, gram]])     # rank 5, size 10
    c = np.concatenate([rng.normal(size=5) * 0.01] * 2)
    _, _, log = subsample_score_moments(c, gram, 10_000.0, 7_000.0, var_y=1.0)
    assert log["rank"] == 5 and log["n_scores"] == 10
    assert "var_y * G noise term is zero" in log["rank_warning"]


def test_subsample_validates_its_inputs():
    gram, c = np.eye(3), np.zeros(3)
    with pytest.raises(ValueError, match="gram must be"):
        subsample_score_moments(c, np.eye(2), 1000.0, 700.0, var_y=1.0)
    with pytest.raises(ValueError, match="n must be"):
        subsample_score_moments(c, gram, -1.0, 700.0, var_y=1.0)
    for bad in (0.0, 1000.0, 1500.0):
        with pytest.raises(ValueError, match="n_train must satisfy"):
            subsample_score_moments(c, gram, 1000.0, bad, var_y=1.0)
    with pytest.raises(ValueError, match="var_y must be"):
        subsample_score_moments(c, gram, 1000.0, 700.0, var_y=0.0)


# ---------------------------------------------------------------------------
# Low-rank LD blocks
#
# ldpred3 stores a large LD block as a factor, R = U U' + diag(residual), and
# score_gram reads that factor directly rather than asking ld_matmul to project
# back up to the block's variant dimension. Nothing else in this file builds
# such a block, so without these the fast path ships untested — and it is the
# path a real genome-wide reference spends most of its time in.
# ---------------------------------------------------------------------------

def _lowrank_factor(rng, m, rank):
    """A factor whose row norms leave a positive residual under a unit diagonal.

    ``LowRankLD`` describes a *correlation* block, so it requires
    ``diag(U U') + residual_diag == 1``. Each row's factor mass is drawn below
    one and the remainder becomes its residual.
    """
    factor = rng.standard_normal((m, rank))
    explained = rng.uniform(0.55, 0.9, size=m)
    factor *= np.sqrt(explained / np.sum(factor * factor, axis=1))[:, None]
    return factor, 1.0 - explained


def _lowrank_block(rng, m, rank):
    """A LowRankLD block and the dense correlation matrix it stands for."""
    from ldpred3 import LowRankLD

    factor, residual = _lowrank_factor(rng, m, rank)
    dense = factor @ factor.T + np.diag(residual)
    return LowRankLD(U=factor, m=m, scale=1.0, residual_diag=residual), dense


def test_low_rank_blocks_give_the_same_gram_as_their_dense_form():
    rng = np.random.default_rng(11)
    m, k = 60, 8
    block, dense = _lowrank_block(rng, m, rank=9)
    weights = [(np.sort(rng.choice(m, 20, replace=False)),
                rng.standard_normal(20)) for _ in range(k)]

    from_factor, var_factor = score_gram([(idx, w) for idx, w in weights],
                                         [(block, np.arange(m))],
                                         n_variants=m)
    from_dense, var_dense = score_gram([(idx, w) for idx, w in weights],
                                       [(dense, np.arange(m))], n_variants=m)
    assert np.allclose(from_factor, from_dense, atol=1e-10)
    assert np.allclose(var_factor, var_dense, atol=1e-10)
    # And against the definition, not just against the other route.
    explicit = np.zeros((k, k))
    dense_w = np.zeros((m, k))
    for j, (idx, w) in enumerate(weights):
        dense_w[idx, j] = w
    explicit = dense_w.T @ dense @ dense_w
    assert np.allclose(from_factor, explicit, atol=1e-10)


def test_low_rank_and_dense_blocks_mix_in_one_reference():
    """A real reference stores small blocks densely and large ones as factors."""
    rng = np.random.default_rng(12)
    m_lr, m_dense, k = 50, 20, 6
    block, lr_dense = _lowrank_block(rng, m_lr, rank=7)
    small = rng.standard_normal((m_dense, m_dense))
    small = small @ small.T / m_dense + np.eye(m_dense)
    sd = np.sqrt(np.diag(small))
    small = small / np.outer(sd, sd)

    total = m_lr + m_dense
    weights = [(np.sort(rng.choice(total, 15, replace=False)),
                rng.standard_normal(15)) for _ in range(k)]
    mixed = [(block, np.arange(m_lr)),
             (small, np.arange(m_lr, total))]
    all_dense = [(lr_dense, np.arange(m_lr)),
                 (small, np.arange(m_lr, total))]
    assert np.allclose(score_gram(weights, mixed, n_variants=total)[0],
                       score_gram(weights, all_dense, n_variants=total)[0],
                       atol=1e-10)


def test_int8_low_rank_blocks_are_dequantized_before_use():
    """An int8 factor read at face value would inflate the Gram enormously."""
    from ldpred3 import LowRankLD
    from ldpred3.ld_repr import dequantize_ld

    rng = np.random.default_rng(13)
    m, rank, k = 40, 6, 5
    factor, _ = _lowrank_factor(rng, m, rank)
    quantized = np.clip(np.round(factor * 127.0), -127, 127).astype(np.int8)
    # Rounding moves the row norms, and LowRankLD checks the unit diagonal on
    # the stored factor, so take the residual from the quantized rows.
    stored = np.asarray(quantized, dtype=float) / 127.0
    residual = 1.0 - np.sum(stored * stored, axis=1)
    block = LowRankLD(U=quantized, m=m, scale=1.0 / 127.0,
                      residual_diag=residual)
    reference = dequantize_ld(block)

    weights = [(np.sort(rng.choice(m, 12, replace=False)),
                rng.standard_normal(12)) for _ in range(k)]
    gram, _ = score_gram(weights, [(block, np.arange(m))], n_variants=m)
    expected_dense = (np.asarray(reference.U, dtype=float)
                      @ np.asarray(reference.U, dtype=float).T
                      + np.diag(np.asarray(reference.residual_diag,
                                           dtype=float)))
    expected, _ = score_gram(weights, [(expected_dense, np.arange(m))],
                             n_variants=m)
    assert np.allclose(gram, expected, atol=1e-10)
    # The failure this guards against is silent and enormous, not subtle.
    naive = np.asarray(quantized, dtype=float)
    naive = naive @ naive.T + np.diag(residual)
    wrong, _ = score_gram(weights, [(naive, np.arange(m))], n_variants=m)
    assert np.abs(wrong).max() > 100.0 * np.abs(gram).max()
