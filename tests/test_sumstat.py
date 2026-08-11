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


def test_it_reproduces_the_individual_level_fit_exactly():
    """Same LD, same data: this is not an approximation of the stacked fit."""
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

    fit = multi_pgs_sumstats(w, z, ld, n_lambda=30, alpha=1.0)
    assert np.allclose(fit.score_sd, sd, atol=1e-10)
    assert np.allclose(fit.lambdas, lambdas[:fit.lambdas.size], atol=1e-10)
    assert np.allclose(fit.path, expected[:fit.path.shape[0]], atol=1e-10)


def test_variant_weights_reproduce_the_combined_score():
    x, w, scores, _, ld, z = _setup(seed=4)
    fit = multi_pgs_sumstats(w, z, ld, n_lambda=20)
    assert np.allclose(x @ fit.variant_weights(w), fit.multi_pgs(scores),
                       atol=1e-10)


def test_pseudo_r2_is_the_realized_r2_when_ld_is_in_sample():
    x, w, scores, y, ld, z = _setup(seed=5)
    fit = multi_pgs_sumstats(w, z, ld, n_lambda=20, var_y=float(y.var()))
    predicted = x @ fit.variant_weights(w)
    realized = np.corrcoef(predicted, y)[0, 1] ** 2
    assert fit.pseudo_r2 == pytest.approx(realized, abs=1e-8)


def test_in_sample_selection_is_labelled_and_an_independent_gwas_is_too():
    _, w, _, _, ld, z = _setup(seed=6)
    plain = multi_pgs_sumstats(w, z, ld, n_lambda=15)
    assert plain.log["selection"] == "in-sample"
    assert "optimistic" in plain.log["warning"]
    assert "optimistic" in plain.summary()

    honest = multi_pgs_sumstats(w, z, ld, n_lambda=15, z_valid=z * 0.9)
    assert honest.log["selection"] == "independent GWAS"
    assert "warning" not in honest.log


def test_a_score_with_no_variance_is_dropped_not_nan():
    _, w, _, _, ld, z = _setup(seed=7)
    w = w.copy()
    w[:, 2] = 0.0
    fit = multi_pgs_sumstats(w, z, ld, n_lambda=15)
    assert fit.log["n_dead"] == 1
    assert fit.beta[2] == 0.0
    assert np.all(np.isfinite(fit.beta))
    assert np.all(np.isfinite(fit.variant_weights(w)))


def test_an_r2_above_one_is_flagged_as_a_scale_error():
    """var_y left at 1 for an unstandardized y is the mistake this catches."""
    _, w, _, y, ld, z = _setup(seed=8)
    assert y.var() > 2.0
    fit = multi_pgs_sumstats(w, z, ld, n_lambda=15)
    assert fit.pseudo_r2 > 1.0
    assert "not an R2" in fit.log["scale_warning"]


def test_penalty_factors_can_force_a_score_in():
    _, w, _, _, ld, z = _setup(seed=9)
    pf = np.ones(w.shape[1])
    pf[7] = 0.0
    fit = multi_pgs_sumstats(w, z, ld, n_lambda=5, penalty_factor=pf)
    assert fit.beta[7] != 0.0


def test_pseudo_r2_rejects_a_non_psd_reference():
    gram = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="not\n?\\s*positive semi-definite|"
                                         "positive semi-definite"):
        pseudo_r2(np.array([0.0, 1.0]), gram, np.array([1.0, 1.0]))
    assert np.isnan(pseudo_r2(np.zeros(2), np.eye(2), np.ones(2)))


def test_multi_pgs_checks_the_score_order():
    _, w, scores, _, ld, z = _setup(seed=10)
    ids = [f"PGS{i:06d}" for i in range(w.shape[1])]
    fit = multi_pgs_sumstats(w, z, ld, n_lambda=10, score_ids=ids)
    assert np.allclose(fit.multi_pgs(scores, score_ids=ids),
                       fit.multi_pgs(scores))
    with pytest.raises(ValueError, match="different order"):
        fit.multi_pgs(scores, score_ids=ids[::-1])
    with pytest.raises(ValueError, match="columns"):
        fit.multi_pgs(scores[:, :3])


def test_input_validation():
    _, w, _, _, ld, z = _setup(seed=11, m=60, k=4)
    with pytest.raises(ValueError, match="must be square"):
        multi_pgs_sumstats(w, z, ld[:, :5])
    with pytest.raises(ValueError, match="z covers"):
        multi_pgs_sumstats(w, z[:10], ld)
    with pytest.raises(ValueError, match="non-finite"):
        multi_pgs_sumstats(w, np.full(z.size, np.nan), ld)
    with pytest.raises(ValueError, match="score_ids has"):
        multi_pgs_sumstats(w, z, ld, score_ids=["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        multi_pgs_sumstats(w, z, ld, score_ids=["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="penalty_factor has"):
        multi_pgs_sumstats(w, z, ld, penalty_factor=[1.0, 1.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        multi_pgs_sumstats(w, z, ld, penalty_factor=[1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="contiguous"):
        score_gram(w, [(ld[:2, :2], np.array([0, 5]))])
    with pytest.raises(ValueError, match="cover"):
        score_gram(w, [(ld[:2, :2], np.array([0, 1]))])


def test_align_to_reference_converts_catalog_weights_to_the_standardized_scale(
        tmp_path):
    from multipgs import simulate_target
    from multipgs.catalog import read_scoring_file

    target = simulate_target(str(tmp_path / "sim"), n=80, n_variants=150,
                             n_scores=3, seed=12)
    from ldpred3.genotype_io import read_bim
    variants = read_bim(target["prefix"] + ".bim")
    af = np.full(len(np.asarray(variants.id).ravel()), 0.3)

    pairs, ids, log = align_to_reference(target["scoring_files"], variants,
                                         af=af)
    assert len(pairs) == 3 and len(ids) == 3
    assert log["standardized"] is True
    assert "warning" not in log

    raw, _, raw_log = align_to_reference(target["scoring_files"], variants)
    assert raw_log["standardized"] is False
    assert "warning" in raw_log
    # sqrt(2 f (1-f)) at f = 0.3 is a constant here, so the two differ by it.
    sd = float(np.sqrt(2 * 0.3 * 0.7))
    first_scoring = read_scoring_file(target["scoring_files"][0])
    assert len(first_scoring) > 0
    assert np.allclose(pairs[0][1], raw[0][1] * sd)


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

    fit = multi_pgs_sumstats(w, z_f, ld_f, n_lambda=30, var_y=float(y_f.var()))
    c_e, gram_e, _ = score_moments(w, z_e, ld_e)
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
    fit = multi_pgs_sumstats(w, z_f, ld_f, n_lambda=20, var_y=float(y_f.var()))

    internal = fit.evaluate(*score_moments(w, z_f, ld_f)[:2],
                            var_y=float(y_f.var()))
    assert internal.regime == "C" and not internal.is_validation
    assert "upper bound" in internal.log["warning"]
    assert "not a validation" in internal.summary()

    with pytest.raises(ValueError, match="cannot infer the evaluation regime"):
        fit.evaluate(*score_moments(w, z_e, ld_e)[:2], var_y=float(y_e.var()))

    external = fit.evaluate(*score_moments(w, z_e, ld_e)[:2],
                            var_y=float(y_e.var()), regime="A")
    assert external.regime == "A" and external.is_validation
    pseudo = fit.evaluate(*score_moments(w, z_e, ld_e)[:2],
                          var_y=float(y_e.var()), regime="B")
    assert pseudo.regime == "B" and pseudo.is_validation
    assert pseudo.r2 == external.r2      # same number, different provenance


def test_every_regime_is_documented():
    assert set(REGIMES) == {"A", "B", "C"}
    for text in REGIMES.values():
        assert text and isinstance(text, str)


def test_evaluate_rejects_mismatched_shapes_and_bad_var_y():
    beta = np.ones(3)
    c = np.ones(3)
    gram = np.eye(3)
    assert evaluate_sumstat(beta, c, gram, regime="A").r2 == pytest.approx(3.0)
    with pytest.raises(ValueError, match="beta is length 3"):
        evaluate_sumstat(beta, c[:2], gram, regime="A")
    with pytest.raises(ValueError, match="beta is length 3"):
        evaluate_sumstat(beta, c, np.eye(2), regime="A")
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="var_y must be"):
            evaluate_sumstat(beta, c, gram, var_y=bad, regime="A")
    with pytest.raises(ValueError, match="regime must be one of"):
        evaluate_sumstat(beta, c, gram, regime="D")
    with pytest.raises(ValueError, match="positive semi-definite"):
        evaluate_sumstat(np.array([0.0, 1.0]), np.ones(2),
                         np.array([[1.0, 0.0], [0.0, -1.0]]), regime="A")


def test_an_r2_above_one_and_a_useless_model_are_both_flagged():
    high = evaluate_sumstat(np.ones(2), np.array([5.0, 5.0]), np.eye(2),
                            regime="A")
    assert "not an R2" in high.log["scale_warning"]
    useless = evaluate_sumstat(np.array([1.0, 0.0]), np.array([0.01, 0.0]),
                               np.eye(2), var_y=1.0, regime="A")
    assert "worse than the mean" in useless.log["mse_warning"]


def test_score_moments_agrees_with_the_fit_it_feeds():
    w, (x_f, y_f), _ = _two_samples(22)
    z_f = x_f.T @ y_f / x_f.shape[0]
    ld_f = x_f.T @ x_f / x_f.shape[0]
    c, gram, var = score_moments(w, z_f, ld_f)
    fit = multi_pgs_sumstats(w, z_f, ld_f, n_lambda=10)
    assert np.allclose(c, fit.c_raw)
    assert np.allclose(np.sqrt(var), fit.score_sd)
    # raw_beta pairs with the raw moments; beta pairs with the scaled ones.
    assert np.allclose(fit.raw_beta * fit.score_sd, fit.beta)


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

    plain = multi_pgs_sumstats(w, z, ld, n_lambda=40, var_y=1.0)
    pumas = multi_pgs_sumstats(w, z, ld, n_lambda=40, var_y=1.0, tune="pumas",
                               n_eff=n_gwas, n_repeats=6, rng=0)

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
        multi_pgs_sumstats(w, z, ld, tune="pumas")
    with pytest.raises(ValueError, match="tune must be"):
        multi_pgs_sumstats(w, z, ld, tune="nonsense")
    with pytest.raises(ValueError, match="train_fraction"):
        multi_pgs_sumstats(w, z, ld, tune="pumas", n_eff=1000,
                           train_fraction=1.5)
    with pytest.raises(ValueError, match="n_repeats"):
        multi_pgs_sumstats(w, z, ld, tune="pumas", n_eff=1000, n_repeats=0)


def test_an_impossible_single_score_r2_is_refused_as_a_provenance_error():
    """c_k^2/(var_y G_kk) > 1 means a scale, var_y, or overlap error."""
    gram = np.eye(3)
    c = np.array([0.1, 2.0, 0.1])          # score 1 implies R2 = 4
    with pytest.raises(ValueError, match="single-score R2 above 1"):
        subsample_score_moments(c, gram, 10_000.0, 7_000.0, var_y=1.0)
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
    assert "overfit there without being penalised" in log["rank_warning"]


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
