"""Training-free combination of same-trait scores."""

import numpy as np
import pytest

from multipgs import daetwyler_r2, meta_pgs, r2, simulate_same_trait_panel


def _expected(sim, h2=0.4, m_causal=5000, n_variants=1_000_000):
    return daetwyler_r2(h2, m_causal / n_variants, sim.n_eff, n_variants)


def test_beats_the_best_single_score_with_no_phenotype():
    sim = simulate_same_trait_panel(n=20_000, seed=1)
    res = meta_pgs(sim.scores, n_eff=sim.n_eff, score_ids=sim.score_ids)
    best_single = max(r2(sim.y, sim.scores[:, k])
                      for k in range(sim.scores.shape[1]))
    assert r2(sim.y, res.multi_pgs(sim.scores)) > best_single
    assert np.all(res.weight > 0)
    assert res.weight[0] > res.weight[-1]        # larger GWAS, larger weight
    assert "sqrt_n_eff" in res.summary()


def test_the_gain_is_large_when_underpowered_and_small_when_not():
    """Pins the (1 - R_f^2)^2 factor: combining helps where scores are weak.

    A fixed margin here would only measure the simulation. The theory predicts
    the *ratio* of gains between the two regimes, so assert that.
    """
    def gain(n_eff):
        sim = simulate_same_trait_panel(n=40_000, n_eff=n_eff, h2=0.4,
                                        m_causal=5_000, seed=2)
        best = max(r2(sim.y, sim.scores[:, k])
                   for k in range(sim.scores.shape[1]))
        combined = r2(sim.y, meta_pgs(sim.scores, n_eff=np.asarray(n_eff))
                      .multi_pgs(sim.scores))
        return (combined - best) / best

    weak = gain((5_000., 3_000., 2_000.))        # x ~ 0.4, 0.24, 0.16
    strong = gain((150_000., 60_000., 20_000.))  # x ~ 12, 4.8, 1.6
    assert weak > 0.25                    # a real gain where power is short
    assert strong < 0.05                  # almost none where it is not
    assert weak > 8 * strong


def test_bigger_discovery_gwas_gets_more_weight():
    sim = simulate_same_trait_panel(n=2000, n_eff=(200_000, 50_000, 10_000),
                                    seed=2)
    res = meta_pgs(sim.scores, n_eff=sim.n_eff)
    assert np.all(np.diff(res.weight) < 0)
    # sqrt(n_eff), normalised: the ratio of weights is the ratio of sqrt(N).
    ratio = res.weight[0] / res.weight[2]
    assert ratio == pytest.approx(np.sqrt(200_000 / 10_000), rel=1e-6)


def test_decorrelation_helps_when_discovery_cohorts_overlap():
    """With overlap, C^-1 rho must beat the plain weighted sum."""
    sim = simulate_same_trait_panel(n=20_000, shared=0.8, seed=1)
    er2 = _expected(sim)
    plain = r2(sim.y, meta_pgs(sim.scores, n_eff=sim.n_eff).multi_pgs(
        sim.scores))
    decorr = r2(sim.y, meta_pgs(sim.scores, expected_r2=er2,
                                method="decorrelated").multi_pgs(sim.scores))
    best_single = max(r2(sim.y, sim.scores[:, k])
                      for k in range(sim.scores.shape[1]))
    assert decorr > plain
    assert decorr > best_single


def test_expected_r2_weighting_runs_and_ranks_sensibly():
    sim = simulate_same_trait_panel(n=5000, seed=3)
    res = meta_pgs(sim.scores, expected_r2=_expected(sim),
                   method="expected_r2")
    assert np.all(np.diff(res.weight) < 0)
    assert np.isclose(np.linalg.norm(res.weight), 1.0)


def test_frozen_standardization_scores_a_second_cohort_consistently():
    sim = simulate_same_trait_panel(n=4000, seed=4)
    tr, te = slice(0, 2000), slice(2000, None)
    fit = meta_pgs(sim.scores[tr], n_eff=sim.n_eff)
    same = meta_pgs(sim.scores[te], n_eff=sim.n_eff, center=fit.center,
                    scale=fit.scale)
    assert np.allclose(fit.beta, same.beta)
    assert np.allclose(fit.multi_pgs(sim.scores[te]),
                       same.multi_pgs(sim.scores[te]))


def test_constant_score_gets_zero_weight():
    sim = simulate_same_trait_panel(n=1000, n_eff=(100_000, 50_000, 20_000),
                                    seed=5)
    scores = sim.scores.copy()
    scores[:, 1] = 7.0
    res = meta_pgs(scores, n_eff=sim.n_eff)
    assert res.beta[1] == 0.0
    assert res.log["dead_scores"] == 1
    assert np.all(np.isfinite(res.multi_pgs(scores)))


def test_accepts_a_score_panel_directly(tmp_path):
    from multipgs import panel_from_catalog, simulate_target
    target = simulate_target(str(tmp_path / "sim"), n=120, n_variants=200,
                             n_scores=3, seed=6)
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    res = meta_pgs(panel, n_eff=[100_000, 50_000, 20_000])
    assert list(res.score_ids) == list(panel.score_ids)
    assert np.allclose(res.multi_pgs(panel), res.multi_pgs(panel.scores))
    with pytest.raises(ValueError, match="different order"):
        res.multi_pgs(panel.select([2, 1, 0]))
    with pytest.raises(ValueError, match="not the ones"):
        res.multi_pgs(panel.scores, score_ids=["a", "b", "c"])
    with pytest.raises(ValueError, match="do not match"):
        meta_pgs(panel, n_eff=[100_000, 50_000, 20_000],
                 score_ids=panel.score_ids[::-1])


def test_decorrelated_accepts_one_score():
    scores = np.arange(10.0)[:, None]
    res = meta_pgs(scores, expected_r2=[0.1], method="decorrelated")
    assert res.weight == pytest.approx([1.0])
    assert np.all(np.isfinite(res.multi_pgs(scores)))
    assert res.log["condition_number"] == pytest.approx(1.0)


def test_input_validation():
    sim = simulate_same_trait_panel(n=200, seed=7)
    with pytest.raises(ValueError, match="method must be"):
        meta_pgs(sim.scores, n_eff=sim.n_eff, method="nope")
    with pytest.raises(ValueError, match="needs n_eff"):
        meta_pgs(sim.scores)
    with pytest.raises(ValueError, match="needs expected_r2"):
        meta_pgs(sim.scores, method="expected_r2")
    with pytest.raises(ValueError, match="3 scores"):
        meta_pgs(sim.scores, n_eff=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite and positive"):
        meta_pgs(sim.scores, n_eff=[1.0, -2.0, 3.0])
    for bad_expected_r2 in ([0.1, 1.01, 0.2], [0.1, -0.01, 0.2],
                            [0.1, np.nan, 0.2]):
        with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
            meta_pgs(sim.scores, expected_r2=bad_expected_r2,
                     method="expected_r2")
    with pytest.raises(ValueError, match="non-finite"):
        meta_pgs(np.full((10, 3), np.nan), n_eff=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="unique"):
        meta_pgs(sim.scores, n_eff=sim.n_eff, score_ids=["a", "a", "b"])
    with pytest.raises(ValueError, match="center must have shape"):
        meta_pgs(sim.scores, n_eff=sim.n_eff, center=[0.0, 0.0])
    with pytest.raises(ValueError, match="center.*finite"):
        meta_pgs(sim.scores, n_eff=sim.n_eff,
                 center=[0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="scale must have shape"):
        meta_pgs(sim.scores, n_eff=sim.n_eff, scale=[1.0, 1.0])
    with pytest.raises(ValueError, match="scale.*finite"):
        meta_pgs(sim.scores, n_eff=sim.n_eff,
                 scale=[1.0, np.inf, 1.0])
    with pytest.raises(ValueError, match="scale must be non-negative"):
        meta_pgs(sim.scores, n_eff=sim.n_eff, scale=[1.0, -1.0, 1.0])
    for bad_ridge in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError,
                           match="ridge must be finite and non-negative"):
            meta_pgs(sim.scores, n_eff=sim.n_eff, ridge=bad_ridge)


def test_multi_pgs_checks_the_column_count():
    sim = simulate_same_trait_panel(n=200, seed=8)
    res = meta_pgs(sim.scores, n_eff=sim.n_eff)
    with pytest.raises(ValueError, match="3 columns"):
        res.multi_pgs(sim.scores[:, :2])
    with pytest.raises(ValueError, match="3 columns"):
        res.multi_pgs(np.zeros((2, 3, 1)))


def test_decorrelated_records_but_does_not_police_rho_alignment():
    """The alignment is descriptive; it must not be sold as a failure detector.

    On the real panel that motivated this, the configuration scoring R2 0.00001
    had alignment 0.67 while one scoring three hundred times better had 0.40, so
    a threshold would have reassured the caller in exactly the worst case. This
    pins the value as logged and unpoliced, so the temptation is not retried.
    """
    rng = np.random.default_rng(0)
    n, k = 4000, 6
    shared = rng.standard_normal(n)
    columns = [shared + 0.02 * rng.standard_normal(n) for _ in range(k - 1)]
    columns.append(rng.standard_normal(n))
    duplicated = meta_pgs(np.column_stack(columns), n_eff=np.full(k, 1e5),
                          method="decorrelated")
    independent = meta_pgs(rng.standard_normal((n, k)), n_eff=np.full(k, 1e5),
                           method="decorrelated")

    for fit in (duplicated, independent):
        assert "rho_alignment" in fit.log
        assert -1.0 <= fit.log["rho_alignment"] <= 1.0
        assert "alignment_warning" not in fit.log

    # It does describe what it claims to: an orthogonal panel needs no
    # correction, a duplicate-heavy one gets a large one.
    assert independent.log["rho_alignment"] > duplicated.log["rho_alignment"]
