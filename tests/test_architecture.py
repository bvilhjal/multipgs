"""Score screening and accuracy-derived penalties."""

import numpy as np
import pytest

from multipgs import (Architecture, architectures_from_panel, daetwyler_r2,
                      penalty_from_accuracy, screen)


def test_daetwyler_matches_the_closed_form():
    h2, p, n, m_total = 0.4, 1e-3, 100_000, 1_000_000
    m = m_total * p                            # 1000 causal variants
    x = n * h2 / m
    assert daetwyler_r2(h2, p, n, m_total) == pytest.approx(h2 * x / (1 + x))


def test_daetwyler_is_monotone_and_bounded_by_h2():
    h2 = 0.5
    r2 = daetwyler_r2(h2, 1e-3, np.array([1e3, 1e4, 1e5, 1e7]), 1e6)
    assert np.all(np.diff(r2) > 0)             # more samples, more accuracy
    assert np.all(r2 < h2)                     # never exceeds heritability
    assert daetwyler_r2(h2, 1e-3, 1e12, 1e6) == pytest.approx(h2, rel=1e-4)


def test_daetwyler_returns_nan_for_impossible_inputs():
    assert np.isnan(daetwyler_r2(0.0, 1e-3, 1e5, 1e6))
    assert np.isnan(daetwyler_r2(0.4, 0.0, 1e5, 1e6))
    assert np.isnan(daetwyler_r2(0.4, 1e-3, np.nan, 1e6))


def _arch(**kw):
    base = dict(score_id="s", h2=0.3, p=1e-3, r2_infer=0.1, n_chains_kept=50,
                n_chains=50, n_variants=1_000_000, n_eff=100_000)
    base.update(kw)
    return Architecture(**base)


def test_screen_passes_a_good_score_and_names_each_failure():
    cases = {
        "ok": (_arch(score_id="ok"), None),
        "low_h2": (_arch(score_id="low_h2", h2=0.001), "heritability below"),
        "high_h2": (_arch(score_id="high_h2", h2=1.5), "heritability above"),
        "chains": (_arch(score_id="chains", n_chains_kept=5), "chains"),
        "variants": (_arch(score_id="variants", n_variants=1000), "variants"),
        "small_n": (_arch(score_id="small_n", n_eff=500), "effective sample"),
    }
    res = screen([a for a, _ in cases.values()])
    assert res.n_kept == 1
    for sid, (_, expect) in cases.items():
        if expect is None:
            assert sid not in res.reasons
        else:
            assert expect in res.reasons[sid]
    assert "kept 1 of 6" in res.summary()


def test_unscreenable_scores_are_kept_by_default_and_flagged():
    archs = [_arch(score_id="fitted"),
             Architecture(score_id="catalog")]          # all-nan
    res = screen(archs)
    assert res.keep.tolist() == [True, True]
    assert res.unscreenable.tolist() == [False, True]
    assert "no architecture" in res.summary()
    dropped = screen(archs, keep_unscreenable=False)
    assert dropped.keep.tolist() == [True, False]
    assert "no architecture available" in dropped.reasons["catalog"]


def test_expected_r2_gate_is_opt_in():
    weak = _arch(score_id="weak", h2=0.02, n_eff=11_000)
    assert screen([weak]).n_kept == 1
    assert screen([weak], min_expected_r2=0.5).n_kept == 0


def test_penalty_factors_have_geometric_mean_one_and_respect_the_clip():
    pf = penalty_from_accuracy([0.2, 0.05, 0.01, 0.001])
    assert np.exp(np.mean(np.log(pf))) == pytest.approx(1.0)
    assert np.all(np.diff(pf) > 0)            # less accurate, more penalty
    assert pf.max() / pf.min() <= 4.0 ** 2 + 1e-9


def test_penalty_treats_missing_accuracy_as_the_worst_case():
    pf = penalty_from_accuracy([0.2, np.nan, -1.0])
    assert pf[1] == pf[2] == pf.max()


def test_penalty_power_zero_is_flat():
    pf = penalty_from_accuracy([0.2, 0.05, 0.001], power=0.0)
    assert np.allclose(pf, 1.0)


def test_penalty_rejects_a_clip_below_one():
    with pytest.raises(ValueError, match="clip must be"):
        penalty_from_accuracy([0.1, 0.2], clip=0.5)


def test_architectures_read_back_from_a_panel():
    from multipgs.panel import ScorePanel

    panel = ScorePanel(
        scores=np.zeros((5, 2)), sample_fid=np.arange(5),
        sample_iid=np.arange(5),
        score_ids=np.array(["a", "b"], dtype=object),
        standardized=np.ones(2, dtype=bool), weights=[],
        meta=[{"n_matched": 900_000,
               "inference": {"h2_est": 0.25, "p_est": 2e-3, "r2_est": 0.08,
                             "n_chains_kept": 40}},
              {"n_matched": 5000}])
    archs = architectures_from_panel(panel, n_eff={"a": 200_000})
    assert archs[0].h2 == 0.25 and archs[0].n_chains_kept == 40
    assert archs[0].n_eff == 200_000
    assert archs[0].expected_r2() > 0
    assert np.isnan(archs[1].h2) and np.isnan(archs[1].n_eff)
    res = screen(archs)
    assert res.keep.tolist() == [True, True]   # second is unscreenable


def test_architectures_from_panel_checks_n_eff_length():
    from multipgs.panel import ScorePanel

    panel = ScorePanel(scores=np.zeros((2, 2)), sample_fid=np.arange(2),
                       sample_iid=np.arange(2),
                       score_ids=np.array(["a", "b"], dtype=object),
                       standardized=np.zeros(2, dtype=bool), weights=[],
                       meta=[{}, {}])
    with pytest.raises(ValueError, match="1 entries for 2 scores"):
        architectures_from_panel(panel, n_eff=[100.0])
