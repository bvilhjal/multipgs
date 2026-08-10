"""Panel construction from genotypes, and folding a fit back to one weight set."""

import numpy as np
import pytest

from multipgs import (combine_weights, multi_pgs_fit, panel_from_catalog,
                      read_panel, simulate_target, write_panel)


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    d = tmp_path_factory.mktemp("target")
    return simulate_target(str(d / "sim"), n=250, n_variants=300, n_scores=4,
                           missing=0.02, seed=5)


def test_panel_reproduces_the_scores_the_weights_imply(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    assert panel.scores.shape == (250, 4)
    assert np.allclose(panel.scores, target["true_scores"], atol=1e-6)
    assert list(panel.sample_iid) == list(target["sample_iid"])
    assert list(panel.score_ids) == ["PGS000001", "PGS000002", "PGS000003",
                                     "PGS000004"]
    assert not panel.standardized.any()
    assert "4 scores" in panel.summary()


def test_block_size_does_not_change_the_answer(target):
    a = panel_from_catalog(target["scoring_files"], target["prefix"])
    b = panel_from_catalog(target["scoring_files"], target["prefix"], block=7)
    assert np.allclose(a.scores, b.scores, atol=1e-9)


def test_a_directory_of_scoring_files_is_accepted(target, tmp_path):
    import os
    import shutil
    d = tmp_path / "scores"
    d.mkdir()
    for p in target["scoring_files"]:
        shutil.copy(p, d / os.path.basename(p))
    panel = panel_from_catalog(str(d), target["prefix"])
    assert panel.n_scores == 4


def test_panel_records_target_allele_frequency_and_sd(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    for table in panel.weights:
        assert np.all(np.isfinite(table["af"]))
        assert np.all(table["sd"] > 0)
        assert np.all((table["af"] >= 0) & (table["af"] <= 1))


def test_min_matched_rejects_a_thin_score(target):
    with pytest.raises(ValueError, match="min_matched"):
        panel_from_catalog(target["scoring_files"], target["prefix"],
                           min_matched=10_000)
    with pytest.raises(ValueError, match="none of the"):
        panel_from_catalog(target["scoring_files"], target["prefix"],
                           min_matched=10_000, on_error="skip")


def test_unreadable_file_can_be_skipped(target, tmp_path):
    bad = tmp_path / "broken.txt"
    bad.write_text("#pgs_id=BAD\nnot_a_header_we_know\n", encoding="utf-8")
    files = list(target["scoring_files"]) + [str(bad)]
    with pytest.raises(ValueError):
        panel_from_catalog(files, target["prefix"])
    panel = panel_from_catalog(files, target["prefix"], on_error="skip")
    assert panel.n_scores == 4
    assert panel.log["n_failed"] == 1


def test_combined_weights_reproduce_the_multi_pgs_exactly(target):
    """The deployable artefact must equal the model it came from."""
    from ldpred3 import score_from_weights

    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    rng = np.random.default_rng(0)
    y = (2 * target["true_scores"][:, 0] - target["true_scores"][:, 2]
         + rng.normal(size=250))
    fit = multi_pgs_fit(panel.scores, y, n_folds=4, n_lambda=30, seed=0,
                        score_ids=panel.score_ids)
    assert fit.n_selected > 0

    import tempfile
    import os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "combined.weights")
        table = combine_weights(panel, fit, path=path)
        assert table["weight"].size > 0
        for scaling in ("frozen", "target"):
            scored = score_from_weights(path, target["prefix"],
                                        scaling=scaling)
            direct = fit.multi_pgs(panel.scores)
            assert abs(np.corrcoef(scored.scores, direct)[0, 1] - 1) < 1e-8


def test_combine_weights_checks_the_fit_matches_the_panel(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    rng = np.random.default_rng(1)
    y = target["true_scores"][:, 0] + rng.normal(size=250)
    fit = multi_pgs_fit(panel.scores, y, n_folds=4, n_lambda=20, seed=0,
                        score_ids=panel.score_ids)
    with pytest.raises(ValueError, match="4 coefficients but the panel has 3"):
        combine_weights(panel.select([0, 1, 2]), fit)
    renamed = panel.select([0, 1, 2, 3])
    renamed.score_ids = np.array(["a", "b", "c", "d"], dtype=object)
    with pytest.raises(ValueError, match="do not match"):
        combine_weights(renamed, fit)
    stripped = read_panel_roundtrip(panel)
    with pytest.raises(ValueError, match="no per-variant weights"):
        combine_weights(stripped, fit)


def read_panel_roundtrip(panel, tmp=None):
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.tsv")
        write_panel(panel, path)
        return read_panel(path)


def test_write_and_read_panel_round_trip(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    back = read_panel_roundtrip(panel)
    assert np.allclose(back.scores, panel.scores, rtol=1e-5)
    assert list(back.score_ids) == list(panel.score_ids)
    assert list(back.sample_iid) == list(panel.sample_iid)


def test_select_by_id_index_and_mask(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    by_id = panel.select(["PGS000002", "PGS000004"])
    by_idx = panel.select([1, 3])
    mask = np.zeros(4, dtype=bool)
    mask[[1, 3]] = True
    by_mask = panel.select(mask)
    for sub in (by_id, by_idx, by_mask):
        assert sub.n_scores == 2
        assert np.allclose(sub.scores, panel.scores[:, [1, 3]])
        assert len(sub.weights) == 2
    assert panel.index_of("PGS000003") == 2
    with pytest.raises(KeyError):
        panel.index_of("nope")


def test_align_matches_individuals_across_panels(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    other = panel._rows(np.arange(200)[::-1])
    a, b = panel.align(other)
    assert len(a) == len(b) == 200
    assert list(a.sample_iid) == list(b.sample_iid)
    assert np.allclose(a.scores, b.scores)


def test_align_without_shared_individuals_is_an_error(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    other = panel._rows(np.arange(5))
    other.sample_iid = np.array([f"OTHER{i}" for i in range(5)], dtype=object)
    other.sample_fid = other.sample_iid
    with pytest.raises(ValueError, match="share no individuals"):
        panel.align(other)
