"""Panel construction from genotypes, and folding a fit back to one weight set."""

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from multipgs import (ScorePanel, architectures_from_panel, check_weights,
                      combine_weights, load_panel, multi_pgs_fit,
                      panel_from_catalog, panel_from_sumstats,
                      panel_from_weights, read_panel, read_trait_table,
                      save_panel, screen, simulate_target, write_panel)


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


def test_catalog_weight_metadata_is_shared_and_compact(tmp_path):
    """A fully overlapping panel stores metadata once, not K times."""
    made = simulate_target(str(tmp_path / "shared"), n=20, n_variants=128,
                           n_scores=8, n_per_score=128, seed=17)
    panel = panel_from_catalog(made["scoring_files"], made["prefix"])
    shared = panel.weights[0].variant_table
    assert all(table.variant_table is shared for table in panel.weights)

    compact_bytes = sum(array.nbytes for array in shared.values())
    compact_bytes += sum(table.index.nbytes + table["weight"].nbytes
                         for table in panel.weights)
    legacy_bytes = sum(
        table["weight"].nbytes
        + sum(table[key].nbytes
              for key in ("id", "chrom", "pos", "a1", "a2", "af", "sd"))
        for table in panel.weights)
    # Eight completely overlapping scores should need far below half of the
    # legacy arrays. This counts NumPy buffers only; Python-string payloads are
    # shared already and would not make the compact representation look better.
    assert compact_bytes < 0.35 * legacy_bytes

    # The legacy mapping interface remains available and materializes the
    # score-specific rows only when requested.
    assert set(panel.weights[0]) == {
        "id", "chrom", "pos", "a1", "a2", "weight", "af", "sd"}
    assert np.array_equal(panel.weights[0]["id"], made["variants"].id)


def test_compact_catalog_panel_is_pickleable_and_deepcopyable(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    for restored in (pickle.loads(pickle.dumps(panel)), copy.deepcopy(panel)):
        shared = restored.weights[0].variant_table
        assert all(table.variant_table is shared for table in restored.weights)
        assert np.allclose(restored.scores, panel.scores)
        assert np.array_equal(restored.weights[0]["id"],
                              panel.weights[0]["id"])
        assert not shared["id"].flags.writeable
        with pytest.raises(TypeError):
            shared["id"] = np.array([], dtype=object)


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


def test_npz_round_trip_keeps_weights_and_scale(target, tmp_path):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    path = tmp_path / "panel.npz"
    save_panel(panel, path)
    back = load_panel(path)
    assert np.allclose(back.scores, panel.scores)
    assert list(back.score_ids) == list(panel.score_ids)
    assert np.array_equal(back.standardized, panel.standardized)
    assert len(back.weights) == panel.n_scores
    assert np.allclose(back.weights[0]["weight"], panel.weights[0]["weight"])
    assert np.array_equal(back.weights[0]["id"], panel.weights[0]["id"])


def test_concat_joins_on_shared_individuals_and_rejects_id_clash(target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    left, right = panel.select([0, 1]), panel.select([2, 3])
    both = left.concat(right)
    assert both.n_scores == 4
    assert list(both.score_ids) == list(panel.score_ids)
    assert np.allclose(both.scores, panel.scores)
    with pytest.raises(ValueError, match="collide"):
        left.concat(panel.select([1, 2]))


def test_concat_preserves_partial_metadata_identity_and_rejects_mixed_weights(
        target):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    left, right = panel.select([0]), panel.select([1])
    left.meta = []
    both = left.concat(right)
    assert both.meta[0] == {}
    assert both.meta[1]["path"] == right.meta[0]["path"]
    assert len(both.meta) == both.n_scores

    plain = read_panel_roundtrip(panel).select([0])
    with pytest.raises(ValueError, match="carrying per-score weights"):
        plain.concat(right)


def test_panel_from_weights_without_scale_matches_target_standardization(
        target, tmp_path):
    from ldpred3 import score_from_weights
    from ldpred3.weights import write_weights

    panel = panel_from_catalog(target["scoring_files"][:1], target["prefix"])
    table = panel.weights[0]
    path = tmp_path / "one.weights"
    write_weights(path, id=table["id"], chrom=table["chrom"], pos=table["pos"],
                  effect_allele=table["a1"], other_allele=table["a2"],
                  weight=table["weight"])
    scored = panel_from_weights(str(path), target["prefix"])
    expected = score_from_weights(str(path), target["prefix"],
                                  scaling="target")
    assert scored.n_scores == 1
    assert scored.standardized[0]
    assert np.allclose(scored.scores[:, 0], expected.scores, atol=1e-9)
    assert np.all(np.isfinite(scored.weights[0]["af"]))
    assert np.all(scored.weights[0]["sd"] > 0)

    fit = SimpleNamespace(beta=np.array([1.3]), score_ids=scored.score_ids,
                          multi_pgs=lambda p: 1.3 * p.scores[:, 0])
    combined_path = tmp_path / "target-scaled.weights"
    combine_weights(scored, fit, path=str(combined_path))
    deployed = score_from_weights(str(combined_path), target["prefix"],
                                  scaling="frozen")
    direct = fit.multi_pgs(scored)
    assert np.allclose(deployed.scores - deployed.scores.mean(),
                       direct - direct.mean(), rtol=2e-7, atol=2e-7)


def test_check_weights_accepts_a_faithful_combined_file(target, tmp_path):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    rng = np.random.default_rng(2)
    y = panel.scores[:, 0] + rng.normal(size=len(panel)) * 0.1
    fit = multi_pgs_fit(panel.scores, y, n_folds=4, n_lambda=20, seed=0,
                        score_ids=panel.score_ids)
    path = tmp_path / "multi.weights"
    combine_weights(panel, fit, path=str(path))
    info = check_weights(panel, fit, str(path), target["prefix"])
    assert info["corr"] > 0.999999
    assert info["slope"] == pytest.approx(1.0, rel=1e-6)
    assert info["max_abs_error"] <= info["tolerance"]
    assert info["n"] == len(panel)


def test_check_weights_rejects_a_rescaled_predictor(target, monkeypatch):
    import ldpred3

    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    fit = SimpleNamespace(
        beta=np.ones(panel.n_scores), score_ids=panel.score_ids,
        multi_pgs=lambda p: np.asarray(p.scores) @ np.ones(p.n_scores))
    direct = fit.multi_pgs(panel)
    fake = SimpleNamespace(
        scores=2.0 * direct, sample_fid=panel.sample_fid,
        sample_iid=panel.sample_iid)
    monkeypatch.setattr(ldpred3, "score_from_weights",
                        lambda *args, **kwargs: fake)
    with pytest.raises(ValueError, match="does not reproduce"):
        check_weights(panel, fit, "scaled.weights", target["prefix"])


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"min_corr": 1.01}, "min_corr must be in"),
    ({"min_corr": np.nan}, "finite numeric"),
    ({"min_corr": True}, "finite numeric"),
    ({"rtol": -1.0}, "non-negative"),
    ({"atol": np.inf}, "finite numeric"),
    ({"atol": False}, "finite numeric"),
])
def test_check_weights_rejects_invalid_tolerances(kwargs, message):
    panel = ScorePanel(
        scores=np.ones((2, 1)), sample_fid=np.array(["F1", "F2"]),
        sample_iid=np.array(["I1", "I2"]),
        score_ids=np.array(["s"], dtype=object),
        standardized=np.ones(1, dtype=bool))
    fit = SimpleNamespace(beta=np.ones(1), score_ids=panel.score_ids)
    with pytest.raises(ValueError, match=message):
        check_weights(panel, fit, "unused.weights", "unused", **kwargs)


def test_read_trait_table_converts_blank_n_eff(tmp_path):
    path = tmp_path / "traits.tsv"
    path.write_text(
        "TRAIT\tPATH\tN_EFF\tN_CASES\tN_CONTROLS\tMETHOD\tALPHA\n"
        "cad\tcad.tsv\t.\t60801\t123504\tauto\t-1\n"
        "bmi\tbmi.tsv\t681275\t.\t.\tauto\t-0.5\n", encoding="utf-8")
    rows = read_trait_table(str(path))
    assert rows[0]["id"] == "cad" and rows[0]["n_eff"] is None
    assert rows[0]["n_cases"] == 60801 and rows[1]["n_eff"] == 681275
    assert rows[1]["alpha"] == -0.5


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

    one = panel.select(1)
    assert one.scores.shape == (len(panel), 1)
    assert list(one.score_ids) == ["PGS000002"]
    empty = panel.select([])
    assert empty.scores.shape == (len(panel), 0)
    assert empty.n_scores == 0


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


def test_align_uses_tuple_keys_and_rejects_duplicate_samples(target):
    left = panel_from_catalog(target["scoring_files"], target["prefix"])._rows([0])
    right = left._rows([0])
    left.sample_fid = np.array(["a:b"], dtype=object)
    left.sample_iid = np.array(["c"], dtype=object)
    right.sample_fid = np.array(["a"], dtype=object)
    right.sample_iid = np.array(["b:c"], dtype=object)
    with pytest.raises(ValueError, match="share no individuals"):
        left.align(right)

    duplicate = left._rows([0, 0])
    with pytest.raises(ValueError, match="duplicate FID:IID"):
        duplicate.align(left)


def test_empty_panel_round_trips_with_its_two_dimensional_shape(target,
                                                                tmp_path):
    panel = panel_from_catalog(target["scoring_files"], target["prefix"])
    cases = (panel._rows([]), panel.select([]))
    for i, case in enumerate(cases):
        path = tmp_path / f"empty_{i}.tsv"
        write_panel(case, path)
        back = read_panel(path)
        assert back.scores.shape == case.scores.shape
        assert list(back.score_ids) == list(case.score_ids)


def _ldpred3_result(*, fid=("F1", "F2"), iid=("I1", "I2"), inference=None):
    return SimpleNamespace(
        sample_fid=np.array(fid, dtype=object),
        sample_iid=np.array(iid, dtype=object),
        scores=np.array([0.1, 0.2]), var_index=np.array([0]),
        harmonize_log={}, qc_log={}, inference=inference,
        variant_id=np.array(["rs1"], dtype=object),
        chrom=np.array(["1"], dtype=object), pos=np.array([100]),
        effect_allele=np.array(["A"], dtype=object),
        other_allele=np.array(["G"], dtype=object),
        beta_adjusted=np.array([0.3]), af=np.array([0.2]), sd=np.array([0.5]))


def test_panel_from_sumstats_merges_per_trait_n_eff(monkeypatch, tmp_path):
    import ldpred3

    seen = []

    def fake(path, plink, **kwargs):
        seen.append(kwargs.get("n_eff"))
        return _ldpred3_result()

    monkeypatch.setattr(ldpred3, "run_ldpred3_prs", fake)
    panel = panel_from_sumstats(
        None, "target", ld_cache=str(tmp_path / "ld"),
        traits=[{"id": "cad", "path": "cad.tsv", "n_cases": 100, "n_controls": 300},
                {"id": "bmi", "path": "bmi.tsv", "n_eff": 50_000}])
    assert seen[0] == pytest.approx(4.0 / (1 / 100 + 1 / 300))
    assert seen[1] == 50_000
    assert panel.meta[0]["n_eff"] == pytest.approx(seen[0])
    assert panel.meta[1]["n_eff"] == 50_000


def test_catalog_attaches_sidecar_metadata(target, tmp_path):
    import os
    import shutil
    from multipgs.fetch import write_score_metadata, ScoreRecord

    d = tmp_path / "cat"
    d.mkdir()
    for p in target["scoring_files"]:
        shutil.copy(p, d / os.path.basename(p))
    records = [ScoreRecord(pgs_id="PGS000001", n_eff=12_345.0)]
    write_score_metadata(records, str(d / "metadata.tsv"), columns=["N_EFF"])
    panel = panel_from_catalog(str(d), target["prefix"])
    assert panel.meta[0].get("n_eff") == 12_345.0


def test_panel_from_sumstats_runs_later_traits_after_the_cache(monkeypatch,
                                                              tmp_path):
    """The first successful fit writes the LD cache; later traits may share threads."""
    import threading

    import ldpred3

    cache = tmp_path / "ld_cache"
    seen = []
    lock = threading.Lock()

    def fake(path, plink, **kwargs):
        with lock:
            seen.append((str(path), "ld_out" in kwargs,
                         threading.current_thread().name))
        return _ldpred3_result()

    monkeypatch.setattr(ldpred3, "run_ldpred3_prs", fake)
    panel = panel_from_sumstats(
        {"one": "one.tsv", "two": "two.tsv", "three": "three.tsv"},
        "target", ld_cache=str(cache), n_jobs=2)
    assert list(panel.score_ids) == ["one", "two", "three"]
    assert seen[0][0] == "one.tsv" and seen[0][1] is True
    later = {path: wrote_ld_out for path, wrote_ld_out, _ in seen[1:]}
    assert later == {"two.tsv": False, "three.tsv": False}
    assert panel.log["ld_reused"] is True


def test_parallel_sumstats_with_ld_prefix_requires_a_shared_cache(monkeypatch):
    import ldpred3

    monkeypatch.setattr(ldpred3, "run_ldpred3_prs",
                        lambda *args, **kwargs: _ldpred3_result())
    with pytest.raises(ValueError, match="requires ld_cache"):
        panel_from_sumstats(
            {"one": "one.tsv", "two": "two.tsv"}, "target",
            ld_prefix="reference", n_jobs=2)


def test_panel_from_sumstats_accepts_missing_inference_and_preserves_mapping(
        monkeypatch):
    import ldpred3

    results = iter([_ldpred3_result(inference=None),
                    _ldpred3_result(inference={"n_chains": 50,
                                               "shrink_corr": 0.7})])
    monkeypatch.setattr(ldpred3, "run_ldpred3_prs",
                        lambda *args, **kwargs: next(results))
    panel = panel_from_sumstats({"one": "one.tsv", "two": "two.tsv"},
                                "target")
    assert panel.meta[0]["inference"] == {}
    assert panel.meta[1]["inference"] == {"n_chains": 50,
                                           "shrink_corr": 0.7}


def test_panel_from_sumstats_compares_both_fid_and_iid(monkeypatch):
    import ldpred3

    results = iter([_ldpred3_result(),
                    _ldpred3_result(fid=("OTHER", "F2"))])
    monkeypatch.setattr(ldpred3, "run_ldpred3_prs",
                        lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="different sample order"):
        panel_from_sumstats({"one": "one.tsv", "two": "two.tsv"}, "target")


def test_panel_from_sumstats_recovers_explicit_inference_controls(monkeypatch):
    import ldpred3

    result = _ldpred3_result(inference={"h2_est": 0.2,
                                        "n_chains_kept": 23})
    monkeypatch.setattr(ldpred3, "run_ldpred3_prs",
                        lambda *args, **kwargs: result)
    panel = panel_from_sumstats(
        ["one.tsv"], "target", infer=True,
        infer_params={"n_chains": 50, "shrink_corr": 0.6})
    inference = panel.meta[0]["inference"]
    assert inference["n_chains_kept"] == 23
    assert inference["n_chains"] == 50
    assert inference["shrink_corr"] == pytest.approx(0.6)
    arch = architectures_from_panel(panel, n_eff=[20_000])
    assert screen(arch, min_variants=1).keep.tolist() == [True]

    # The retained count does not reveal how many chains were attempted.
    panel = panel_from_sumstats(["one.tsv"], "target", infer=True)
    assert "n_chains" not in panel.meta[0]["inference"]

    no_summary = _ldpred3_result(inference=None)
    monkeypatch.setattr(ldpred3, "run_ldpred3_prs",
                        lambda *args, **kwargs: no_summary)
    panel = panel_from_sumstats(
        ["one.tsv"], "target", auto_chains=40,
        infer_params={"n_chains": 50, "shrink_corr": 0.7})
    assert panel.meta[0]["inference"] == {"n_chains": 40,
                                           "shrink_corr": 0.7}


def _weight_panel(tables):
    k = len(tables)
    return ScorePanel(
        scores=np.zeros((2, k)), sample_fid=np.array(["F1", "F2"]),
        sample_iid=np.array(["I1", "I2"]),
        score_ids=np.array([f"s{i}" for i in range(k)], dtype=object),
        standardized=np.ones(k, dtype=bool), weights=tables)


def _weight_table(a1, a2, weight, variant_id="rs1"):
    return {"id": np.array([variant_id], dtype=object),
            "chrom": np.array(["1"], dtype=object), "pos": np.array([100]),
            "a1": np.array([a1], dtype=object),
            "a2": np.array([a2], dtype=object),
            "weight": np.array([weight]), "af": np.array([0.2]),
            "sd": np.array([0.5])}


def test_combine_weights_preserves_multiallelic_variants_at_one_position():
    panel = _weight_panel([_weight_table("A", "G", 1.0, "rs_ag"),
                           _weight_table("A", "C", 2.0, "rs_ac")])
    fit = SimpleNamespace(beta=np.ones(2), score_ids=panel.score_ids)
    out = combine_weights(panel, fit)
    assert out["weight"].size == 2
    assert set(zip(out["a1"], out["a2"])) == {("A", "G"), ("A", "C")}


def test_combine_weights_reports_exact_cancellation():
    first = _weight_table("A", "G", 1.0)
    second = _weight_table("G", "A", 1.0)
    # Same physical reference after orienting G back to A.
    second["af"] = 1.0 - first["af"]
    panel = _weight_panel([first, second])
    fit = SimpleNamespace(beta=np.ones(2), score_ids=panel.score_ids)
    with pytest.raises(ValueError, match="cancel exactly"):
        combine_weights(panel, fit)


def test_frozen_weight_panels_and_mixed_scales_fold_exactly(target, tmp_path):
    """Different AF/SD bases and missing calls survive one-file deployment."""
    from ldpred3 import score_from_weights
    from ldpred3.weights import write_weights

    catalog = panel_from_catalog(target["scoring_files"][:1], target["prefix"])
    table = catalog.weights[0]
    base_af = np.asarray(table["af"], dtype=float)
    base_sd = np.asarray(table["sd"], dtype=float)
    base_w = np.asarray(table["weight"], dtype=float)
    af1 = np.clip(0.8 * base_af + 0.05, 0.01, 0.99)
    af2_target = np.clip(0.7 * base_af + 0.15, 0.01, 0.99)
    sd1 = 0.8 * base_sd + 0.1
    sd2 = 1.4 * base_sd + 0.2
    one = tmp_path / "one.weights"
    two = tmp_path / "two.weights"
    write_weights(
        one, id=table["id"], chrom=table["chrom"], pos=table["pos"],
        effect_allele=table["a1"], other_allele=table["a2"],
        weight=base_w, af=af1, sd=sd1)
    # Store the second file on the opposite allele. Harmonisation must flip its
    # weight and AF back before the two distinct frozen bases are folded.
    write_weights(
        two, id=table["id"], chrom=table["chrom"], pos=table["pos"],
        effect_allele=table["a2"], other_allele=table["a1"],
        weight=-0.6 * base_w, af=1.0 - af2_target, sd=sd2)

    panel = panel_from_weights([str(one), str(two)], target["prefix"])
    expected_one = score_from_weights(str(one), target["prefix"],
                                      scaling="frozen")
    expected_two = score_from_weights(str(two), target["prefix"],
                                      scaling="frozen")
    assert np.allclose(panel.scores[:, 0], expected_one.scores, atol=1e-9)
    assert np.allclose(panel.scores[:, 1], expected_two.scores, atol=1e-9)

    fit = SimpleNamespace(beta=np.array([0.7, 1.2]),
                          score_ids=panel.score_ids,
                          multi_pgs=lambda p: np.asarray(p.scores)
                          @ np.array([0.7, 1.2]))
    combined_path = tmp_path / "combined.weights"
    out = combine_weights(panel, fit, path=str(combined_path))
    deployed = score_from_weights(str(combined_path), target["prefix"],
                                  scaling="frozen")
    direct = fit.multi_pgs(panel)
    assert np.allclose(deployed.scores - deployed.scores.mean(),
                       direct - direct.mean(), rtol=2e-7, atol=2e-7)
    assert np.all(np.isfinite(out["af"]))
    assert np.all(out["sd"] > 0)
    info = check_weights(panel, fit, str(combined_path), target["prefix"])
    assert info["slope"] == pytest.approx(1.0, rel=1e-6)


def test_combine_rejects_an_unrepresentable_signed_frozen_centre():
    first = _weight_table("A", "G", 1.0)
    second = _weight_table("A", "G", 1.0)
    first["af"] = np.array([0.1])
    second["af"] = np.array([0.9])
    first["sd"] = second["sd"] = np.ones(1)
    panel = _weight_panel([first, second])
    fit = SimpleNamespace(beta=np.array([1.0, -0.5]),
                          score_ids=panel.score_ids)
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        combine_weights(panel, fit)


def test_sumstat_fit_combines_raw_coefficients_at_nonunit_score_sd():
    """Summary fitting and deployment must use the same coefficient scale."""
    from multipgs.sumstat import multi_pgs_sumstats

    weights = np.array([[2.0, 0.0],
                        [0.0, 0.5],
                        [1.0, 1.5]])
    variant_ids = np.array(["rs1", "rs2", "rs3"], dtype=object)

    def table(column):
        return {
            "id": variant_ids,
            "chrom": np.array(["1", "1", "1"], dtype=object),
            "pos": np.array([100, 200, 300]),
            "a1": np.array(["A", "C", "G"], dtype=object),
            "a2": np.array(["G", "T", "A"], dtype=object),
            "weight": weights[:, column],
            "af": np.full(3, 0.2),
            "sd": np.ones(3),
        }

    panel = _weight_panel([table(0), table(1)])
    fit = multi_pgs_sumstats(
        weights, np.array([0.5, -0.2, 0.3]), np.eye(3),
        weights_gwas=weights,
        score_ids=panel.score_ids, tune="none", n_lambda=12,
        ld_shrinkage=[0.1])
    assert not np.allclose(fit.score_sd, 1.0)
    assert not np.allclose(fit.beta, fit.beta_std)

    combined = combine_weights(panel, fit)
    order = {str(s): i for i, s in enumerate(combined["id"])}
    deployed = np.array([combined["weight"][order[str(s)]]
                         for s in variant_ids])
    assert np.allclose(deployed, fit.variant_weights(weights))
    assert np.allclose(deployed, weights @ fit.beta)


@pytest.mark.parametrize("beta, message", [
    (np.ones((1, 2)), "1-D"),
    (np.array([1.0, np.nan]), "non-finite"),
])
def test_combine_weights_rejects_invalid_raw_beta(beta, message):
    panel = _weight_panel([_weight_table("A", "G", 1.0),
                           _weight_table("A", "G", 2.0)])
    fit = SimpleNamespace(beta=beta, score_ids=panel.score_ids)
    with pytest.raises(ValueError, match=message):
        combine_weights(panel, fit)


def test_combine_weights_requires_unique_score_binding():
    panel = _weight_panel([_weight_table("A", "G", 1.0),
                           _weight_table("A", "G", 2.0)])
    panel.score_ids = np.array(["same", "same"], dtype=object)
    fit = SimpleNamespace(beta=np.ones(2), score_ids=panel.score_ids)
    with pytest.raises(ValueError, match="score_ids must be unique"):
        combine_weights(panel, fit)
