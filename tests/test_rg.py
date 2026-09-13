"""Optional bipred r_G screen: import contract and ranking helper."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from multipgs import penalty_from_relevance


def test_rg_module_is_public():
    from multipgs import RgScreen, align_sumstats_to_cache, ldsc_rg_screen
    assert callable(align_sumstats_to_cache)
    assert callable(ldsc_rg_screen)
    assert RgScreen.__name__ == "RgScreen"


def test_ldsc_rg_screen_names_the_missing_extra(monkeypatch):
    import multipgs.rg as rg

    monkeypatch.setitem(sys.modules, "bipred", None)
    with pytest.raises(ImportError, match="multipgs\\[bipred\\]"):
        rg._require_bipred()


def test_relevance_is_zero_when_rg_is_missing():
    pf = penalty_from_relevance([0.4, 0.4], [0.9, np.nan])
    assert pf[1] > pf[0]


def test_rg_screen_loads_scores_and_closes_cache_once(monkeypatch):
    import ldpred3
    import ldpred3.interop
    import multipgs.rg as rg

    calls = {"load": 0, "ld_scores": 0, "close": 0, "align": [],
             "m_snps": []}

    class Blocks:
        def close(self):
            calls["close"] += 1

    ids = np.array([f"rs{i}" for i in range(5)])
    meta = {
        "counted_allele": np.repeat("A", 5),
        "other_allele": np.repeat("G", 5),
        "chrom": np.array(["6", "1", "1", "2", "2"]),
        "pos": np.arange(5),
    }

    def load(*args, **kwargs):
        calls["load"] += 1
        return Blocks(), ids, meta

    def scores(blocks):
        calls["ld_scores"] += 1
        return np.ones(5)

    def align(source, variants, **kwargs):
        calls["align"].append((source, id(variants)))
        return np.repeat(0.01, 5), np.repeat(100_000.0, 5), {}

    fake_bipred = types.ModuleType("bipred")
    fake_bipred.ldsc_chi2_mask = lambda beta, n: np.ones(beta.size, dtype=bool)

    def fake_rg(*args, **kwargs):
        calls["m_snps"].append(kwargs["m_snps"])
        return SimpleNamespace(rg=0.3, rg_se=0.04)

    fake_bipred.ldsc_rg = fake_rg
    fake_bipred.estimate_sample_overlap = lambda *args: {
        "overlap_corr": 0.1, "cross_corr_valid": True}
    fake_bipred.in_long_range_ld = lambda chrom, pos: np.array(
        [True, False, False, False, False])
    monkeypatch.setitem(sys.modules, "bipred", fake_bipred)
    monkeypatch.setattr(ldpred3.interop, "load_ld_blocks", load)
    monkeypatch.setattr(ldpred3, "ld_scores", scores)
    monkeypatch.setattr(rg, "_align_sumstats", align)

    result = rg.ldsc_rg_screen(
        "focal", [("a", "aux-a"), ("b", "aux-b")], "cache",
        min_snps=2, exclude_long_range_ld=True)
    assert calls["load"] == calls["ld_scores"] == calls["close"] == 1
    assert [source for source, _ in calls["align"]] == [
        "focal", "aux-a", "aux-b"]
    assert len({variant_id for _, variant_id in calls["align"]}) == 1
    assert calls["m_snps"] == [5, 5]
    assert result.rg.tolist() == pytest.approx([0.3, 0.3])
    assert result.n_used.tolist() == [4, 4]
    assert result.log["screen"]["n_long_range_excluded"] == 1


def test_cache_metadata_is_validated_before_alignment(monkeypatch):
    import ldpred3.interop
    import multipgs.rg as rg

    closed = []

    class Blocks:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        ldpred3.interop, "load_ld_blocks",
        lambda *args, **kwargs: (
            Blocks(), np.array(["a", "b", "c"]),
            {"counted_allele": ["A"] * 3, "other_allele": ["G"] * 3,
             "chrom": ["1"] * 3, "pos": [1, 2]}))
    with pytest.raises(ValueError, match="metadata 'pos'.*expected 3"):
        rg.align_sumstats_to_cache("not-read", "cache")
    assert closed == [True]


def test_real_ldpred3_cache_alignment_uses_variant_table(tmp_path):
    """Exercise the real harmonizer; mocks previously hid a missing ``len``."""
    from ldpred3 import standardize_betas
    from ldpred3.ld import save_ld_blocks

    cache = tmp_path / "ld.npz"
    ids = np.array(["rs1", "rs2", "rs3"], dtype=object)
    save_ld_blocks(
        cache, [(np.eye(3, dtype=np.float32), np.arange(3))], ids,
        counted_allele=np.array(["A", "A", "A"]),
        other_allele=np.array(["C", "C", "C"]),
        chrom=np.array(["1", "1", "2"]), pos=np.array([10, 20, 30]),
        reference_af=np.array([0.2, 0.3, 0.4]), n_ref=500, ridge=0.0)
    sumstats = tmp_path / "gwas.tsv"
    sumstats.write_text(
        "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n"
        "rs1\t1\t10\tA\tC\t0.10\t0.05\t1000\n"
        "rs2\t1\t20\tC\tA\t0.20\t0.10\t2000\n"
        "absent\t9\t99\tA\tC\t1.00\t0.10\t3000\n",
        encoding="utf-8")

    from multipgs import align_sumstats_to_cache
    beta, n_eff, log = align_sumstats_to_cache(sumstats, cache, qc=False)
    expected, _ = standardize_betas(
        np.array([0.10, -0.20]), np.array([0.05, 0.10]),
        np.array([1000.0, 2000.0]))
    assert beta[:2].tolist() == pytest.approx(expected)
    assert np.isnan(beta[2])
    assert n_eff.tolist()[:2] == [1000.0, 2000.0]
    assert np.isnan(n_eff[2])
    assert log["n_matched"] == 2


def _n_anchor_guard():
    import multipgs.rg as rg
    if not rg._HAVE_N_ANCHOR:
        pytest.skip("installed ldpred3 predates the n_eff anchor helpers")


def _three_variant_cache(tmp_path):
    from ldpred3.ld import save_ld_blocks
    cache = tmp_path / "ld.npz"
    save_ld_blocks(
        cache, [(np.eye(3, dtype=np.float32), np.arange(3))],
        np.array(["rs1", "rs2", "rs3"], dtype=object),
        counted_allele=np.array(["A", "A", "A"]),
        other_allele=np.array(["C", "C", "C"]),
        chrom=np.array(["1", "1", "1"]), pos=np.array([10, 20, 30]),
        reference_af=np.array([0.2, 0.3, 0.4]), n_ref=500, ridge=0.0)
    return cache


def _three_variant_gwas(tmp_path, ns, name="gwas.tsv", n_header="N"):
    path = tmp_path / name
    header = "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE" + (f"\t{n_header}" if n_header else "")
    lines = [header]
    for (rsid, pos), n in zip((("rs1", 10), ("rs2", 20), ("rs3", 30)), ns):
        row = f"{rsid}\t1\t{pos}\tA\tC\t0.10\t0.05"
        lines.append(row + (f"\t{n}" if n_header else ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_scalar_n_eff_anchors_the_files_n_column(tmp_path):
    """scalar + per-variant N: median anchored, relative pattern kept."""
    _n_anchor_guard()
    from multipgs import align_sumstats_to_cache
    cache = _three_variant_cache(tmp_path)
    gwas = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000))
    with pytest.warns(UserWarning, match="rescaled by factor"):
        beta, n_eff, log = align_sumstats_to_cache(
            gwas, cache, n_eff=4_000, qc=False)
    # median 8_000 -> anchored at 4_000 (factor 0.5); 1:2:4 ratios preserved
    np.testing.assert_allclose(n_eff, [2_000, 4_000, 8_000])
    record = log["qc"]["n_eff_rescale"]
    assert record["applied"] and record["factor"] == pytest.approx(0.5)
    assert record["source_median"] == pytest.approx(8_000)
    assert record["target_effective_n"] == 4_000
    assert not record["upward_scaling_refused"]


def test_scalar_n_eff_above_the_column_median_is_refused(tmp_path):
    _n_anchor_guard()
    from multipgs import align_sumstats_to_cache
    cache = _three_variant_cache(tmp_path)
    gwas = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000))
    with pytest.warns(UserWarning, match="upward rescale refused"):
        beta, n_eff, log = align_sumstats_to_cache(
            gwas, cache, n_eff=40_000, qc=False)
    np.testing.assert_allclose(n_eff, [4_000, 8_000, 16_000])
    record = log["qc"]["n_eff_rescale"]
    assert record["upward_scaling_refused"] and not record["applied"]


def test_scalar_n_eff_without_an_n_column_stays_constant(tmp_path):
    from multipgs import align_sumstats_to_cache
    cache = _three_variant_cache(tmp_path)
    gwas = _three_variant_gwas(tmp_path, (None, None, None), n_header=None)
    beta, n_eff, log = align_sumstats_to_cache(
        gwas, cache, n_eff=4_000, qc=False)
    np.testing.assert_allclose(n_eff, [4_000, 4_000, 4_000])
    assert "n_eff_rescale" not in log["qc"]


def _n_report_guard():
    import multipgs.rg as rg
    if not rg._HAVE_N_REPORT:
        pytest.skip("installed ldpred3 predates the n_eff report helpers")


def test_align_sumstats_composes_the_n_eff_report(tmp_path):
    """qc.n_eff reports offered/supplied/transform/fitted together."""
    _n_report_guard()
    from multipgs import align_sumstats_to_cache
    cache = _three_variant_cache(tmp_path)
    gwas = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000))
    with pytest.warns(UserWarning, match="rescaled by factor"):
        beta, n_eff, log = align_sumstats_to_cache(
            gwas, cache, n_eff=4_000, qc=False)
    report = log["qc"]["n_eff"]
    # File stats are pre-transform: the column median was 8_000 before the
    # anchor halved it.
    assert report["file_column"] == "N"
    assert report["file_n_usable"] == 3
    assert report["file_n_total"] == 3
    assert report["file_median"] == pytest.approx(8_000)
    assert report["supplied_scalar"] == 4_000.0
    assert report["transform"] is log["qc"]["n_eff_rescale"]
    # The fitted summary describes the matched panel's post-transform N.
    assert report["fitted"]["n_variants"] == 3
    assert report["fitted"]["median"] == pytest.approx(4_000)
    assert report["fitted"]["min"] == pytest.approx(2_000)
    assert report["fitted"]["max"] == pytest.approx(8_000)

    # No scalar: the column is used as offered; nothing supplied, no
    # transform ran.
    beta, n_eff, log = align_sumstats_to_cache(gwas, cache, qc=False)
    report = log["qc"]["n_eff"]
    assert report["file_column"] == "N"
    assert report["file_median"] == pytest.approx(8_000)
    assert report["supplied_scalar"] is None
    assert report["transform"] is None
    assert report["fitted"]["median"] == pytest.approx(8_000)

    # No column + scalar: the file offered nothing; the scalar went in as
    # a constant and the fitted panel is flat.
    bare = _three_variant_gwas(tmp_path, (None, None, None),
                               name="bare.tsv", n_header=None)
    beta, n_eff, log = align_sumstats_to_cache(
        bare, cache, n_eff=4_000, qc=False)
    report = log["qc"]["n_eff"]
    assert report["file_column"] is None
    assert report["file_n_usable"] == 0
    assert report["file_n_total"] == 3
    assert report["supplied_scalar"] == 4_000.0
    assert report["transform"] is None
    assert report["fitted"]["median"] == pytest.approx(4_000)
    assert report["fitted"]["min"] == report["fitted"]["max"]


def test_rg_screen_summary_reports_n_anchoring(tmp_path, monkeypatch):
    """The summary line names the focal and counts anchored auxiliaries."""
    _n_report_guard()
    import ldpred3
    import multipgs.rg as rg

    cache = _three_variant_cache(tmp_path)
    focal = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000), "focal.tsv")
    aux_a = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000), "a.tsv")
    aux_b = _three_variant_gwas(tmp_path, (2_000, 4_000, 8_000), "b.tsv")

    fake_bipred = types.ModuleType("bipred")
    fake_bipred.ldsc_chi2_mask = lambda beta, n: np.ones(beta.size, dtype=bool)
    fake_bipred.ldsc_rg = lambda *args, **kwargs: SimpleNamespace(
        rg=0.3, rg_se=0.04)
    fake_bipred.estimate_sample_overlap = lambda *args: {
        "overlap_corr": 0.1, "cross_corr_valid": True}
    monkeypatch.setitem(sys.modules, "bipred", fake_bipred)
    monkeypatch.setattr(ldpred3, "ld_scores", lambda blocks: np.ones(3))

    with pytest.warns(UserWarning, match="rescaled by factor"):
        result = rg.ldsc_rg_screen(
            focal, [("a", aux_a), ("b", aux_b)], cache,
            n_eff_focal=4_000, n_eff={"a": 4_000, "b": 500},
            qc=False, min_snps=2)
    line = [ln for ln in result.summary().splitlines()
            if "anchored" in ln]
    assert line == ["  N anchored at supplied scalar: focal (×0.5), "
                    "2 auxiliaries"]

    # Without a supplied scalar nothing anchored: no such line.
    result = rg.ldsc_rg_screen(
        focal, [("a", aux_a)], cache, qc=False, min_snps=2)
    assert "anchored" not in result.summary()


def test_rg_screen_anchors_focal_and_per_auxiliary_n_eff(
        tmp_path, monkeypatch):
    """The n_eff map rescales each auxiliary file's own N column."""
    _n_anchor_guard()
    import ldpred3
    import multipgs.rg as rg

    cache = _three_variant_cache(tmp_path)
    focal = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000), "focal.tsv")
    aux_a = _three_variant_gwas(tmp_path, (4_000, 8_000, 16_000), "a.tsv")
    aux_b = _three_variant_gwas(tmp_path, (2_000, 4_000, 8_000), "b.tsv")

    fake_bipred = types.ModuleType("bipred")
    fake_bipred.ldsc_chi2_mask = lambda beta, n: np.ones(beta.size, dtype=bool)
    fake_bipred.ldsc_rg = lambda *args, **kwargs: SimpleNamespace(
        rg=0.3, rg_se=0.04)
    fake_bipred.estimate_sample_overlap = lambda *args: {
        "overlap_corr": 0.1, "cross_corr_valid": True}
    monkeypatch.setitem(sys.modules, "bipred", fake_bipred)
    monkeypatch.setattr(ldpred3, "ld_scores", lambda blocks: np.ones(3))

    with pytest.warns(UserWarning, match="rescaled by factor"):
        result = rg.ldsc_rg_screen(
            focal, [("a", aux_a), ("b", aux_b)], cache,
            n_eff_focal=4_000, n_eff={"a": 4_000, "b": 500},
            qc=False, min_snps=2)

    focal_rec = result.log["focal"]["qc"]["n_eff_rescale"]
    assert focal_rec["applied"] and focal_rec["factor"] == pytest.approx(0.5)
    rec_a = result.log["aux"]["a"]["qc"]["n_eff_rescale"]
    assert rec_a["factor"] == pytest.approx(0.5)
    assert rec_a["target_effective_n"] == 4_000
    rec_b = result.log["aux"]["b"]["qc"]["n_eff_rescale"]
    # aux-b's own column (median 4_000) anchors at its own scalar 500
    assert rec_b["factor"] == pytest.approx(0.125)
    assert rec_b["target_effective_n"] == 500
    assert result.n_used.tolist() == [3, 3]


def test_rg_alignment_borrows_prepared_cache_without_reloading(
        tmp_path, monkeypatch):
    """A caller-owned prepared cache remains open and is never reloaded."""
    import ldpred3.interop as interop
    from ldpred3.ld import save_ld_blocks

    cache = tmp_path / "prepared.npz"
    save_ld_blocks(
        cache, [(np.eye(2, dtype=np.float32), np.arange(2))],
        np.array(["rs1", "rs2"], dtype=object),
        counted_allele=np.array(["A", "C"]),
        other_allele=np.array(["G", "T"]),
        chrom=np.array(["1", "1"]), pos=np.array([10, 20]),
        reference_af=np.array([0.2, 0.3]), n_ref=500, ridge=0.0)
    sumstats = tmp_path / "gwas.tsv"
    sumstats.write_text(
        "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n"
        "rs1\t1\t10\tA\tG\t0.10\t0.05\t1000\n",
        encoding="utf-8")

    prepared = interop.prepare_ld_cache(cache)
    try:
        monkeypatch.setattr(
            interop, "load_ld_blocks",
            lambda *args, **kwargs: pytest.fail("prepared cache was reloaded"))
        from multipgs import align_sumstats_to_cache
        beta, n_eff, log = align_sumstats_to_cache(
            sumstats, prepared, qc=False)
        assert np.isfinite(beta[0]) and np.isnan(beta[1])
        assert n_eff[0] == 1000 and np.isnan(n_eff[1])
        assert log["n_matched"] == 1
        assert not prepared.closed
    finally:
        prepared.close()
    assert prepared.closed


@pytest.mark.parametrize("value", [1, 1.5, True])
def test_rg_screen_rejects_tiny_or_noninteger_min_snps(monkeypatch, value):
    fake_bipred = types.ModuleType("bipred")
    monkeypatch.setitem(sys.modules, "bipred", fake_bipred)
    import multipgs.rg as rg

    with pytest.raises(ValueError, match="integer >= 2"):
        rg.ldsc_rg_screen("focal", [], "cache", min_snps=value)
