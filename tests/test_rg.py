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
