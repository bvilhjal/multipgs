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
    import ldpred3.ld
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
    monkeypatch.setattr(ldpred3.ld, "load_ld_blocks", load)
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
    import ldpred3.ld
    import multipgs.rg as rg

    closed = []

    class Blocks:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        ldpred3.ld, "load_ld_blocks",
        lambda *args, **kwargs: (
            Blocks(), np.array(["a", "b", "c"]),
            {"counted_allele": ["A"] * 3, "other_allele": ["G"] * 3,
             "chrom": ["1"] * 3, "pos": [1, 2]}))
    with pytest.raises(ValueError, match="metadata 'pos'.*expected 3"):
        rg.align_sumstats_to_cache("not-read", "cache")
    assert closed == [True]


@pytest.mark.parametrize("value", [1, 1.5, True])
def test_rg_screen_rejects_tiny_or_noninteger_min_snps(monkeypatch, value):
    fake_bipred = types.ModuleType("bipred")
    monkeypatch.setitem(sys.modules, "bipred", fake_bipred)
    import multipgs.rg as rg

    with pytest.raises(ValueError, match="integer >= 2"):
        rg.ldsc_rg_screen("focal", [], "cache", min_snps=value)
