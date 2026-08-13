"""Optional bipred r_G screen: import contract and ranking helper."""

import numpy as np
import pytest

from multipgs import penalty_from_relevance


def test_rg_module_is_public():
    from multipgs import RgScreen, align_sumstats_to_cache, ldsc_rg_screen
    assert callable(align_sumstats_to_cache)
    assert callable(ldsc_rg_screen)
    assert RgScreen.__name__ == "RgScreen"


def test_ldsc_rg_screen_names_the_missing_extra(monkeypatch):
    import sys

    import multipgs.rg as rg

    monkeypatch.setitem(sys.modules, "bipred", None)
    with pytest.raises(ImportError, match="multipgs\\[bipred\\]"):
        rg._require_bipred()


def test_relevance_is_zero_when_rg_is_missing():
    pf = penalty_from_relevance([0.4, 0.4], [0.9, np.nan])
    assert pf[1] > pf[0]
