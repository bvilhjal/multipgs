"""multipgs — multivariate polygenic scoring: many scores in, one score out.

A single-trait polygenic score uses one GWAS. Most traits have a small GWAS and
a great many genetically correlated neighbours with larger ones, and that
correlated information is simply discarded. **Multi-PGS** does not discard it:
it computes polygenic scores for many traits and combines them, which improves
prediction most exactly where single-trait scores are weakest.

Two combiners, for two situations:

``multi_pgs_fit`` — **learned weights**, when a training cohort exists.
    A penalized regression of the phenotype on the ``K`` scores, selected by
    Cross-Model Selection and Averaging. This is the estimator of Albiñana
    et al., `Nat Commun 14, 4702 (2023)
    <https://doi.org/10.1038/s41467-023-40330-w>`_, which combined 937 PGS
    Catalog scores. It learns which traits are relevant, so the input panel can
    be large and mostly irrelevant.

``meta_pgs`` — **derived weights**, when no phenotype is available.
    Scores of the *same* trait from different discovery GWAS, standardized and
    weighted by :math:`\\sqrt{n_\\mathrm{eff}}` (or by fitted accuracy, or
    decorrelated against each other). No training cohort, no tuning. It assumes
    the scores estimate one genetic value, which ``multi_pgs_fit`` does not.

Getting the ``K`` scores is the other half of the problem, and
:mod:`multipgs.panel` does it in one pass over the target genotypes — from PGS
Catalog scoring files (:func:`panel_from_catalog`), or by fitting each GWAS with
LDpred3 (:func:`panel_from_sumstats`). :func:`combine_weights` folds a fitted
combination back into a single per-variant weight file, which is the artefact
you deploy.

:mod:`multipgs.architecture` decides what deserves to be in the panel at all,
using the summary-statistic screening of Hansen et al.
(`Research Square, 2026 <https://doi.org/10.21203/rs.3.rs-9415305/v1>`_).

A worked end-to-end example is in ``examples/minimal.py``; the input contract
and the ways this goes wrong are in ``docs/guide.md``.

Names are imported **lazily** (PEP 562), so ``import multipgs`` stays cheap and
does not pull in ldpred3's Numba kernels until something needs them.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# Public name -> submodule it lives in. No module name may equal one of its own
# exported names: importing a submodule binds it on this package, and the cache
# below would then overwrite that binding with the function, making
# ``multipgs.<name>`` resolve to the module or the function depending on import
# order.
_EXPORTS = {
    "stack": ["multi_pgs_fit", "MultiPGSFit", "FoldFit"],
    "meta": ["meta_pgs", "MetaPGS"],
    "panel": ["ScorePanel", "panel_from_catalog", "panel_from_sumstats",
              "combine_weights", "read_panel", "write_panel"],
    "catalog": ["read_scoring_file", "ScoringFile", "harmonize_scoring_file"],
    "architecture": ["Architecture", "daetwyler_r2", "architectures_from_panel",
                     "screen", "ScreenResult", "penalty_from_accuracy"],
    "metrics": ["evaluate", "EvalResult", "r2", "incremental_r2", "auc",
                 "nagelkerke_r2", "liability_r2"],
    "simulate": ["simulate_panel", "SimPanel", "simulate_same_trait_panel",
                 "simulate_target"],
}

_NAME_TO_MODULE = {name: module
                   for module, names in _EXPORTS.items() for name in names}

__all__ = sorted(_NAME_TO_MODULE) + ["__version__"]


def __getattr__(name):
    module = _NAME_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__) | set(globals()))
