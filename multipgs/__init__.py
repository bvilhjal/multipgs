"""multipgs — multivariate polygenic scoring: many scores in, one score out.

A single-trait polygenic score uses one GWAS. Many traits have smaller discovery
GWAS than genetically correlated phenotypes. **Multi-PGS** can borrow from that
auxiliary information by computing and combining scores for many traits; gains
are possible when those signals generalize to the target cohort.

Three fitting routes, for three information sets:

``multi_pgs_fit`` — **learned weights**, when a training cohort exists.
    A penalized regression of the phenotype on the ``K`` scores, selected by
    Cross-Model Selection and Averaging. This is the estimator of Albiñana
    et al., `Nat Commun 14, 4702 (2023)
    <https://doi.org/10.1038/s41467-023-40330-w>`_, which combined 937 PGS
    Catalog scores. It learns which traits are relevant, so the input panel can
    be large and mostly irrelevant.

``multi_pgs_sumstats`` — **learned weights**, when only summary statistics
    exist. The same Gaussian score-stacking objective is fitted from
    ``W_ld.T @ D @ W_ld`` and ``W_gwas.T @ z``. Its lasso selects whole
    component scores; it is inspired by lassosum but is not SNP-level lassosum.
    An independent target-trait GWAS can tune the penalty, while assessment
    still requires a third untouched GWAS or individuals.

``meta_pgs`` — **derived weights**, when no phenotype is available.
    Scores of the *same* trait from different discovery GWAS, standardized and
    weighted by :math:`\\sqrt{n_\\mathrm{eff}}` (or by fitted accuracy, or
    decorrelated against each other). No training cohort, no tuning. It assumes
    the scores estimate one genetic value, which ``multi_pgs_fit`` does not.

Getting the ``K`` scores is the other half of the problem.
:mod:`multipgs.fetch` acquires PGS Catalog scoring files and their provenance;
:mod:`multipgs.panel` then scores them in one pass over the target genotypes
(:func:`panel_from_catalog`), or fits each GWAS with LDpred3
(:func:`panel_from_sumstats`). :func:`combine_weights` folds a fitted
combination back into a single per-variant weight file, which is the artefact
you deploy.

:mod:`multipgs.architecture` decides what deserves to be in the panel at all,
using the summary-statistic screening of Hansen et al.
(`Research Square, 2026 <https://doi.org/10.21203/rs.3.rs-9415305/v1>`_).

A worked end-to-end example is in the repository's
`examples/minimal.py <https://github.com/bvilhjal/multipgs/blob/main/examples/minimal.py>`_;
the input contract and failure modes are in
`docs/guide.md <https://github.com/bvilhjal/multipgs/blob/main/docs/guide.md>`_.

Names are imported **lazily** (PEP 562), so ``import multipgs`` stays cheap and
does not pull in ldpred3's Numba kernels until something needs them.
"""

from __future__ import annotations

import importlib

__version__ = "0.3.1"

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
    "fetch": ["search_scores", "ScoreRecord", "download_scores",
              "write_score_metadata", "cohort_overlap"],
    "sumstat": ["multi_pgs_sumstats", "SumstatFit", "score_gram", "pseudo_r2",
                "align_to_reference", "evaluate_sumstat", "SumstatEval",
                "score_moments", "REGIMES", "subsample_score_moments"],
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
