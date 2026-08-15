"""Fit the multi-PGS combination from summary statistics alone.

:func:`multipgs.multi_pgs_fit` needs a training cohort: genotypes, a phenotype,
and enough individuals to cross-validate over. That requirement is what stops
most people using multi-PGS, and it is not actually necessary. The stacking
regression of a phenotype on ``K`` scores depends on the individual-level data
only through two sufficient statistics,

.. math::

    G = X^\\top X / n, \\qquad r = X^\\top y / n,

and both can be estimated from summary-level data. The genotype standard
deviation used to express a raw score differs between datasets, so write
``W_ld`` for the score weights on the LD-reference standardized-genotype basis
and ``W_gwas`` for the same raw scores on the GWAS basis. With ``D`` the LD
correlation matrix and ``z`` the target trait's standardized marginal effects,

.. math::

    G = W_{ld}^\\top D W_{ld}, \\qquad c = W_{gwas}^\\top z.

When LD and marginal effects come from the same individuals these equal
``n^{-1} X^T X`` and ``n^{-1} X^T y`` exactly only for unadjusted centred data,
or when genotype columns and phenotype were jointly residualized on exactly the
same covariate design before both moments were formed. A conventionally
covariate-adjusted marginal GWAS does not, by itself, imply this exact identity.
With an external LD reference, ``G`` is instead a plug-in covariance estimate.
The K-by-K score covariance therefore comes from an **LD reference**, while the
K-vector of score-phenotype covariances comes from a **GWAS of the target
trait**. The two weight matrices may cover different variant sets and orders;
only their K score columns and raw score definition must agree.

This is the multivariate form of the summary-statistic accuracy identity used
by ``ppb`` (this author's cross-ancestry portability benchmark), which is the
quasi-correlation of `Pattee & Pan (2020)
<https://doi.org/10.1371/journal.pcbi.1008271>`_ and the square of lassosum's
pseudovalidation criterion (`Mak et al. 2017
<https://doi.org/10.1002/gepi.22050>`_). For a single score it reads
``R^2 = (w^T z)^2 / (w^T D w)``; the combined score of a stack has weights
``W beta``, so its accuracy is ``(beta^T r)^2 / (beta^T G beta)`` — a function
of exactly the two matrices above and nothing else.

The numerical core needs no changes at all. :func:`multipgs._coord.enet_path_gaussian`
was written with covariance updates so that a fit would cost ``O(K^2)`` per
sweep independent of ``n``; it therefore already takes ``G`` and ``r`` and never
sees ``X`` or ``y``. The summary-statistic path is a different way of filling in
the same two arguments.

**This is lassosum in score space.** With ``alpha=1`` — the default, and the
primary mode — the objective is

.. math::

    \\hat a_\\lambda = \\arg\\min_a \\tfrac12 a^\\top G a - c^\\top a
                      + \\lambda \\sum_k p_k |a_k|,

which is exactly lassosum's problem with the ``m`` SNP effects replaced by ``K``
score coefficients and the SNP LD matrix replaced by the score covariance. It is
*not* variant-level lassosum: ``||a||_1`` is not ``||W a||_1``, so sparsity here
selects whole component scores while the final per-variant weights ``W a`` are
constrained to their span.

Two things are tuned, and separately:

``lambda``
    How many scores survive. Selected by ``tune=``.

``ld_shrinkage`` (``delta``)
    ``G_delta = G + delta P``, with ``P`` the indicator of the penalized scores.
    This is lassosum's separate LD-stabilisation parameter, and it is what makes
    a near-collinear panel or an imperfect LD reference tractable. It is
    deliberately not folded into the elastic-net ``alpha``, which would tie the
    ridge to the lasso; and leaving forced scores out of ``P`` means a baseline
    score held in the model is not shrunk by the repair applied to the rest.

**Model selection is the hard part, and it has three regimes** (see
:data:`REGIMES`). With individuals, ``multi_pgs_fit`` uses Cross-Model Selection
and Averaging on held-out people. Here there are none, so:

``tune="none"``
    Summary MSE on the same ``z`` that was fitted. This is **regime C** and is
    optimistic because it reuses the fitting moments. Components of ``c`` in
    the LD Gram's numerical nullspace are projected out before fitting: the
    same Gram cannot identify or evaluate them.
``tune="independent"``
    A second, genuinely independent GWAS via ``z_valid``. It is an honest
    **regime B tuning set**, but it is no longer independent after choosing the
    best path point on it. Its ``c_valid`` is projected onto its own Gram's
    identifiable range before selection, and the training ``c`` is projected
    onto that same range before fitting so the returned coefficients cannot
    chase directions the tuning LD cannot assess. Assess the fixed result on a
    third GWAS for regime A.
``tune="pumas"``
    Score-space subsampling of the single available GWAS, needing ``n_eff``.
    **Regime B** pseudovalidation. The component weights must have been learned
    independently of that GWAS; otherwise holding them fixed leaks the
    pseudo-validation participants back into the fit. As in ``tune="none"``,
    the full and pseudo-split cross-moments are projected onto ``range(G)``.
    See
    :func:`subsample_score_moments`.

lassosum's own local-FDR pseudovalidation is deliberately *not* inherited: its
authors later reported it insufficiently robust for an automatic mode (`Privé
et al. 2022 <https://doi.org/10.1016/j.xhgg.2022.100136>`_). The
LD-stabilisation parameter is kept; the tuning criterion is not.

``log["regime"]`` records which of the three produced any given number, so one
cannot later be mistaken for another.

**What is still impossible from ordinary marginal summaries.** Exact logistic
fitting, arbitrary covariate adjustment, AUC, and any participant-level
bootstrap. The Gaussian moment identity remains exact after joint linear
residualization, but ordinary adjusted GWAS output does not reveal enough to
reconstruct a different adjustment. The other quantities need individuals, and
no amount of summary-statistic algebra recovers them.

**What must line up.** Each matrix must be aligned within its own source:
``W_gwas`` to ``z`` and ``W_ld`` to ``D``, with weights and effects counting the
same allele. The two sources need not contain the same variants or row order.
:func:`multipgs.catalog.harmonize_scoring_file` and ``ldpred3.harmonize`` do the
alignment; :func:`align_to_reference` composes them for a panel of scoring files.
``z`` must be on the standardized (allele-correlation) scale, which is what
``ldpred3.standardize_betas(beta, se, n_eff)`` returns — raw per-allele GWAS
betas are on a different scale and will silently produce wrong weights.

**The LD reference must match the target ancestry**, and its sample size bounds
what any of this can resolve. Substituting a discovery-ancestry reference
estimates accuracy in the discovery ancestry, not the target — the whole point
of ``ppb``'s cross-ancestry work, and the same trap here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _coord
from ._numba import warn_no_numba
from ._align import align_to_reference
from ._evaluate import REGIMES, SumstatEval, evaluate_sumstat
from ._gram import (
    _collapse_parsed_weights,
    _ld_variant_count,
    _parsed_weight_digest,
    _parsed_weight_info,
    _score_cross_moment,
    _score_cross_moment_parsed,
    _score_gram_from_coo,
    _weight_columns,
    score_gram,
    score_moments,
)
from ._moments import (
    _bounded_path_mask,
    _boundedness_context,
    _pseudo_r2_batch,
    _range_basis,
    _range_basis_from_factor,
    _selection_candidates_valid,
    _symmetrized,
    _validate_moments,
    pseudo_r2,
)
from ._pumas import (
    _draw_subsample_score_moments,
    _prepare_subsample_score_moments,
    subsample_score_moments,
)
from ._validate import _positive_integer

__all__ = ["multi_pgs_sumstats", "SumstatFit", "score_gram", "score_moments",
           "pseudo_r2", "align_to_reference", "evaluate_sumstat",
           "SumstatEval", "REGIMES", "subsample_score_moments"]

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

@dataclass
class SumstatFit:
    """A multi-PGS combination learned without individual-level data.

    ``beta`` is on the **raw score scale**, matching
    :class:`multipgs.MultiPGSFit`: ``scores @ beta`` is the combination applied
    to new individuals. ``beta_std`` gives the same coefficients on the
    standardized-score scale used by the solver. Likewise, ``path`` is raw and
    ``path_std`` is standardized. This scale contract lets
    :func:`multipgs.combine_weights` consume either fit type safely.

    ``gram`` and ``r`` are the raw-scale moments actually fitted. Same-Gram
    tuning projects ``r`` onto the training Gram's range; independent tuning
    projects it onto the tuning Gram's range. ``c_raw`` retains the observed,
    unprojected cross-moment for provenance and same-data detection.
    """

    beta: np.ndarray
    score_ids: np.ndarray
    score_sd: np.ndarray
    gram: np.ndarray
    r: np.ndarray
    lambdas: np.ndarray
    path: np.ndarray
    lambda_index: int
    alpha: float
    pseudo_r2: float
    pseudo_r2_path: np.ndarray
    selection_mse: float
    selection_mse_path: np.ndarray
    c_raw: np.ndarray = None
    log: dict = field(default_factory=dict)
    weights_ld_digest: str = None
    n_variants_ld: int = None

    @property
    def n_selected(self):
        return int(np.sum(self.beta != 0.0))

    @property
    def raw_beta(self):
        """Compatibility alias for :attr:`beta`, now always on the raw scale."""
        return self.beta

    @property
    def beta_std(self):
        """Combination coefficients on the solver's standardized-score scale."""
        return self.beta * self.score_sd

    @property
    def path_std(self):
        """The fitted path on the solver's standardized-score scale."""
        return self.path * self.score_sd[None, :]

    @property
    def gram_std(self):
        """Score covariance on the standardized coordinates used by the solver."""
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(self.score_sd > 0.0, 1.0 / self.score_sd, 0.0)
        out = self.gram * np.outer(scale, scale)
        dead = self.score_sd <= 0.0
        out[dead, dead] = 1.0
        return out

    @property
    def r_std(self):
        """Score-trait covariance on standardized score coordinates."""
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(self.score_sd > 0.0, 1.0 / self.score_sd, 0.0)
        return self.r * scale

    def evaluate(self, c_eval, gram_eval, *, var_y=1.0, regime=None):
        """Score this fit against evaluation moments from :func:`score_moments`.

        ``c_eval``, ``gram_eval`` and :attr:`beta` are all on the raw score
        scale. Both the observed training moment :attr:`c_raw` and its
        identifiable projection :attr:`r` are recognized as same-data regime
        C inputs.
        """
        c_eval_array = np.asarray(c_eval, dtype=float)
        fitted_on = self.c_raw
        if (c_eval_array.shape == np.asarray(self.r).shape
                and np.array_equal(c_eval_array, np.asarray(self.r))):
            fitted_on = self.r
        return evaluate_sumstat(self.beta, c_eval, gram_eval, var_y=var_y,
                                regime=regime, fitted_on=fitted_on)

    def multi_pgs(self, scores, *, score_ids=None):
        """Apply the combination to an individual-level score matrix.

        The bridge back to :func:`multipgs.evaluate`: the weights were learned
        from summary statistics, but measuring them still needs a cohort, and
        this is how the two meet. ``scores`` must be raw (unstandardized) score
        columns in this fit's order.
        """
        if hasattr(scores, "scores") and hasattr(scores, "score_ids"):
            if score_ids is None:
                score_ids = scores.score_ids
            scores = scores.scores
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != self.beta.size:
            raise ValueError(f"scores must have {self.beta.size} columns, got "
                             f"{scores.shape}")
        if score_ids is not None:
            got = [str(s) for s in np.asarray(score_ids, dtype=object)]
            want = [str(s) for s in np.asarray(self.score_ids, dtype=object)]
            if got != want:
                raise ValueError(
                    "these scores are not the ones this fit was trained on, or "
                    "are in a different order. Realign with "
                    "panel.select(list(fit.score_ids)).")
        return scores @ self.beta

    @property
    def weight_digest(self):
        """Compatibility alias for the frozen LD-basis matrix digest."""
        return self.weights_ld_digest

    @property
    def n_variants(self):
        """Compatibility alias for the frozen LD-basis variant count."""
        return self.n_variants_ld

    def frozen_variant_weights(self, weights_ld, *, n_variants_ld=None):
        """Collapse the stack on the frozen LD standardized-genotype basis.

        ``weights_ld`` must be exactly the LD-basis matrix given to the fit.
        The result is a bare vector in that matrix's variant order and scale.
        It is usable only together with the LD source's variant and genotype-SD
        metadata; it is **not** a raw allele-count deployment vector. Use
        :func:`multipgs.combine_weights` on the raw :class:`multipgs.ScorePanel`
        to obtain that.
        """
        if n_variants_ld is None:
            n_variants_ld = self.n_variants_ld
        parsed = _weight_columns(weights_ld, n_variants_ld)
        m, k, _ = _parsed_weight_info(parsed)
        if k != self.beta.size:
            raise ValueError(f"weights describe {k} scores but this fit has "
                             f"{self.beta.size}")
        if self.weights_ld_digest is not None:
            got = _parsed_weight_digest(parsed)
            if got != self.weights_ld_digest:
                raise ValueError(
                    "weights differ from the aligned score matrix used to fit "
                    "this combination; variant_weights cannot safely attach "
                    "coefficients to them")
        return _collapse_parsed_weights(parsed, self.beta)

    def variant_weights(self, weights_ld, *, n_variants_ld=None):
        """Compatibility alias for :meth:`frozen_variant_weights`.

        The returned vector is on the LD standardized-genotype basis, not the
        raw allele-count scale used by a scoring file.
        """
        return self.frozen_variant_weights(
            weights_ld, n_variants_ld=n_variants_ld)

    def summary(self):
        lines = [
            f"multi-PGS from summary statistics: {self.n_selected} of "
            f"{self.beta.size} scores selected",
            f"  alpha {self.alpha:g}, lambda "
            f"{self.lambdas[self.lambda_index]:.4g} "
            f"(index {self.lambda_index} of {self.lambdas.size})",
            f"  selection MSE {self.selection_mse:.4f}; descriptive R2 "
            f"{self.pseudo_r2:.4f} ({self.log.get('selection', 'unknown')})",
        ]
        if self.log.get("selection") == "in-sample":
            lines.append("  this R2 is optimistic: lambda was chosen on the "
                         "same z it was fitted on")
        if self.log.get("n_dead"):
            lines.append(f"  {self.log['n_dead']} score(s) have zero variance "
                         "under this LD reference and were dropped")
        if self.log.get("boundedness_warning"):
            lines.append(f"  {self.log['boundedness_warning']}")
        if self.log.get("selection_filter_warning"):
            lines.append(f"  {self.log['selection_filter_warning']}")
        if self.log.get("convergence_warning"):
            lines.append(f"  {self.log['convergence_warning']}")
        if self.log.get("selection_moment_warning"):
            lines.append(f"  {self.log['selection_moment_warning']}")
        return "\n".join(lines)

def _resolve_score_columns(selection, score_ids, n_scores):
    """Resolve an unpenalized score mask, indices, or ids without ambiguity."""
    selected = np.asarray(selection)
    if selected.dtype == bool:
        if selected.shape != (n_scores,):
            raise ValueError(
                f"boolean unpenalized_scores must have length {n_scores}")
        indices = np.flatnonzero(selected)
    elif np.issubdtype(selected.dtype, np.integer):
        indices = np.atleast_1d(selected).astype(int)
        if np.any(indices < 0) or np.any(indices >= n_scores):
            raise ValueError("unpenalized score index out of range")
    else:
        lookup = {str(value): i for i, value in enumerate(score_ids)}
        try:
            indices = np.array(
                [lookup[str(value)] for value in np.atleast_1d(selected)],
                dtype=int)
        except KeyError as exc:
            raise ValueError(
                f"unknown unpenalized score id {exc.args[0]!r}") from None
    if np.unique(indices).size != indices.size:
        raise ValueError("unpenalized_scores contains duplicate scores")
    return indices

def multi_pgs_sumstats(weights_ld, z, ld, *, weights_gwas=None,
                       score_ids=None, alpha=1.0,
                       penalty_factor=None, unpenalized_scores=None,
                       n_lambda=100,
                       lambda_min_ratio=None, z_valid=None, ld_valid=None,
                       weights_gwas_valid=None, weights_ld_valid=None,
                       var_y=1.0, n_variants_ld=None,
                       n_variants_ld_valid=None, tol=1e-7, max_iter=1000,
                       tune="auto", n_eff=None, n_repeats=4,
                       train_fraction=0.75, rng=None, ld_shrinkage=None,
                       weights_independent_of_z=False):
    """Learn a multi-PGS combination from summary statistics alone.

    Parameters
    ----------
    weights_ld : ndarray or sequence of (index, weight)
        Component weights multiplied by the empirical genotype SD of the LD
        source. They define ``G = W_ld.T @ D @ W_ld`` and their exact LD-basis
        representation is bound to the returned fit.
    weights_gwas : ndarray or sequence of (index, weight)
        The same component scores and column order, but multiplied by the
        genotype SD of the GWAS behind ``z``. They define
        ``c = W_gwas.T @ z``. This argument is mandatory even when intentionally
        equal to ``weights_ld``: silently assuming two datasets have identical
        genotype SD is unsafe.
    z : array_like, shape (m,)
        Target-trait marginal effects on the standardized scale, aligned to the
        rows of ``weights_gwas``. Use
        ``ldpred3.standardize_betas(beta, se, n_eff)``;
        passing raw per-allele betas is the single easiest way to get plausible
        and wrong weights out of this function. Exact equivalence to an
        individual-level Gaussian regression additionally requires unadjusted
        moments, or genotypes and phenotype jointly residualized using the same
        covariates; generic covariate-adjusted marginal GWAS coefficients need
        not satisfy that identity.
    ld : sequence of (corr_block, idx), or ndarray
        LD reference **matched to the target ancestry**.
    score_ids : sequence of str, optional
    alpha : float
        Elastic-net mixing, 1 = lasso. Unlike the individual-level fit there is
        no honest way to search this here, so it is a single value.
    penalty_factor : array_like, optional
        Per-score penalty factors, e.g. from
        :func:`multipgs.penalty_from_accuracy`. Zero forces a score in.
    unpenalized_scores : sequence, optional
        Score indices, ids, or a boolean mask to leave unpenalized. This is the
        explicit baseline-score convenience used by :func:`multi_pgs_fit`.
    z_valid : array_like, optional
        Marginal effects from an **independent** GWAS of the same trait. When
        given, ``lambda`` is selected against these instead of against ``z``,
        making it an honest tuning criterion. Because it chooses the model, it
        is not a clean assessment; evaluate the fixed fit on a third GWAS.
    ld_valid : optional
        LD reference for ``z_valid``. Independent tuning also requires its own
        explicit ``weights_gwas_valid`` and ``weights_ld_valid``.
    var_y : float
        Phenotype variance on the scale ``z`` was formed on. 1.0 for
        standardized effects.
    weights_independent_of_z : bool
        Required acknowledgement for ``tune="pumas"``. The component weights
        must not have been trained or selected using the GWAS behind ``z``;
        PUMAS holds them fixed and therefore cannot remove that leakage.

    Notes
    -----
    With ``tune="none"`` or ``tune="pumas"``, the same LD Gram supplies both
    fitting and selection. The observed ``c`` is therefore projected onto
    ``range(G)`` before fitting and before every PUMAS pseudo-split; discarded
    norm and fraction are recorded in ``fit.log``. Independent tuning retains
    training directions unresolved by the training Gram only when its distinct
    tuning Gram can identify them: both training and tuning ``c`` are projected
    onto the tuning Gram's range before fitting and selection.

    Returns
    -------
    SumstatFit
    """
    warn_no_numba()
    if weights_gwas is None:
        raise ValueError(
            "weights_gwas is required separately from weights_ld: c uses GWAS "
            "genotype SD while G uses LD-reference genotype SD. Pass the same "
            "matrix explicitly only when those scales are genuinely identical")
    z = np.asarray(z, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")
    n_variants_ld = _ld_variant_count(
        weights_ld, ld, n_variants_ld, "n_variants_ld")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    n_lambda = _positive_integer(n_lambda, "n_lambda")
    max_iter = _positive_integer(max_iter, "max_iter")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    if lambda_min_ratio is not None:
        lambda_min_ratio = float(lambda_min_ratio)
        if (not np.isfinite(lambda_min_ratio)
                or not 0.0 < lambda_min_ratio <= 1.0):
            raise ValueError("lambda_min_ratio must be finite and lie in (0, 1]")
    if tune == "auto":
        tune = ("independent" if z_valid is not None
                else "pumas" if n_eff is not None else None)
        if tune is None:
            raise ValueError(
                "tune='auto' has no tuning data: give z_valid, give n_eff for "
                "PUMAS, or explicitly pass tune='none' to accept optimistic "
                "same-GWAS selection")
    if tune not in ("none", "independent", "pumas"):
        raise ValueError("tune must be 'auto', 'none', 'independent' or "
                         f"'pumas', got {tune!r}")
    valid_inputs = {"z_valid": z_valid, "ld_valid": ld_valid,
                    "weights_gwas_valid": weights_gwas_valid,
                    "weights_ld_valid": weights_ld_valid,
                    "n_variants_ld_valid": n_variants_ld_valid}
    supplied_valid = [name for name, value in valid_inputs.items()
                      if value is not None]
    if tune in ("none", "pumas") and supplied_valid:
        raise ValueError(f"tune={tune!r} does not use {', '.join(supplied_valid)}")
    if tune != "pumas" and n_eff is not None:
        raise ValueError(f"tune={tune!r} does not use n_eff")
    if tune == "independent":
        missing = [name for name in ("z_valid", "ld_valid",
                                     "weights_gwas_valid", "weights_ld_valid")
                   if valid_inputs[name] is None]
        if missing:
            raise ValueError("tune='independent' requires " + ", ".join(missing))
    if tune == "pumas":
        if n_eff is None:
            raise ValueError(
                "tune='pumas' needs n_eff, the effective sample size of the "
                "GWAS behind z")
        if (not isinstance(weights_independent_of_z, (bool, np.bool_))
                or not bool(weights_independent_of_z)):
            raise ValueError(
                "tune='pumas' holds component weights fixed, so it requires "
                "weights_independent_of_z=True to acknowledge that no score "
                "was trained or selected using the GWAS behind z")
        if not np.isfinite(float(n_eff)) or float(n_eff) <= 0.0:
            raise ValueError("n_eff must be finite and strictly positive")
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must lie strictly in (0, 1)")
        n_repeats = _positive_integer(n_repeats, "n_repeats")

    # Parsed once and used for both the Gram and the fit's weight digest. Sparse
    # panels become canonical COO arrays; dense panels remain dense, avoiding
    # three genome-wide arrays per non-zero entry.
    parsed_ld = _weight_columns(weights_ld, n_variants_ld)
    m_ld, k_ld, n_weight_entries_ld = _parsed_weight_info(parsed_ld)
    gram_raw, score_var = _score_gram_from_coo(parsed_ld, ld)
    k = gram_raw.shape[0]
    if k_ld != k:
        raise RuntimeError("parsed weight columns do not match their Gram")

    if weights_gwas is weights_ld:
        wz, n_weight_entries_gwas, m_gwas = _score_cross_moment_parsed(
            parsed_ld, z, k, "weights_gwas",
            n_entries=n_weight_entries_ld)
    else:
        wz, n_weight_entries_gwas, m_gwas = _score_cross_moment(
            weights_gwas, z, k, "weights_gwas")
    gram_moment_cache = {}
    gram_raw, gram_factor_raw, coherence = _validate_moments(
        wz, gram_raw, var_y, label="fitting", prepared=gram_moment_cache)
    score_var = np.diag(gram_raw).copy()

    if score_ids is None:
        score_ids = np.array([f"score_{i}" for i in range(k)], dtype=object)
    else:
        score_ids = np.asarray(list(score_ids), dtype=object)
        if score_ids.size != k:
            raise ValueError(f"score_ids has {score_ids.size} entries for {k} "
                             "scores")
        if len(set(map(str, score_ids))) != k:
            raise ValueError("score_ids must be unique")

    # A score with no variance under this reference cannot enter a regression:
    # its column of the Gram is zero, so the coordinate update divides by zero.
    # Dropping it explicitly is better than letting it produce a nan that
    # propagates into every other coefficient.
    dead = ~(score_var > 0.0)
    score_sd = np.sqrt(np.maximum(score_var, 0.0))
    safe_sd = np.where(dead, 1.0, score_sd)
    scale = np.where(dead, 0.0, 1.0 / safe_sd)
    gram = gram_raw * np.outer(scale, scale)
    gram[dead, :] = 0.0
    gram[:, dead] = 0.0
    gram[dead, dead] = 1.0          # keep the diagonal usable; beta stays zero
    r_observed = wz * scale
    r = r_observed.copy()

    pf = np.ones(k) if penalty_factor is None else \
        np.asarray(penalty_factor, dtype=float).ravel()
    if pf.size != k:
        raise ValueError(f"penalty_factor has {pf.size} entries for {k} scores")
    if np.any(pf < 0) or not np.all(np.isfinite(pf)):
        raise ValueError("penalty_factor must be finite and non-negative")
    # A dead score has r = 0 and a zero off-diagonal Gram row, so the soft
    # threshold leaves its coefficient at zero for every lambda; it needs no
    # separate handling in the penalty vector.
    pf = pf.copy()
    if unpenalized_scores is not None:
        pf[_resolve_score_columns(unpenalized_scores, score_ids, k)] = 0.0

    # Equation (3): G_delta = G + delta * P, with P the indicator of the
    # PENALIZED scores. Stabilising a near-collinear panel and an imperfect LD
    # reference is a separate job from selecting scores, so delta gets its own
    # grid rather than being folded into the elastic-net alpha, which would tie
    # the ridge to the lasso. Leaving forced scores out of P means a baseline
    # score held in the model is not shrunk by the repair applied to the rest.
    penalized = (pf > 0.0).astype(float)
    if not np.any(penalized):
        raise ValueError("every score is unpenalized; nothing to select over")
    deltas = np.atleast_1d(np.asarray(
        [0.0] if ld_shrinkage is None else ld_shrinkage, dtype=float)).ravel()
    if deltas.size == 0 or np.any(deltas < 0) or not np.all(np.isfinite(deltas)):
        raise ValueError("ld_shrinkage must be finite and non-negative")
    solver_factor = gram_factor_raw * scale[:, None]
    if np.any(dead):
        solver_factor = np.hstack([solver_factor, np.eye(k)[:, dead]])
    boundedness = _boundedness_context(
        gram, pf, base_basis=_range_basis_from_factor(solver_factor))
    if tune != "independent":
        # The same Gram supplies both the fit and selection criterion. Signal in
        # its exact nullspace is not estimable; a positive delta must stabilize
        # resolved small-eigenvalue directions, not invent zero-variance ones.
        range_vectors = boundedness["base_basis"][0]
        r = range_vectors @ (range_vectors.T @ r_observed)
    discarded_r = r_observed - r
    discarded_r_norm = float(np.linalg.norm(discarded_r))
    discarded_r_fraction = discarded_r_norm / max(
        float(np.linalg.norm(r_observed)), np.finfo(float).tiny)
    c_fit_raw = r * score_sd

    splits = None
    sel_gram = sel_r = None
    if tune == "pumas":
        generator = np.random.default_rng(rng)
        n_train = float(train_fraction) * float(n_eff)
        # Drawn once and reused across the delta grid, so the comparison
        # between delta values is not confounded by different split noise. The
        # O(K^3) PSD factor is also computed once, not once per repeat.
        split_factor, split_base_log = _prepare_subsample_score_moments(
            c_fit_raw, gram_raw, var_y, check=True, factor=gram_factor_raw)
        splits = [_draw_subsample_score_moments(
                      c_fit_raw, n_eff, n_train, var_y, split_factor,
                      split_base_log,
                      generator)
                  for _ in range(n_repeats)]
        split_log = splits[0][2]
        selection = "PUMAS pseudo-split"
    elif tune == "independent":
        z_valid = np.asarray(z_valid, dtype=float).ravel()
        if not np.all(np.isfinite(z_valid)):
            raise ValueError("z_valid contains non-finite values")
        n_variants_ld_valid = _ld_variant_count(
            weights_ld_valid, ld_valid, n_variants_ld_valid,
            "n_variants_ld_valid")
        reuse_validation_ld = (
            weights_ld_valid is weights_ld and ld_valid is ld
            and n_variants_ld_valid == n_variants_ld)
        if reuse_validation_ld:
            parsed_ld_valid = parsed_ld
            gram_v = gram_raw
        else:
            parsed_ld_valid = _weight_columns(
                weights_ld_valid, n_variants_ld_valid)
            gram_v = _score_gram_from_coo(parsed_ld_valid, ld_valid)[0]
        m_ld_valid, k_valid, n_weight_entries_ld_valid = (
            _parsed_weight_info(parsed_ld_valid))
        if k_valid != k:
            raise ValueError(
                f"weights_ld_valid describes {k_valid} scores but weights_ld "
                f"describes {k}; score identity and column order must agree")
        if weights_gwas_valid is weights_ld_valid:
            (wz_v, n_weight_entries_gwas_valid,
             m_gwas_valid) = _score_cross_moment_parsed(
                 parsed_ld_valid, z_valid, k, "weights_gwas_valid",
                 n_entries=n_weight_entries_ld_valid)
        else:
            (wz_v, n_weight_entries_gwas_valid,
             m_gwas_valid) = _score_cross_moment(
                 weights_gwas_valid, z_valid, k, "weights_gwas_valid")
        gram_v, _, validation_coherence = _validate_moments(
            wz_v, gram_v, var_y, label="tuning",
            prepared=(gram_moment_cache if reuse_validation_ld else None))
        # The TRAINING scale, deliberately. beta comes off a path fitted in
        # training-standardized coordinates, so both halves of
        # (beta'r)^2 / (beta'G beta) must use those same coordinates.
        sel_gram = gram_v * np.outer(scale, scale)
        sel_r_observed = wz_v * scale
        # Multiplying the tuning factor by the TRAINING scale generally
        # destroys orthogonality between its columns.  Build the range basis
        # from the resulting tuning Gram itself; column-normalising that
        # transformed factor would make B B' a non-idempotent non-projection.
        tuning_basis = (boundedness["base_basis"][0]
                        if reuse_validation_ld else _range_basis(sel_gram)[0])
        sel_r = tuning_basis @ (tuning_basis.T @ sel_r_observed)
        training_r_before_tuning_projection = r.copy()
        r = tuning_basis @ (tuning_basis.T @ r)
        c_fit_raw = r * score_sd
        tuning_discarded_training = training_r_before_tuning_projection - r
        tuning_discarded_training_norm = float(
            np.linalg.norm(tuning_discarded_training))
        tuning_discarded_training_fraction = (
            tuning_discarded_training_norm / max(
                float(np.linalg.norm(training_r_before_tuning_projection)),
                np.finfo(float).tiny))
        tuning_discarded = sel_r_observed - sel_r
        tuning_discarded_norm = float(np.linalg.norm(tuning_discarded))
        tuning_discarded_fraction = tuning_discarded_norm / max(
            float(np.linalg.norm(sel_r_observed)), np.finfo(float).tiny)
        selection = "independent GWAS"
    else:
        sel_gram, sel_r, selection = gram, r, "in-sample"

    # The selection Gram is fixed across the shrinkage grid, so symmetrize it
    # once here rather than once per candidate inside the path scoring.
    sel_prepared = None if sel_gram is None else _symmetrized(sel_gram)

    def _score_path(path_d, gram_fit, delta_value, fit_converged):
        """Descriptive R2 and calibration-sensitive MSE for one fitted path.

        Both metrics use the UNSHRUNK Gram: delta is a fitting device
        for stabilising the solve, not a claim about the score covariance, so
        candidates are judged against the best estimate of the truth rather
        than against the regularised surrogate. Selection minimizes MSE; unlike
        squared correlation, it rejects a coefficient vector with the wrong
        sign or scale.
        """
        if splits is None:
            valid, r2, mse = _selection_candidates_valid(
                path_d, sel_gram, sel_r, var_y, prepared=sel_prepared)
            selectable = valid & fit_converged
            r2 = np.where(selectable, r2, np.nan)
            mse = np.where(selectable, mse, np.nan)
            return (r2, mse, int(np.sum(~valid)),
                    int(np.sum(~fit_converged)), 0)
        n_rep = len(splits)
        n_lam = path_d.shape[0]
        r_trains = np.empty((n_rep, k), dtype=float)
        r_vals = np.empty((n_rep, k), dtype=float)
        bounded_every_repeat = np.ones(n_lam, dtype=bool)
        range_vectors = boundedness["base_basis"][0]
        for rep, (c_tr, c_val, _) in enumerate(splits):
            r_train_observed = c_tr * scale
            r_trains[rep] = range_vectors @ (range_vectors.T @ r_train_observed)
            r_val_observed = c_val * scale
            r_vals[rep] = range_vectors @ (range_vectors.T @ r_val_observed)
            safe_split, _, _ = _bounded_path_mask(
                boundedness, r_trains[rep], pf, alpha, lambdas_d, delta_value)
            bounded_every_repeat &= safe_split[:n_lam]
        # Boundedness is still a per-repeat prefix. Fit only the lambdas that
        # every repeat can certify and whose full-data fit converged, in one
        # compiled walk of the shared Gram.
        per_repeat_r2 = np.full((n_rep, n_lam), np.nan)
        per_repeat_mse = np.full((n_rep, n_lam), np.nan)
        repeat_converged = np.ones(n_lam, dtype=bool)
        safe_indices = np.flatnonzero(bounded_every_repeat & fit_converged)
        n_tuning_exhausted = 0
        if safe_indices.size:
            fitted_paths, n_fitted_rep, batch_info = (
                _coord.enet_path_gaussian_batch(
                    gram_fit, r_trains, pf=pf, alpha=alpha,
                    lambdas=lambdas_d[safe_indices], tol=tol,
                    max_iter=max_iter, return_info=True))
            n_tuning_exhausted = int(
                batch_info["n_iteration_exhausted"])
            for rep in range(n_rep):
                n_ok = min(int(n_fitted_rep[rep]), safe_indices.size)
                if n_ok == 0:
                    repeat_converged[safe_indices] = False
                    continue
                if n_ok < safe_indices.size:
                    repeat_converged[safe_indices[n_ok:]] = False
                use = safe_indices[:n_ok]
                converged_rep = np.asarray(
                    batch_info["converged_path"][rep, :n_ok], dtype=bool)
                repeat_converged[use[~converged_rep]] = False
                split_r2, quadratic = _pseudo_r2_batch(
                    fitted_paths[rep, :n_ok], gram, r_vals[rep], var_y)
                use_converged = use[converged_rep]
                per_repeat_r2[rep, use_converged] = split_r2[converged_rep]
                split_mse = (
                    var_y - 2.0 * (fitted_paths[rep, :n_ok] @ r_vals[rep])
                    + quadratic)
                per_repeat_mse[rep, use_converged] = split_mse[converged_rep]
        selectable = (bounded_every_repeat & fit_converged
                      & repeat_converged)
        per_repeat_r2[:, ~selectable] = np.nan
        per_repeat_mse[:, ~selectable] = np.nan
        r2_mean = np.full(path_d.shape[0], np.nan)
        r2_count = np.sum(np.isfinite(per_repeat_r2), axis=0)
        np.divide(np.nansum(per_repeat_r2, axis=0), r2_count, out=r2_mean,
                  where=r2_count > 0)
        mse_mean = np.full(path_d.shape[0], np.nan)
        mse_mean[selectable] = np.mean(
            per_repeat_mse[:, selectable], axis=0)
        r2_mean[~selectable] = np.nan
        mse_mean[~selectable] = np.nan
        nonconverged = (~fit_converged
                        | (bounded_every_repeat & fit_converged
                           & ~repeat_converged))
        return (r2_mean, mse_mean,
                int(np.sum(~bounded_every_repeat)),
                int(np.sum(nonconverged)), n_tuning_exhausted)

    best_value = np.inf
    best = None
    delta_audit = []
    for delta in deltas:
        gram_d = gram + float(delta) * np.diag(penalized)
        _, grad = _coord.unpenalized_fit(gram_d, r, pf)
        lambdas_d = _coord.lambda_grid(grad, pf, alpha, n_lambda=n_lambda,
                                       lambda_min_ratio=lambda_min_ratio)
        bounded, null_residual, free_null_residual = _bounded_path_mask(
            boundedness, r, pf, alpha, lambdas_d, float(delta))
        n_unsafe = int(np.sum(~bounded))
        if not np.any(bounded):
            delta_audit.append({"delta": float(delta), "n_fitted": 0,
                                "n_converged": 0,
                                "n_iteration_exhausted": 0,
                                "n_tuning_iteration_exhausted": 0,
                                "n_rejected_nonconverged": 0,
                                "n_rejected_unbounded": n_unsafe,
                                "n_rejected_selection_moments": 0,
                                "null_c_norm": null_residual,
                                "free_null_c_norm": free_null_residual,
                                "best_index": None, "best_lambda": None,
                                "selection_mse": None,
                                "descriptive_r2": None})
            continue
        # The conservative certificate is a decreasing-lambda prefix. Unsafe
        # lower-lambda objectives never reach coordinate descent: an unbounded
        # path can otherwise stop at max_iter with finite-looking coefficients.
        lambdas_d = lambdas_d[bounded]
        path_d, n_fitted, fit_info = _coord.enet_path_gaussian(
            gram_d, r, pf=pf, alpha=alpha, lambdas=lambdas_d, tol=tol,
            max_iter=max_iter, return_info=True)
        path_d = path_d[:n_fitted]
        lambdas_d = lambdas_d[:n_fitted]
        fit_converged = np.asarray(
            fit_info["converged_path"][:n_fitted], dtype=bool)
        (r2_d, mse_d, n_selection_invalid, n_nonconverged,
         n_tuning_exhausted) = _score_path(
            path_d, gram_d, float(delta), fit_converged)
        if np.all(np.isnan(mse_d)):
            delta_audit.append({"delta": float(delta),
                                "n_fitted": int(n_fitted),
                                "n_converged": int(np.sum(fit_converged)),
                                "n_iteration_exhausted": int(
                                    fit_info["n_iteration_exhausted"]),
                                "n_tuning_iteration_exhausted":
                                    n_tuning_exhausted,
                                "n_rejected_nonconverged": n_nonconverged,
                                "n_rejected_unbounded": n_unsafe,
                                "n_rejected_selection_moments":
                                    n_selection_invalid,
                                "null_c_norm": null_residual,
                                "free_null_c_norm": free_null_residual,
                                "best_index": None, "best_lambda": None,
                                "selection_mse": None,
                                "descriptive_r2": None})
            continue
        index = int(np.nanargmin(mse_d))
        delta_audit.append({
            "delta": float(delta), "n_fitted": int(n_fitted),
            "n_converged": int(np.sum(fit_converged)),
            "n_iteration_exhausted": int(
                fit_info["n_iteration_exhausted"]),
            "n_tuning_iteration_exhausted": n_tuning_exhausted,
            "n_rejected_nonconverged": n_nonconverged,
            "n_rejected_unbounded": n_unsafe,
            "n_rejected_selection_moments": n_selection_invalid,
            "null_c_norm": null_residual,
            "free_null_c_norm": free_null_residual,
            "best_index": index, "best_lambda": float(lambdas_d[index]),
            "selection_mse": float(mse_d[index]),
            "descriptive_r2": float(r2_d[index])})
        if mse_d[index] < best_value:
            best_value = float(mse_d[index])
            best = (float(delta), index, path_d, lambdas_d, r2_d, mse_d)

    if best is None:
        n_exhausted = sum(
            row["n_iteration_exhausted"]
            + row["n_tuning_iteration_exhausted"] for row in delta_audit)
        if n_exhausted:
            raise RuntimeError(
                "no converged path point was available for selection; increase "
                "max_iter or relax tol. Unconverged coordinate-descent "
                "solutions are never selected.")
        raise ValueError(
            "no bounded objective was available on any requested path. Add a "
            "positive ld_shrinkage, remove an incompatible unpenalized score, "
            "or repair the score/LD/GWAS alignment.")
    delta, index, path, lambdas, scores_path, mse_path = best

    log = {"n_scores": k, "n_variants_ld": m_ld,
           "n_variants_gwas": m_gwas, "selection": selection,
           "n_dead": int(np.sum(dead)), "n_lambda": int(lambdas.size),
           "alpha": float(alpha), "var_y": float(var_y),
           "n_weight_entries_ld": n_weight_entries_ld,
           "n_weight_entries_gwas": n_weight_entries_gwas,
           "ld_shrinkage": delta,
           "n_shrinkage": int(deltas.size), "selection_metric": "MSE",
           "delta_audit": delta_audit,
           "discarded_ld_null_c_norm": discarded_r_norm,
           "discarded_ld_null_c_fraction": discarded_r_fraction,
           "n_rejected_unbounded": int(sum(
               row["n_rejected_unbounded"] for row in delta_audit)),
           "n_rejected_selection_moments": int(sum(
               row["n_rejected_selection_moments"]
               for row in delta_audit)),
           "n_iteration_exhausted_fit": int(sum(
               row["n_iteration_exhausted"] for row in delta_audit)),
           "n_iteration_exhausted_tuning": int(sum(
               row["n_tuning_iteration_exhausted"]
               for row in delta_audit)),
           "n_rejected_nonconverged": int(sum(
               row["n_rejected_nonconverged"] for row in delta_audit)),
           "unpenalized_scores": int(np.count_nonzero(pf <= 0.0))}
    log.update(coherence)
    if log["n_rejected_unbounded"]:
        log["boundedness_warning"] = (
            f"{log['n_rejected_unbounded']} path point(s) were not fitted "
            "because boundedness could not be certified")
    if log["n_rejected_selection_moments"]:
        log["selection_filter_warning"] = (
            f"{log['n_rejected_selection_moments']} path point(s) were "
            "excluded because the plug-in selection moments were physically "
            "impossible")
    if log["n_rejected_nonconverged"]:
        log["convergence_warning"] = (
            f"{log['n_rejected_nonconverged']} path point(s) were excluded "
            "because coordinate descent exhausted max_iter; increase max_iter "
            "or relax tol if this affects the useful part of the path")
    if tune == "pumas":
        log.update({"regime": "B", "selection_role": "tuning",
                    "n_repeats": len(splits),
                    "n_eff": float(n_eff),
                    "train_fraction": float(train_fraction),
                    "split": {kk: vv for kk, vv in split_log.items()
                              if kk not in ("n", "n_train", "n_val", "var_y")}})
        for key in ("warning", "rank_warning"):
            if key in split_log:
                log[key] = split_log[key]
    elif tune == "independent":
        log.update({"regime": "B", "selection_role": "tuning"})
        log.update({"n_variants_ld_valid": m_ld_valid,
                    "n_variants_gwas_valid": m_gwas_valid,
                    "n_weight_entries_ld_valid": n_weight_entries_ld_valid,
                    "n_weight_entries_gwas_valid":
                        n_weight_entries_gwas_valid,
                    "tuning_discarded_ld_null_c_norm":
                        tuning_discarded_norm,
                    "tuning_discarded_ld_null_c_fraction":
                        tuning_discarded_fraction,
                    "training_discarded_by_tuning_ld_c_norm":
                        tuning_discarded_training_norm,
                    "training_discarded_by_tuning_ld_c_fraction":
                        tuning_discarded_training_fraction,
                    "tuning_ld_reused": bool(reuse_validation_ld)})
        log.update({f"tuning_{key}": value
                    for key, value in validation_coherence.items()})
    else:
        log.update({"regime": "C", "selection_role": "same-data fit"})
        log["warning"] = ("lambda was selected on the same summary statistics "
                          "it was fitted on, which drives the choice towards "
                          "the least-penalized point on the path; the reported "
                          "descriptive R2 is optimistic")
    if deltas.size > 1 and delta in (deltas.min(), deltas.max()):
        log["shrinkage_warning"] = (
            f"the selected ld_shrinkage {delta:g} is at the edge of the grid; "
            "widen it, since the optimum may lie outside")
    selected_warnings = []
    if np.isfinite(scores_path[index]) and scores_path[index] > 1.0:
        selected_warnings.append(
            f"selected plug-in R2 is {scores_path[index]:.6g}, above 1")
    if not np.isfinite(scores_path[index]):
        selected_warnings.append(
            "selected plug-in R2 is undefined under the unshrunk LD moments")
    if mse_path[index] < 0.0:
        selected_warnings.append(
            f"selected plug-in MSE is {mse_path[index]:.6g}, below zero")
    if selected_warnings:
        log["selection_moment_warning"] = (
            "; ".join(selected_warnings)
            + "; finite-sample external moments are noisy, so these are "
              "diagnostics rather than population constraints")
    path_raw = path * scale[None, :]
    return SumstatFit(
        beta=path_raw[index].copy(), score_ids=score_ids, score_sd=score_sd,
        gram=gram_raw, r=c_fit_raw.copy(), lambdas=lambdas, path=path_raw,
        lambda_index=index,
        alpha=float(alpha), pseudo_r2=float(scores_path[index]),
        pseudo_r2_path=scores_path, selection_mse=float(mse_path[index]),
        selection_mse_path=mse_path, c_raw=wz.copy(), log=log,
        weights_ld_digest=_parsed_weight_digest(parsed_ld),
        n_variants_ld=m_ld)
