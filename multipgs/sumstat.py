"""Fit the multi-PGS combination from summary statistics alone.

:func:`multipgs.multi_pgs_fit` needs a training cohort: genotypes, a phenotype,
and enough individuals to cross-validate over. That requirement is what stops
most people using multi-PGS, and it is not actually necessary. The stacking
regression of a phenotype on ``K`` scores depends on the individual-level data
only through two sufficient statistics,

.. math::

    G = X^\\top X / n, \\qquad r = X^\\top y / n,

and both are available from summary-level data. Writing ``W`` for the
``m x K`` matrix of per-variant score weights, ``D`` for the LD correlation
matrix and ``z`` for the target trait's standardized marginal effects,

.. math::

    W^\\top D W = n^{-1} X^\\top X, \\qquad W^\\top z = n^{-1} X^\\top y,

because score ``k`` evaluated on standardized genotypes is ``g W_{\\cdot k}``.
So the K-by-K score covariance comes from an **LD reference** and the K-vector
of score-phenotype covariances comes from a **GWAS of the target trait**. No
individuals, no phenotypes, no genotypes.

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
    The pseudovalidation criterion on the same ``z`` that was fitted. This is
    **regime C** and it barely selects at all — measured on a 200-score panel it
    lands on the least-penalized point of the path every time. Reported for
    completeness, not for use.
``tune="independent"``
    A second, genuinely independent GWAS via ``z_valid``. **Regime A.**
``tune="pumas"``
    Score-space subsampling of the single available GWAS, needing ``n_eff``.
    **Regime B**, and in measurement it beats regime A while using one GWAS
    rather than two. See :func:`subsample_score_moments`.

lassosum's own local-FDR pseudovalidation is deliberately *not* inherited: its
authors later reported it insufficiently robust for an automatic mode (`Privé
et al. 2022 <https://doi.org/10.1016/j.xhgg.2022.100136>`_). The
LD-stabilisation parameter is kept; the tuning criterion is not.

``log["regime"]`` records which of the three produced any given number, so one
cannot later be mistaken for another.

**What is still impossible from ordinary marginal summaries.** Exact logistic
fitting, covariate adjustment, AUC, and any participant-level bootstrap. Those
need individuals, and no amount of summary-statistic algebra recovers them.

**What must line up.** ``W``, ``z`` and ``D`` must all be on the same variants,
in the same order, with weights and effects counting the same allele.
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


__all__ = ["multi_pgs_sumstats", "SumstatFit", "score_gram", "score_moments",
           "pseudo_r2", "align_to_reference", "evaluate_sumstat",
           "SumstatEval", "REGIMES", "subsample_score_moments"]


#: What the evaluation statistics were computed against, and what each is worth.
#:
#: The distinction is the whole difference between a publishable number and a
#: meaningless one, and it is not visible in the number itself — a regime C R²
#: looks exactly like a regime A R², only larger. Every evaluation this module
#: returns carries its regime, and :func:`evaluate_sumstat` determines it from
#: the inputs rather than taking the caller's word for it.
REGIMES = {
    "A": "fitted on one GWAS, evaluated on an independent GWAS "
         "(clean external validation)",
    "B": "fitted on a PUMAS pseudo-training split, evaluated on its paired "
         "pseudo-validation split (summary-statistic pseudovalidation)",
    "C": "fitted and evaluated on the same unsplit GWAS "
         "(optimistically biased; not a validation)",
}


# ---------------------------------------------------------------------------
# Gram accumulation
# ---------------------------------------------------------------------------

def _as_blocks(ld, n_variants):
    """Normalize an LD argument to a list of ``(corr_block, idx)``.

    Accepts ldpred3's native block list, a single dense correlation matrix, or
    anything ``ldpred3.ld_matmul`` understands paired with explicit indices.
    """
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"a dense LD matrix must be square, got {ld.shape}")
        if ld.shape[0] != n_variants:
            raise ValueError(f"LD matrix is {ld.shape[0]} x {ld.shape[0]} but "
                             f"the weights cover {n_variants} variants")
        return [(ld, np.arange(n_variants))]

    blocks = list(ld)
    if not blocks:
        raise ValueError("the LD reference has no blocks")
    out = []
    covered = 0
    for b, block in enumerate(blocks):
        if not (isinstance(block, tuple) and len(block) == 2):
            raise ValueError(
                f"LD block {b} is not a (corr_block, idx) pair; this is the "
                "layout ldpred3.compute_ld_blocks and ldpred3.load_ld_blocks "
                f"return, got {type(block).__name__}")
        corr, idx = block
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            continue
        if np.any(np.diff(idx) != 1):
            raise ValueError(
                f"LD block {b} does not cover a contiguous run of variants. "
                "The streaming Gram slices weights by block range; use "
                "ldpred3.compute_ld_blocks, whose blocks tile 0..m-1.")
        covered += idx.size
        out.append((corr, idx))
    if covered != n_variants:
        raise ValueError(f"the LD blocks cover {covered} variants but the "
                         f"weights cover {n_variants}")
    return out


def _weight_columns(weights, n_variants=None):
    """Normalize weights to ``(coo_variant, coo_score, coo_value, m, K)``.

    Dense ``(m, K)`` input is accepted for small problems. A panel of catalog
    scores is overwhelmingly sparse — most scores carry a few hundred variants
    out of a reference of millions — so a list of per-score ``(index, weight)``
    pairs is the form that scales, and it is what
    :func:`multipgs.catalog.harmonize_scoring_file` already returns.
    """
    if isinstance(weights, np.ndarray):
        if weights.ndim != 2:
            raise ValueError(f"dense weights must be 2-D (m, K), got "
                             f"{weights.shape}")
        m, k = weights.shape
        rows, cols = np.nonzero(weights)
        return rows.astype(np.int64), cols.astype(np.int64), \
            weights[rows, cols].astype(float), m, k

    pairs = list(weights)
    if not pairs:
        raise ValueError("no score weights given")
    idx_parts, col_parts, val_parts = [], [], []
    largest = -1
    for k, pair in enumerate(pairs):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(f"score {k} must be an (index, weight) pair, got "
                             f"{type(pair).__name__}")
        idx = np.asarray(pair[0], dtype=np.int64).ravel()
        val = np.asarray(pair[1], dtype=float).ravel()
        if idx.shape != val.shape:
            raise ValueError(f"score {k} has {idx.size} indices and {val.size} "
                             "weights")
        if idx.size and not np.all(np.isfinite(val)):
            raise ValueError(f"score {k} has non-finite weights")
        if idx.size:
            largest = max(largest, int(idx.max()))
            if int(idx.min()) < 0:
                raise ValueError(f"score {k} has a negative variant index")
        idx_parts.append(idx)
        col_parts.append(np.full(idx.size, k, dtype=np.int64))
        val_parts.append(val)

    m = int(largest + 1) if n_variants is None else int(n_variants)
    if m <= largest:
        raise ValueError(f"a weight indexes variant {largest} but the reference "
                         f"has {m}")
    return (np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64),
            np.concatenate(col_parts) if col_parts else np.zeros(0, np.int64),
            np.concatenate(val_parts) if val_parts else np.zeros(0),
            m, len(pairs))


def score_gram(weights, ld, *, n_variants=None):
    """The ``K x K`` score covariance ``W^T D W`` from an LD reference.

    Streams the reference one LD block at a time, densifying only that block's
    slice of the weights, so peak memory is ``O(block_size * K)`` rather than
    ``O(m * K)``. For a 900-score panel and 500-variant blocks that is a few
    megabytes instead of tens of gigabytes.

    Parameters
    ----------
    weights : ndarray or sequence of (index, weight)
        Per-variant weights for each score, on the **standardized** genotype
        scale, aligned to the LD reference's variants. PGS Catalog weights count
        raw alleles and must be converted first — see :func:`align_to_reference`.
    ld : sequence of (corr_block, idx), or ndarray
        ldpred3 LD blocks, or one dense correlation matrix.
    n_variants : int, optional
        Reference size, needed only when the weights are sparse and no score
        touches the last variant.

    Returns
    -------
    (gram, score_var) : (ndarray, ndarray)
        ``gram`` is ``W^T D W`` (``K x K``); ``score_var`` is its diagonal, the
        variance of each score under the reference's LD.
    """
    from ldpred3 import ld_matmul

    rows, cols, vals, m, k = _weight_columns(weights, n_variants)
    blocks = _as_blocks(ld, m)

    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]

    gram = np.zeros((k, k), dtype=float)
    for corr, idx in blocks:
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        start, stop = np.searchsorted(rows, (lo, hi))
        if start == stop:
            continue
        block_w = np.zeros((idx.size, k), dtype=float)
        block_w[rows[start:stop] - lo, cols[start:stop]] = vals[start:stop]
        # ld_matmul returns float64 even for an int8 or float32 block, so a
        # compact reference does not quietly lower the precision of the Gram.
        gram += block_w.T @ np.asarray(ld_matmul(corr, block_w), dtype=float)

    # W^T D W is symmetric in exact arithmetic; the accumulation is not, and an
    # asymmetric Gram makes the coordinate descent's covariance updates drift.
    gram = 0.5 * (gram + gram.T)
    return gram, np.diag(gram).copy()


def pseudo_r2(beta, gram, r, *, var_y=1.0):
    """Summary-statistic accuracy of the combined score ``W beta``.

    ``(beta^T r)^2 / (beta^T G beta * var_y)`` — the stack's version of
    ``ppb``'s ``R^2 = (w^T z)^2 / (w^T D w)``. Returns ``nan`` for an all-zero
    ``beta`` (nothing is predicted, so the ratio is undefined rather than zero)
    and raises when the denominator is negative, which only a non-PSD LD
    approximation can produce and which would understate the error silently.
    """
    beta = np.asarray(beta, dtype=float)
    gram = np.asarray(gram, dtype=float)
    r = np.asarray(r, dtype=float)
    num = float(beta @ r)
    den = float(beta @ gram @ beta)
    if den < 0.0:
        raise ValueError(
            f"beta^T G beta = {den!r} is negative, so the LD reference is not "
            "positive semi-definite here. Rebuild it with a ridge "
            "(ldpred3.compute_ld_blocks(..., ridge=...)) rather than trusting "
            "the accuracy this would report.")
    if den == 0.0:
        return float("nan")
    return (num * num) / (den * float(var_y))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

@dataclass
class SumstatFit:
    """A multi-PGS combination learned without individual-level data.

    ``beta`` is on the standardized-score scale, matching
    :class:`multipgs.MultiPGSFit`: score ``k`` is divided by ``score_sd[k]``
    before ``beta[k]`` is applied. :meth:`variant_weights` folds that back into
    one per-variant weight vector, which is the artefact you deploy.
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
    c_raw: np.ndarray = None
    log: dict = field(default_factory=dict)

    @property
    def n_selected(self):
        return int(np.sum(self.beta != 0.0))

    @property
    def raw_beta(self):
        """``beta`` rescaled to apply to unstandardized score columns.

        The fit is on standardized scores; ``c`` and ``G`` from
        :func:`score_moments` are not. This is the vector that pairs with them,
        and keeping the conversion in one place is what stops the two scales
        being mixed at an evaluation call site.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.score_sd > 0.0, self.beta / self.score_sd, 0.0)

    def evaluate(self, c_eval, gram_eval, *, var_y=1.0, regime=None):
        """Score this fit against evaluation moments from :func:`score_moments`.

        ``c_eval`` and ``gram_eval`` are on the raw score scale, and the
        rescaling of ``beta`` to match is done here rather than by the caller.
        """
        return evaluate_sumstat(self.raw_beta, c_eval, gram_eval, var_y=var_y,
                                regime=regime, fitted_on=self.c_raw)

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
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(self.score_sd > 0.0, self.beta / self.score_sd,
                              0.0)
        return scores @ scaled

    def variant_weights(self, weights, *, n_variants=None):
        """Collapse the stack into one per-variant weight vector ``W S^-1 beta``.

        ``weights`` is the same argument given to the fit. The result is on the
        standardized genotype scale, so it goes straight to
        ``ldpred3.score_from_weights(..., scaling="frozen")``.
        """
        rows, cols, vals, m, k = _weight_columns(weights, n_variants)
        if k != self.beta.size:
            raise ValueError(f"weights describe {k} scores but this fit has "
                             f"{self.beta.size}")
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(self.score_sd > 0.0, self.beta / self.score_sd,
                              0.0)
        out = np.zeros(m, dtype=float)
        np.add.at(out, rows, vals * scaled[cols])
        return out

    def summary(self):
        lines = [
            f"multi-PGS from summary statistics: {self.n_selected} of "
            f"{self.beta.size} scores selected",
            f"  alpha {self.alpha:g}, lambda "
            f"{self.lambdas[self.lambda_index]:.4g} "
            f"(index {self.lambda_index} of {self.lambdas.size})",
            f"  pseudovalidation R2 {self.pseudo_r2:.4f} "
            f"({self.log.get('selection', 'unknown')})",
        ]
        if self.log.get("selection") == "in-sample":
            lines.append("  this R2 is optimistic: lambda was chosen on the "
                         "same z it was fitted on")
        if self.log.get("n_dead"):
            lines.append(f"  {self.log['n_dead']} score(s) have zero variance "
                         "under this LD reference and were dropped")
        return "\n".join(lines)


def multi_pgs_sumstats(weights, z, ld, *, score_ids=None, alpha=1.0,
                       penalty_factor=None, n_lambda=100,
                       lambda_min_ratio=None, z_valid=None, ld_valid=None,
                       var_y=1.0, n_variants=None, tol=1e-7, max_iter=1000,
                       tune="auto", n_eff=None, n_repeats=4,
                       train_fraction=0.75, rng=None, ld_shrinkage=None):
    """Learn a multi-PGS combination from summary statistics alone.

    Parameters
    ----------
    weights : ndarray or sequence of (index, weight)
        Per-variant weights per score on the standardized genotype scale,
        aligned to the LD reference. See :func:`align_to_reference`.
    z : array_like, shape (m,)
        Target-trait marginal effects on the standardized scale, aligned to the
        same variants. Use ``ldpred3.standardize_betas(beta, se, n_eff)``;
        passing raw per-allele betas is the single easiest way to get plausible
        and wrong weights out of this function.
    ld : sequence of (corr_block, idx), or ndarray
        LD reference **matched to the target ancestry**.
    score_ids : sequence of str, optional
    alpha : float
        Elastic-net mixing, 1 = lasso. Unlike the individual-level fit there is
        no honest way to search this here, so it is a single value.
    penalty_factor : array_like, optional
        Per-score penalty factors, e.g. from
        :func:`multipgs.penalty_from_accuracy`. Zero forces a score in.
    z_valid : array_like, optional
        Marginal effects from an **independent** GWAS of the same trait. When
        given, ``lambda`` is selected against these instead of against ``z``,
        which is the difference between an honest accuracy estimate and an
        optimistic one.
    ld_valid : optional
        LD reference for ``z_valid``; defaults to ``ld``.
    var_y : float
        Phenotype variance on the scale ``z`` was formed on. 1.0 for
        standardized effects.

    Returns
    -------
    SumstatFit
    """
    z = np.asarray(z, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")

    gram_raw, score_var = score_gram(weights, ld, n_variants=n_variants)
    k = gram_raw.shape[0]

    rows, cols, vals, m, _ = _weight_columns(weights, n_variants)
    if z.size != m:
        raise ValueError(f"z covers {z.size} variants but the weights cover "
                         f"{m}; they must be aligned to the same reference")

    wz = np.zeros(k, dtype=float)
    np.add.at(wz, cols, vals * z[rows])

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
    sd = np.sqrt(np.where(dead, 1.0, score_var))
    scale = np.where(dead, 0.0, 1.0 / sd)
    gram = gram_raw * np.outer(scale, scale)
    gram[dead, :] = 0.0
    gram[:, dead] = 0.0
    gram[dead, dead] = 1.0          # keep the diagonal usable; beta stays zero
    r = wz * scale

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

    if tune == "auto":
        tune = ("independent" if z_valid is not None
                else "pumas" if n_eff is not None else "none")
    if tune not in ("none", "independent", "pumas"):
        raise ValueError("tune must be 'auto', 'none', 'independent' or "
                         f"'pumas', got {tune!r}")

    splits = None
    sel_gram = sel_r = None
    if tune == "pumas":
        if n_eff is None:
            raise ValueError(
                "tune='pumas' needs n_eff, the effective sample size of the "
                "GWAS behind z. There is no default: the split scale is set by "
                "it, and for a case/control GWAS the raw total can be several "
                "times n_eff (use ldpred3.n_eff_case_control), which would "
                "inject too little noise and select too small a penalty.")
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must lie strictly in (0, 1)")
        n_repeats = int(n_repeats)
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        generator = np.random.default_rng(rng)
        n_train = float(train_fraction) * float(n_eff)
        # Drawn once and reused across the delta grid, so the comparison
        # between delta values is not confounded by different split noise.
        splits = [subsample_score_moments(wz, gram_raw, n_eff, n_train,
                                          var_y=var_y, rng=generator,
                                          check=(rep == 0))
                  for rep in range(n_repeats)]
        split_log = splits[0][2]
        selection = "PUMAS pseudo-split"
    elif tune == "independent":
        z_valid = np.asarray(z_valid, dtype=float).ravel()
        if z_valid.size != m:
            raise ValueError(f"z_valid covers {z_valid.size} variants but the "
                             f"weights cover {m}")
        if ld_valid is None:
            sel_gram = gram
        else:
            gram_v, _ = score_gram(weights, ld_valid, n_variants=n_variants)
            # The TRAINING scale, deliberately. beta comes off a path fitted in
            # training-standardized coordinates, so both halves of
            # (beta'r)^2 / (beta'G beta) must be expressed in those same
            # coordinates. Restandardizing the validation Gram by its own score
            # SDs would put numerator and denominator on different scales and
            # silently distort the selection.
            sel_gram = gram_v * np.outer(scale, scale)
        wz_v = np.zeros(k, dtype=float)
        np.add.at(wz_v, cols, vals * z_valid[rows])
        sel_r = wz_v * scale
        selection = "independent GWAS"
    else:
        sel_gram, sel_r, selection = gram, r, "in-sample"

    def _score_path(path_d, gram_fit):
        """Tuning criterion for every point of one delta's path.

        The criterion always uses the UNSHRUNK Gram: delta is a fitting device
        for stabilising the solve, not a claim about the score covariance, so
        the accuracy of the resulting coefficients is judged against the best
        estimate of the truth rather than against the regularised surrogate.
        """
        if splits is None:
            return np.array([pseudo_r2(b, sel_gram, sel_r, var_y=var_y)
                             for b in path_d])
        per_repeat = np.full((len(splits), path_d.shape[0]), np.nan)
        for rep, (c_tr, c_val, _) in enumerate(splits):
            fitted, n_rep = _coord.enet_path_gaussian(
                gram_fit, c_tr * scale, pf=pf, alpha=alpha, lambdas=lambdas_d,
                tol=tol, max_iter=max_iter)
            r_val = c_val * scale
            for i in range(min(n_rep, path_d.shape[0])):
                per_repeat[rep, i] = pseudo_r2(fitted[i], gram, r_val,
                                               var_y=var_y)
        with np.errstate(invalid="ignore"):
            return np.nanmean(per_repeat, axis=0)

    best_value = -np.inf
    best = None
    for delta in deltas:
        gram_d = gram + float(delta) * np.diag(penalized)
        _, grad = _coord.unpenalized_fit(gram_d, r, pf)
        lambdas_d = _coord.lambda_grid(grad, pf, alpha, n_lambda=n_lambda,
                                       lambda_min_ratio=lambda_min_ratio)
        path_d, n_fitted = _coord.enet_path_gaussian(
            gram_d, r, pf=pf, alpha=alpha, lambdas=lambdas_d, tol=tol,
            max_iter=max_iter)
        path_d = path_d[:n_fitted]
        lambdas_d = lambdas_d[:n_fitted]
        scores_d = _score_path(path_d, gram_d)
        if np.all(np.isnan(scores_d)):
            continue
        index = int(np.nanargmax(scores_d))
        if scores_d[index] > best_value:
            best_value = float(scores_d[index])
            best = (float(delta), index, path_d, lambdas_d, scores_d)

    if best is None:
        raise ValueError(
            "no point on any path predicts anything. The scores may share no "
            "variants with the LD reference, z may be aligned to different "
            "variants than the weights, or the Gram may be degenerate.")
    delta, index, path, lambdas, scores_path = best

    log = {"n_scores": k, "n_variants": m, "selection": selection,
           "n_dead": int(np.sum(dead)), "n_lambda": int(lambdas.size),
           "alpha": float(alpha), "var_y": float(var_y),
           "n_weight_entries": int(vals.size), "ld_shrinkage": delta,
           "n_shrinkage": int(deltas.size)}
    if tune == "pumas":
        log.update({"regime": "B", "n_repeats": len(splits),
                    "n_eff": float(n_eff),
                    "train_fraction": float(train_fraction),
                    "split": {kk: vv for kk, vv in split_log.items()
                              if kk not in ("n", "n_train", "n_val", "var_y")}})
        for key in ("warning", "rank_warning"):
            if key in split_log:
                log[key] = split_log[key]
    elif tune == "independent":
        log["regime"] = "A"
    else:
        log["regime"] = "C"
        log["warning"] = ("lambda was selected on the same summary statistics "
                          "it was fitted on, which drives the choice towards "
                          "the least-penalized point on the path; the reported "
                          "pseudovalidation R2 is optimistic")
    if deltas.size > 1 and delta in (deltas.min(), deltas.max()):
        log["shrinkage_warning"] = (
            f"the selected ld_shrinkage {delta:g} is at the edge of the grid; "
            "widen it, since the optimum may lie outside")
    if scores_path[index] > 1.0:
        # An R2 above one is not a near miss, it is a unit error: z was almost
        # certainly not on the standardized scale, or var_y does not match the
        # scale it was formed on.
        log["scale_warning"] = (
            f"pseudovalidation R2 is {scores_path[index]:.3g}, which is above 1 "
            "and therefore not an R2. Check that z came from "
            "ldpred3.standardize_betas(beta, se, n_eff) and that var_y matches "
            "the scale z is on.")
    return SumstatFit(
        beta=path[index].copy(), score_ids=score_ids, score_sd=sd,
        gram=gram, r=r, lambdas=lambdas, path=path, lambda_index=index,
        alpha=float(alpha), pseudo_r2=float(scores_path[index]),
        pseudo_r2_path=scores_path, c_raw=wz.copy(), log=log)


# ---------------------------------------------------------------------------
# Score-space subsampling (PUMAS)
# ---------------------------------------------------------------------------

def _psd_sqrt(gram, *, tol=1e-10):
    """A factor ``L`` with ``L L^T`` the PSD projection of ``gram``.

    Cholesky is not used: a score panel is routinely rank-deficient (a
    1000-Genomes reference has ~500 individuals, so a 900-score Gram cannot have
    full rank, and same-trait panels contain near-duplicate scores), and the
    usual remedy of adding jitter until Cholesky succeeds silently invents
    variance in directions where the reference carries no information. An
    eigendecomposition puts exactly zero there instead, and reports the rank so
    the caller can refuse.
    """
    gram = np.asarray(gram, dtype=float)
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    largest = float(values.max()) if values.size else 0.0
    if largest <= 0.0:
        return np.zeros_like(gram), 0, 0.0
    keep = values > tol * largest
    negative = float(-values[values < -tol * largest].sum())
    factor = vectors[:, keep] * np.sqrt(values[keep])
    return factor, int(keep.sum()), negative / max(largest, 1e-300)


def subsample_score_moments(c, gram, n, n_train, *, var_y, rng=None,
                            check=True):
    """One PUMAS-style pseudo-split of score-space moments.

    Draws ``(c_train, c_val)`` with the joint law of the score-space statistics
    of two GWAS run on disjoint subsets of the same ``n`` individuals. The
    per-individual covariance of the score-by-phenotype cross-product
    ``v_i = s_i y_i`` is

    .. math::

        V_S = \\mathrm{var}(y)\\,G + c c^\\top
            = W^\\top(\\mathrm{var}(y) D + z z^\\top) W,

    an exact identity, so drawing in ``K`` dimensions is distributionally
    identical to drawing in ``m`` dimensions and projecting — not an
    approximation of it. That is what makes this cheap: the ``m x m`` matrix
    square root PUMAS implies is genuinely unnecessary, and the rank-one term
    needs no factorisation of its own.

    .. math::

        c_{tr} \\mid c \\sim N\\!\\left(c,\\ \\kappa_{tr} V_S\\right),
        \\qquad \\kappa_{tr} = \\frac{1}{n_{tr}} - \\frac{1}{n},

    and ``c_val`` is the **deterministic** complement
    ``(n c - n_tr c_tr) / n_val``. Do not draw it separately: the negative
    conditional coupling ``Cov(c_tr, c_val | c) = -V_S/n`` is exactly the term
    that debits a combination for fitting the training half, and removing it
    collapses the criterion back to in-sample selection. The variance factor
    must be ``1/n_tr - 1/n`` and not ``1/n_tr``; the former is what makes
    ``c_tr`` and ``c_val`` *unconditionally* independent, and the latter leaves
    them negatively correlated and the tuning anticonservative.

    Parameters
    ----------
    c, gram : array_like
        **Raw**-scale score moments, ``W^T z`` and ``W^T D W``, from
        :func:`score_moments`. Raw, not standardized: a score with no variance
        under the reference has a genuinely zero row here, whereas the unit
        diagonal the fit substitutes would make this inject noise into a score
        that contributes nothing.
    n : float
        Effective sample size of the GWAS behind ``c``. For case/control use
        ``ldpred3.n_eff_case_control``; passing the raw total instead injects
        too little noise and selects too small a penalty.
    n_train : float
        Pseudo-training size, ``0 < n_train < n``.
    var_y : float
        Phenotype variance on the scale ``c`` was formed on. Not optional and
        not defaulted: it is the mixing ratio between the noise floor
        ``var_y G`` and the signal term ``c c^T``, so getting it wrong rescales
        the injected noise rather than the reported number.

    Returns
    -------
    (c_train, c_val, log) : (ndarray, ndarray, dict)
    """
    c = np.asarray(c, dtype=float).ravel()
    gram = np.asarray(gram, dtype=float)
    k = c.size
    if gram.shape != (k, k):
        raise ValueError(f"c is length {k}, so gram must be ({k}, {k}), got "
                         f"{gram.shape}")
    n = float(n)
    n_train = float(n_train)
    var_y = float(var_y)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("n must be finite and positive")
    if not np.isfinite(n_train) or not 0.0 < n_train < n:
        raise ValueError(f"n_train must satisfy 0 < n_train < n = {n}, got "
                         f"{n_train}")
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(gram)):
        raise ValueError("c and gram must be finite")

    n_val = n - n_train
    diag = np.diag(gram)
    log = {"n": n, "n_train": n_train, "n_val": n_val, "var_y": var_y}

    if check:
        # A single score cannot explain more of the phenotype than there is.
        # c_k^2 / (var_y G_kk) above one is not a marginal call, it is a
        # provenance error — a per-allele z, a wrong var_y, or overlap between
        # this score's discovery GWAS and the target — and no downstream check
        # attributes it correctly.
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.where(diag > 0, c * c / (var_y * diag), 0.0)
        bad = np.flatnonzero(implied > 1.0)
        if bad.size:
            raise ValueError(
                f"{bad.size} score(s) imply a single-score R2 above 1 "
                f"(largest {implied.max():.3g}, score index {int(np.argmax(implied))}). "
                "That is impossible, so one of these is wrong: z is on a "
                "per-allele rather than standardized scale, var_y does not "
                "match the scale of z, or the score's discovery GWAS shares "
                "individuals with the GWAS behind z.")
        log["max_implied_r2"] = float(implied.max()) if k else 0.0

    factor, rank, negative_mass = _psd_sqrt(gram)
    log["rank"] = rank
    log["n_scores"] = k
    if negative_mass > 1e-8:
        log["warning"] = (
            f"the Gram has negative eigenvalues carrying {negative_mass:.2e} of "
            "its largest — the LD reference is not positive semi-definite. "
            "They were dropped; rebuild the reference with a ridge instead.")
    if rank < k:
        # In a null direction of G the split is deterministic, so a combination
        # can move there for free and pay no validation penalty.
        log["rank_warning"] = (
            f"the Gram has rank {rank} for {k} scores, so in {k - rank} "
            "direction(s) the pseudo-split carries no noise and a fit can "
            "overfit there without being penalised. Reduce the panel, or use a "
            "larger LD reference.")

    rng = np.random.default_rng(rng)
    kappa = 1.0 / n_train - 1.0 / n
    # V_S = var_y G + c c^T is a rank-one update of G, so one factor of G plus
    # one extra scalar Gaussian samples it exactly; V_S is never formed.
    noise = (np.sqrt(var_y) * (factor @ rng.standard_normal(factor.shape[1]))
             + rng.standard_normal() * c)
    c_train = c + np.sqrt(kappa) * noise
    c_val = (n * c - n_train * c_train) / n_val
    return c_train, c_val, log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class SumstatEval:
    """Accuracy of a fixed combination, measured from summary statistics.

    ``regime`` is one of the keys of :data:`REGIMES` and is the first thing to
    read: ``"C"`` means the combination was scored against the statistics it was
    fitted on and the number is an upper bound, not an estimate.
    """

    r2: float
    mse: float
    regime: str
    var_y: float
    n_scores: int
    log: dict = field(default_factory=dict)

    @property
    def is_validation(self):
        """False for regime C, which measures fit rather than accuracy."""
        return self.regime in ("A", "B")

    def summary(self):
        lines = [f"summary-statistic evaluation (regime {self.regime}): "
                 f"R2 {self.r2:.4f}, MSE {self.mse:.4f}",
                 f"  {REGIMES.get(self.regime, 'unknown provenance')}"]
        if not self.is_validation:
            lines.append("  this is not a validation — do not report it as one")
        if self.log.get("warning"):
            lines.append(f"  {self.log['warning']}")
        return "\n".join(lines)


def evaluate_sumstat(beta, c_eval, gram_eval, *, var_y=1.0, regime=None,
                     fitted_on=None):
    """Accuracy of a fixed combination ``beta`` against evaluation moments.

    Implements the two summary-statistic identities, applied in score space to
    the combined weight vector ``W beta``:

    .. math::

        R^2(a) = \\frac{(a^\\top c)^2}{(a^\\top G a)\\,\\mathrm{var}(y)},
        \\qquad
        \\mathrm{MSE}(a) = \\mathrm{var}(y) - 2 a^\\top c + a^\\top G a.

    Parameters
    ----------
    beta : array_like, shape (K,) or SumstatFit
        The combination to score, on the standardized-score scale.
    c_eval, gram_eval : array_like
        Score-space moments from the **evaluation** data: ``W^T z_eval`` and
        ``W^T D_eval W``, on the same score scaling ``beta`` is on.
    var_y : float
        Phenotype variance on the scale ``c_eval`` was formed on. The MSE is
        meaningless if this is wrong, and the R² is off by a constant factor.
    regime : {"A", "B", "C"}, optional
        Declare the provenance. Omitted, it is inferred: identical to the
        fitting moments means regime C.
    fitted_on : array_like, optional
        The ``c`` the combination was fitted on, so that regime C can be
        detected rather than trusted. Supplied automatically when ``beta`` is a
        :class:`SumstatFit`.

    Returns
    -------
    SumstatEval
    """
    if isinstance(beta, SumstatFit):
        # A fit's coefficients are on the standardized-score scale; the moments
        # are not. Converting here means the two scales cannot be crossed by
        # passing a fit where a vector was expected.
        if fitted_on is None:
            fitted_on = beta.c_raw
        beta = beta.raw_beta
    beta = np.asarray(beta, dtype=float).ravel()
    c_eval = np.asarray(c_eval, dtype=float).ravel()
    gram_eval = np.asarray(gram_eval, dtype=float)
    k = beta.size
    if c_eval.size != k or gram_eval.shape != (k, k):
        raise ValueError(f"beta is length {k}, so c_eval must be ({k},) and "
                         f"gram_eval ({k}, {k}); got {c_eval.shape} and "
                         f"{gram_eval.shape}")
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(c_eval)):
        raise ValueError("beta and c_eval must be finite")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")

    if regime is None:
        # Only regime C is inferable: it is the one with an observable
        # signature, namely evaluation moments identical to the fitting ones.
        # "Not C" does NOT imply A — a PUMAS pseudo-validation split is also
        # not identical to its pseudo-training split, and defaulting to A there
        # would stamp "clean external validation" on a regime B number. An
        # unprovable label is worse than none, so this refuses instead.
        same = (fitted_on is not None
                and np.asarray(fitted_on).shape == c_eval.shape
                and np.allclose(np.asarray(fitted_on, dtype=float), c_eval))
        if not same:
            raise ValueError(
                "cannot infer the evaluation regime: these moments differ from "
                "the ones the combination was fitted on, which rules out C but "
                "does not distinguish A (an independent GWAS) from B (a PUMAS "
                "pseudo-validation split). Pass regime='A' or regime='B' "
                "explicitly — the two are not interchangeable and the "
                "difference is invisible in the number.")
        regime = "C"
    regime = str(regime).upper()
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got "
                         f"{regime!r}")

    quad = float(beta @ gram_eval @ beta)
    if quad < 0.0:
        raise ValueError(
            f"beta^T G beta = {quad!r} is negative, so the evaluation LD is "
            "not positive semi-definite. The MSE this would report understates "
            "the error rather than failing visibly.")
    num = float(beta @ c_eval)
    r2 = float("nan") if quad == 0.0 else (num * num) / (quad * var_y)
    mse = var_y - 2.0 * num + quad

    log = {"n_nonzero": int(np.sum(beta != 0.0)),
           "beta_c": num, "beta_G_beta": quad,
           "regime_detail": REGIMES[regime]}
    if regime == "C":
        log["warning"] = ("evaluated on the same summary statistics it was "
                          "fitted on; this is an upper bound, not a validation")
    if np.isfinite(r2) and r2 > 1.0:
        log["scale_warning"] = (
            f"R2 is {r2:.3g}, above 1 and therefore not an R2 — check that "
            "c_eval is on the standardized scale and that var_y matches it")
    if mse > var_y:
        log["mse_warning"] = ("MSE exceeds var(y): this combination predicts "
                              "worse than the mean")
    return SumstatEval(r2=r2, mse=mse, regime=regime, var_y=var_y, n_scores=k,
                       log=log)


def score_moments(weights, z, ld, *, n_variants=None):
    """The score-space moments ``(c, G)`` for one set of summary statistics.

    The pair that :func:`evaluate_sumstat` scores against, and the same pair
    :func:`multi_pgs_sumstats` fits from. Building them for an *evaluation* GWAS
    is how a combination gets an honest regime A number.
    """
    z = np.asarray(z, dtype=float).ravel()
    rows, cols, vals, m, k = _weight_columns(weights, n_variants)
    if z.size != m:
        raise ValueError(f"z covers {z.size} variants but the weights cover {m}")
    gram, var = score_gram(weights, ld, n_variants=n_variants)
    c = np.zeros(k, dtype=float)
    np.add.at(c, cols, vals * z[rows])
    return c, gram, var


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_to_reference(scoring_files, variants, *, af=None,
                       drop_ambiguous=True, on_error="raise", progress=None):
    """Align PGS Catalog scoring files to an LD reference's variant table.

    Returns weights on the **standardized** genotype scale, which is what
    :func:`score_gram` needs and what catalog files are *not* on: a catalog
    weight multiplies a raw allele count, and converting requires each variant's
    standard deviation ``sqrt(2 f (1 - f))`` under the reference's allele
    frequencies. Without ``af`` the conversion cannot be done and the weights
    are returned unconverted, which is correct only if they were already
    standardized (an LDpred3 weight file).

    Parameters
    ----------
    scoring_files : sequence of str or ScoringFile
    variants : mapping
        The LD reference's variant table, with ``id chrom pos a1 a2``, in the
        reference's own order — the row order of ``D``.
    af : array_like, optional
        Reference allele frequency of ``a1`` per variant.
    on_error : {"raise", "skip"}

    Returns
    -------
    (pairs, score_ids, log) : (list of (index, weight), list of str, dict)
        ``pairs`` goes straight to :func:`score_gram` and
        :func:`multi_pgs_sumstats`.
    """
    from .catalog import harmonize_scoring_file, read_scoring_file

    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    # A reference variant table is either ldpred3's VariantTable (attributes)
    # or the plain mapping multipgs.panel builds for a catalog union.
    ids = variants.id if hasattr(variants, "id") else variants["id"]
    n_variants = len(np.asarray(ids, dtype=object).ravel())
    sd = None
    if af is not None:
        f = np.asarray(af, dtype=float).ravel()
        if f.size != n_variants:
            raise ValueError(f"af has {f.size} entries for {n_variants} "
                             "reference variants")
        sd = np.sqrt(2.0 * f * (1.0 - f))

    files = list(scoring_files)
    pairs, ids, errors = [], [], {}
    matched = []
    for i, item in enumerate(files):
        label = getattr(item, "pgs_id", None) or str(item)
        try:
            scoring = item if hasattr(item, "weight") else read_scoring_file(item)
            idx, w, log = harmonize_scoring_file(scoring, variants,
                                                 drop_ambiguous=drop_ambiguous)
            if sd is not None:
                # A catalog weight counts alleles; on standardized genotypes the
                # same score is w * sd. A monomorphic reference variant has
                # sd = 0 and contributes nothing, which is the truth here.
                w = w * sd[idx]
            pairs.append((idx, w))
            ids.append(scoring.pgs_id)
            matched.append(int(log.get("n_matched", idx.size)))
        except Exception as exc:                      # noqa: BLE001
            if on_error == "raise":
                raise
            errors[label] = str(exc)
        if progress is not None:
            progress(i, len(files), label)

    log = {"n_requested": len(files), "n_aligned": len(pairs),
           "n_failed": len(errors), "n_reference_variants": n_variants,
           "standardized": sd is not None}
    if matched:
        log["n_matched_median"] = int(np.median(matched))
        log["n_matched_min"] = int(min(matched))
    if errors:
        log["errors"] = errors
    if sd is None:
        log["warning"] = ("no allele frequencies given, so catalog weights were "
                          "not converted to the standardized scale; this is "
                          "correct only for weights that were already on it")
    return pairs, ids, log
