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
``n^{-1} X^T X`` and ``n^{-1} X^T y`` exactly. With an external LD reference,
``G`` is instead a plug-in covariance estimate. The K-by-K score covariance
therefore comes from an **LD reference**, while the K-vector of score-phenotype
covariances comes from a **GWAS of the target trait**. The two weight matrices
may cover different variant sets and orders; only their K score columns and raw
score definition must agree.

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
fitting, covariate adjustment, AUC, and any participant-level bootstrap. Those
need individuals, and no amount of summary-statistic algebra recovers them.

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
    "A": "fixed after all model selection, then evaluated on an untouched "
         "independent GWAS (clean external assessment)",
    "B": "used to choose the model, either an independent tuning GWAS or a "
         "PUMAS pseudo-validation split (tuning, not clean assessment)",
    "C": "fitted and evaluated on the same unsplit GWAS "
         "(optimistically biased; not a validation)",
}


# ---------------------------------------------------------------------------
# Gram accumulation
# ---------------------------------------------------------------------------

def _as_blocks(ld, n_variants):
    """Normalize an LD argument to ``(corr_block, idx)`` pairs.

    Accepts ldpred3's native block list, a single dense correlation matrix, or
    anything ``ldpred3.ld_matmul`` understands paired with explicit indices.
    Non-dense inputs are consumed lazily and must tile ``0..m-1`` exactly once.
    """
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"a dense LD matrix must be square, got {ld.shape}")
        if ld.shape[0] != n_variants:
            raise ValueError(f"LD matrix is {ld.shape[0]} x {ld.shape[0]} but "
                             f"the weights cover {n_variants} variants")
        return [(ld, np.arange(n_variants))]

    def validated():
        expected = 0
        seen = False
        for b, block in enumerate(ld):
            if not (isinstance(block, tuple) and len(block) == 2):
                raise ValueError(
                    f"LD block {b} is not a (corr_block, idx) pair; this is the "
                    "layout ldpred3.compute_ld_blocks and "
                    "ldpred3.load_ld_blocks return, got "
                    f"{type(block).__name__}")
            corr, idx = block
            idx = np.asarray(idx, dtype=np.int64).ravel()
            if idx.size == 0:
                continue
            seen = True
            if np.any(np.diff(idx) != 1):
                raise ValueError(
                    f"LD block {b} does not cover a contiguous run of variants. "
                    "The streaming Gram slices weights by block range; use "
                    "ldpred3.compute_ld_blocks, whose blocks tile 0..m-1.")
            if int(idx[0]) != expected:
                kind = "overlaps an earlier block" if int(idx[0]) < expected \
                    else "leaves a gap before it"
                raise ValueError(
                    f"LD block {b} starts at variant {int(idx[0])}, expected "
                    f"{expected}; it {kind}. Blocks must tile 0..m-1 exactly "
                    "once and in order.")
            expected = int(idx[-1]) + 1
            yield corr, idx
        if not seen:
            raise ValueError("the LD reference has no blocks")
        if expected != n_variants:
            raise ValueError(f"the LD blocks cover variants 0..{expected - 1} "
                             f"but the weights cover 0..{n_variants - 1}")

    return validated()


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
        if not np.all(np.isfinite(weights)):
            raise ValueError("dense weights contain non-finite values")
        m, k = weights.shape
        if n_variants is not None:
            if isinstance(n_variants, (bool, np.bool_)):
                raise ValueError("n_variants must be a non-negative integer")
            try:
                requested = int(n_variants)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "n_variants must be a non-negative integer") from None
            if requested < 0 or requested != n_variants:
                raise ValueError("n_variants must be a non-negative integer")
            if requested != m:
                raise ValueError(
                    f"n_variants={requested} but dense weights have {m} rows")
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
            # A sparse score is a mathematical vector, not an insertion log.
            # Coalescing here gives W'DW and W'z the same interpretation.
            unique, inverse = np.unique(idx, return_inverse=True)
            if unique.size != idx.size:
                combined = np.zeros(unique.size, dtype=float)
                np.add.at(combined, inverse, val)
                keep = combined != 0.0
                idx, val = unique[keep], combined[keep]
        idx_parts.append(idx)
        col_parts.append(np.full(idx.size, k, dtype=np.int64))
        val_parts.append(val)

    m = int(largest + 1) if n_variants is None else int(n_variants)
    if m < 0:
        raise ValueError("n_variants must be non-negative")
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
    return _score_gram_from_coo(_weight_columns(weights, n_variants), ld)


def _block_quadform(corr, block_w):
    """``W_b^T D_b W_b`` for one LD block, using its own representation.

    ldpred3 stores a large block as a low-rank factor (LR8):
    ``D = U U^T + diag(residual)``. Going through :func:`ldpred3.ld_matmul`
    computes ``U (U^T W)`` — projecting back up to the block's full variant
    dimension — only for this function to immediately contract it back down
    again. Keeping the factor instead,

        W^T D W = (U^T W)^T (U^T W) + (residual * W)^T W,

    skips that back-projection and shrinks the second product from ``O(k A^2)``
    to ``O(r A^2)``. In the bigsnpr HapMap3+ reference the low-rank blocks hold
    the bulk of the variants — median 3,120 variants at median rank 890, so
    ``r/k`` is about 0.29 — and this is where a genome-wide Gram spends its
    time.

    Everything else, including dense int8 and float32 blocks and any
    representation added later, falls through to ``ld_matmul``, which is also
    the fallback if the pinned ldpred3 does not expose the dequantizer.
    """
    from ldpred3 import ld_matmul

    try:
        from ldpred3 import LowRankLD

        from ._ldpred3_compat import dequantize_ld
    except (ImportError, AttributeError):    # pragma: no cover - older ldpred3
        return block_w.T @ np.asarray(ld_matmul(corr, block_w), dtype=float)

    block = dequantize_ld(corr)
    if not isinstance(block, LowRankLD):
        return block_w.T @ np.asarray(ld_matmul(block, block_w), dtype=float)
    # float64 throughout, so a compact block does not lower the Gram's
    # precision — the same contract ld_matmul documents for its own return.
    factor = np.asarray(block.U, dtype=np.float64)
    residual = np.asarray(block.residual_diag, dtype=np.float64)
    projected = factor.T @ block_w
    return projected.T @ projected + (block_w * residual[:, None]).T @ block_w


def _score_gram_from_coo(parsed, ld):
    """:func:`score_gram` on already-parsed weights.

    Parsing a sparse weight set materializes three arrays over every non-zero
    entry, which for a genome-wide panel is the largest allocation in the whole
    fit. A caller that already holds the parse passes it here instead of
    handing the raw weights back to be parsed a second time.
    """
    rows, cols, vals, m, k = parsed
    blocks = _as_blocks(ld, m)

    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]

    gram = np.zeros((k, k), dtype=float)
    for corr, idx in blocks:
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        start, stop = np.searchsorted(rows, (lo, hi))
        if start == stop:
            continue
        # Catalog scores are sparse across both variants and blocks. Work only
        # on the scores touching this block; forming a B x K matrix and a full
        # K x K product here can waste two orders of magnitude of work.
        active = np.unique(cols[start:stop])
        local_cols = np.searchsorted(active, cols[start:stop])
        block_w = np.zeros((idx.size, active.size), dtype=float)
        block_w[rows[start:stop] - lo, local_cols] = vals[start:stop]
        gram[np.ix_(active, active)] += _block_quadform(corr, block_w)

    # W^T D W is symmetric in exact arithmetic; the accumulation is not, and an
    # asymmetric Gram makes the coordinate descent's covariance updates drift.
    gram = 0.5 * (gram + gram.T)
    return gram, np.diag(gram).copy()


def _symmetrized(gram):
    """``(symmetric, asymmetry)`` for one Gram, computed once.

    Both quantities cost ``O(K^2)`` and depend on nothing but ``gram``, so a
    caller scoring a whole path against a fixed Gram prepares them once instead
    of rebuilding them per candidate.
    """
    asymmetry = float(np.max(np.abs(gram - gram.T))) if gram.size else 0.0
    return 0.5 * (gram + gram.T), asymmetry


def _directional_score_moments(beta, gram, r, var_y, label, *, prepared=None):
    """Validate only the scalar score direction used by a fixed ``beta``."""
    symmetric, asymmetry = _symmetrized(gram) if prepared is None else prepared
    quad = float(beta @ symmetric @ beta)
    quad_scale = float(np.sum(
        beta * beta * np.maximum(np.diag(symmetric), 0.0)))
    quad_tol = 1e-12 * max(quad_scale, np.finfo(float).tiny)
    if quad < -quad_tol:
        raise ValueError(
            f"beta^T G beta = {quad!r} is negative, so {label} is indefinite "
            "in the direction actually used by beta")
    if quad < 0.0:
        quad = 0.0
    num = float(beta @ r)
    num_scale = float(np.sum(np.abs(beta * r)))
    num_tol = 1e-12 * max(num_scale, np.sqrt(var_y * quad_scale),
                          np.finfo(float).tiny)
    if quad <= quad_tol and abs(num) > num_tol:
        raise ValueError(
            f"the score direction used by beta has zero variance under {label} "
            "but nonzero covariance with the trait")
    return num, quad, quad_tol, symmetric, asymmetry


def _project_c_to_gram_range(c, gram):
    """Project ``c`` onto identifiable covariance directions, scale-invariantly."""
    symmetric = 0.5 * (gram + gram.T)
    diagonal = np.diag(symmetric)
    active = diagonal > 0.0
    projected = np.zeros_like(c)
    c_scaled = np.zeros(int(np.sum(active)), dtype=float)
    projected_scaled = np.zeros_like(c_scaled)
    if np.any(active):
        sd = np.sqrt(diagonal[active])
        correlation = symmetric[np.ix_(active, active)] / np.outer(sd, sd)
        correlation = 0.5 * (correlation + correlation.T)
        values, vectors = np.linalg.eigh(correlation)
        scale = max(float(np.max(np.abs(values))) if values.size else 0.0,
                    np.finfo(float).tiny)
        keep = values > 1e-10 * scale
        c_scaled = c[active] / sd
        if np.any(keep):
            basis = vectors[:, keep]
            projected_scaled = basis @ (basis.T @ c_scaled)
        projected[active] = projected_scaled * sd
    discarded = c_scaled - projected_scaled
    discarded_norm = float(np.linalg.norm(discarded))
    discarded_fraction = discarded_norm / max(
        float(np.linalg.norm(c_scaled)), np.finfo(float).tiny)
    if np.any(~active):
        inactive_norm = float(np.linalg.norm(c[~active]))
        discarded_norm = float(np.hypot(discarded_norm, inactive_norm))
        total_norm = float(np.hypot(np.linalg.norm(c_scaled), inactive_norm))
        discarded_fraction = discarded_norm / max(
            total_norm, np.finfo(float).tiny)
    return projected, discarded_norm, discarded_fraction


def _selection_candidate_valid(beta, gram, r, var_y):
    """Whether noisy plug-in moments can physically rank this path point."""
    valid, r2, mse = _selection_candidates_valid(
        np.asarray(beta, dtype=float)[None, :], gram, r, var_y)
    return bool(valid[0]), float(r2[0]), float(mse[0])


def _selection_candidates_valid(path, gram, r, var_y, *, prepared=None):
    """Vectorized :func:`_selection_candidate_valid` over a whole path.

    The per-candidate form rebuilt the symmetrized Gram — ``O(K^2)`` — for
    every one of the hundred-odd path points, for every shrinkage value, for
    every PUMAS repeat. Preparing it once and taking all the quadratic forms as
    one matrix product is the same arithmetic in a different order; at ``K=900``
    it is the difference between 642 ms and 4 ms per path.

    A candidate whose moments are physically impossible is reported invalid
    with ``nan`` statistics, exactly as the scalar form's caught ``ValueError``
    did.
    """
    path = np.atleast_2d(np.asarray(path, dtype=float))
    symmetric, _ = _symmetrized(gram) if prepared is None else prepared
    tiny = np.finfo(float).tiny
    quad = np.einsum("ij,ij->i", path @ symmetric, path)
    quad_scale = (path * path) @ np.maximum(np.diag(symmetric), 0.0)
    quad_tol = 1e-12 * np.maximum(quad_scale, tiny)
    num = path @ r
    num_scale = np.sum(np.abs(path * r), axis=1)
    num_tol = 1e-12 * np.maximum(
        np.maximum(num_scale, np.sqrt(var_y * quad_scale)), tiny)

    indefinite = quad < -quad_tol
    quad = np.where(~indefinite & (quad < 0.0), 0.0, quad)
    degenerate = (quad <= quad_tol) & (np.abs(num) > num_tol)
    impossible = indefinite | degenerate

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(quad <= quad_tol, np.nan, (num * num) / (quad * var_y))
    mse = var_y - 2.0 * num + quad
    valid = ((~np.isfinite(r2) | (r2 <= 1.0 + 1e-8))
             & (mse >= -1e-8 * var_y) & ~impossible)
    r2 = np.where(impossible, np.nan, r2)
    mse = np.where(impossible, np.nan, mse)
    return valid, r2, mse


def pseudo_r2(beta, gram, r, *, var_y=1.0):
    """Summary-statistic accuracy of the combined score ``W beta``.

    ``(beta^T r)^2 / (beta^T G beta * var_y)`` — the stack's version of
    ``ppb``'s ``R^2 = (w^T z)^2 / (w^T D w)``. Returns ``nan`` for an all-zero
    ``beta`` (nothing is predicted, so the ratio is undefined rather than zero)
    and raises when the denominator is negative, which only a non-PSD LD
    approximation can produce and which would understate the error silently.
    The numerator uses ``r`` projected onto the scale-invariant positive range
    of ``gram``; finite-reference null noise is not identifiable prediction.
    """
    beta = np.asarray(beta, dtype=float)
    gram = np.asarray(gram, dtype=float)
    r = np.asarray(r, dtype=float)
    if beta.ndim != 1 or r.shape != beta.shape or \
            gram.shape != (beta.size, beta.size):
        raise ValueError("beta and r must be length K and gram must be (K, K)")
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(r))
            and np.all(np.isfinite(gram))):
        raise ValueError("beta, r and gram must be finite")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    r_identifiable, _, _ = _project_c_to_gram_range(r, gram)
    num, den, den_tol, _, _ = _directional_score_moments(
        beta, gram, r_identifiable, var_y, "gram")
    if den <= den_tol:
        return float("nan")
    return (num * num) / (den * var_y)


def _pseudo_r2_unchecked(beta, gram, r, var_y):
    """Fast fixed-vector R2 after the caller has validated the moments once."""
    r2, _ = _pseudo_r2_batch(np.asarray(beta, dtype=float)[None, :], gram, r,
                             var_y)
    return float(r2[0])


def _pseudo_r2_batch(paths, gram, r, var_y):
    """Vectorized :func:`_pseudo_r2_unchecked`, returning ``(r2, quadratic)``.

    Returning the quadratic form as well means the caller's MSE does not repeat
    the ``O(K^2)`` product this already computed.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    num = paths @ r
    den = np.einsum("ij,ij->i", paths @ gram, paths)
    den_scale = (paths * paths) @ np.maximum(np.diag(gram), 0.0)
    den_tol = 1e-12 * np.maximum(den_scale, np.finfo(float).tiny)
    negative = den < -den_tol
    if np.any(negative):
        worst = den[negative][0]
        raise ValueError(
            f"beta^T G beta = {worst!r} is negative, so the LD reference is not "
            "positive semi-definite here. Rebuild it with a ridge "
            "(ldpred3.compute_ld_blocks(..., ridge=...)) rather than trusting "
            "the accuracy this would report.")
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(den <= den_tol, np.nan, (num * num) / (den * var_y))
    return r2, den


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _weight_digest(rows, cols, vals, n_variants, n_scores):
    """Canonical digest of the exact aligned score matrix used by a fit."""
    import hashlib

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    keep = vals != 0.0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.lexsort((cols, rows))
    digest = hashlib.sha256()
    digest.update(np.asarray([n_variants, n_scores], dtype="<i8").tobytes())
    digest.update(rows[order].astype("<i8", copy=False).tobytes())
    digest.update(cols[order].astype("<i8", copy=False).tobytes())
    digest.update(vals[order].astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


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
        rows, cols, vals, m, k = _weight_columns(weights_ld, n_variants_ld)
        if k != self.beta.size:
            raise ValueError(f"weights describe {k} scores but this fit has "
                             f"{self.beta.size}")
        if self.weights_ld_digest is not None:
            got = _weight_digest(rows, cols, vals, m, k)
            if got != self.weights_ld_digest:
                raise ValueError(
                    "weights differ from the aligned score matrix used to fit "
                    "this combination; variant_weights cannot safely attach "
                    "coefficients to them")
        out = np.zeros(m, dtype=float)
        np.add.at(out, rows, vals * self.beta[cols])
        return out

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
        if self.log.get("selection_moment_warning"):
            lines.append(f"  {self.log['selection_moment_warning']}")
        return "\n".join(lines)


def _validate_moments(c, gram, var_y, *, label):
    """Validate ``G`` globally and retain, but do not police, noisy ``c``.

    External GWAS noise means ``c`` need not lie exactly in the range of a
    finite-reference ``G`` and its plug-in Schur complement can exceed
    ``var_y``. Those are useful diagnostics, not population identities to
    enforce on noisy estimates. Convex boundedness is checked separately for
    every fitted objective; fixed-vector evaluation checks only the direction
    actually used by ``beta``.
    """
    c = np.asarray(c, dtype=float).ravel()
    gram = np.asarray(gram, dtype=float)
    var_y = float(var_y)
    k = c.size
    if k == 0:
        raise ValueError(f"{label} has no scores")
    if gram.shape != (k, k):
        raise ValueError(f"{label}: c is length {k}, so gram must be ({k}, {k}), "
                         f"got {gram.shape}")
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(gram)):
        raise ValueError(f"{label} c and gram must be finite")

    magnitude = float(np.max(np.abs(gram))) if gram.size else 0.0
    asymmetry = float(np.max(np.abs(gram - gram.T))) if gram.size else 0.0
    if asymmetry > 1e-10 * max(magnitude, np.finfo(float).tiny):
        raise ValueError(f"{label} gram is not symmetric (maximum asymmetry "
                         f"{asymmetry:.3g})")
    symmetric = 0.5 * (gram + gram.T)

    # Rank and definiteness must not depend on the arbitrary units of a score.
    # Work on correlation coordinates, then map the cleaned covariance and its
    # factor back to raw units. An eigentolerance on raw G would incorrectly
    # call a perfectly valid score null merely because its weights were scaled
    # by (say) 1e-6.
    diagonal = np.diag(symmetric).copy()
    diagonal_scale = max(float(np.max(np.abs(diagonal))),
                         np.finfo(float).tiny)
    materially_negative = diagonal < -1e-12 * diagonal_scale
    if np.any(materially_negative):
        j = int(np.flatnonzero(materially_negative)[0])
        raise ValueError(
            f"{label} gram is materially indefinite: it has negative variance "
            f"{diagonal[j]:.3g} at score {j}")
    near_negative = (diagonal < 0.0) & ~materially_negative
    diagonal[near_negative] = 0.0
    active = diagonal > 0.0
    inactive = ~active

    # A truly zero-variance random variable has an exactly zero covariance row
    # and score-trait covariance. Clean only floating-point dust; a substantive
    # entry is an incoherent moment pair, not a score to silently drop.
    if np.any(inactive):
        row_error = (float(np.max(np.abs(symmetric[inactive, :])))
                     if np.any(symmetric[inactive, :]) else 0.0)
        row_tol = 1e-12 * max(magnitude, np.finfo(float).tiny)
        if row_error > row_tol:
            raise ValueError(
                f"{label} gram has a zero-variance score with covariance "
                f"{row_error:.3g}; a PSD covariance must have a zero row there")

    active_idx = np.flatnonzero(active)
    if active_idx.size:
        sd = np.sqrt(diagonal[active])
        active_gram = symmetric[np.ix_(active, active)]
        correlation = active_gram / np.outer(sd, sd)
        correlation = 0.5 * (correlation + correlation.T)
        correlation[np.diag_indices_from(correlation)] = 1.0
        c_scaled = c[active] / sd
    else:
        sd = np.zeros(0, dtype=float)
        correlation = np.zeros((0, 0), dtype=float)
        c_scaled = np.zeros(0, dtype=float)

    values, vectors = np.linalg.eigh(correlation)
    spectral_scale = max(float(np.max(np.abs(values))) if values.size else 0.0,
                         np.finfo(float).tiny)
    min_eigenvalue = float(values[0]) if values.size else 0.0
    if min_eigenvalue < -1e-8 * spectral_scale:
        raise ValueError(
            f"{label} gram is materially indefinite on correlation scale: "
            f"minimum eigenvalue {min_eigenvalue:.3g}, spectral scale "
            f"{spectral_scale:.3g}. "
            "Rebuild the LD reference at adequate precision; fitting this "
            "objective is not convex.")

    clipped = np.maximum(values, 0.0)
    largest = float(clipped[-1]) if clipped.size else 0.0
    rank_cutoff = 1e-10 * max(largest, np.finfo(float).tiny)
    keep = clipped > rank_cutoff
    factor_scaled = vectors[:, keep] * np.sqrt(clipped[keep])
    projected_c = vectors[:, keep].T @ c_scaled
    residual = c_scaled - vectors[:, keep] @ projected_c
    residual_norm = float(np.linalg.norm(residual))
    range_scale = max(float(np.linalg.norm(c_scaled)),
                      float(np.sqrt(var_y * max(largest, 0.0))),
                      np.finfo(float).tiny)
    explained = (float(np.sum(projected_c * projected_c / clipped[keep]))
                 if np.any(keep) else 0.0)

    projected = bool(np.any(values < 0.0) or np.any(near_negative))
    clean_correlation = ((vectors * clipped) @ vectors.T
                         if np.any(values < 0.0) else correlation)
    clean = np.zeros_like(symmetric)
    factor = np.zeros((k, int(np.sum(keep))), dtype=float)
    if active_idx.size:
        clean[np.ix_(active, active)] = clean_correlation * np.outer(sd, sd)
        factor[active, :] = sd[:, None] * factor_scaled
    info = {"gram_rank": int(np.sum(keep)),
            "gram_min_eigenvalue": min_eigenvalue,
            "gram_min_correlation_eigenvalue": min_eigenvalue,
            "plugin_joint_r2": explained / var_y,
            "null_c_norm": residual_norm,
            "c_on_zero_variance_scores": int(np.count_nonzero(c[inactive])),
            "gram_psd_projected": projected}
    warnings = []
    if residual_norm > 1e-7 * range_scale or np.any(c[inactive] != 0.0):
        warnings.append("c has sampling signal outside the LD Gram range")
    if explained > var_y * (1.0 + 1e-6) + 1e-12:
        warnings.append("the plug-in c' G+ c exceeds var_y")
    if warnings:
        info["moment_warning"] = "; ".join(warnings)
    return clean, factor, info


def _positive_integer(value, name):
    """Return an integer-valued public argument without silently truncating."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


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


def _ld_variant_count(weights_ld, ld, explicit, label):
    """Infer one LD source's variant count without borrowing a GWAS length."""
    if explicit is not None:
        count = _positive_integer(explicit, label)
        if isinstance(weights_ld, np.ndarray):
            if weights_ld.ndim != 2:
                raise ValueError(f"{label} weights must be two-dimensional")
            if weights_ld.shape[0] != count:
                raise ValueError(
                    f"{label}={count} but dense LD weights have "
                    f"{weights_ld.shape[0]} rows")
        if isinstance(ld, np.ndarray):
            if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
                raise ValueError(f"{label} LD matrix must be square")
            if ld.shape[0] != count:
                raise ValueError(
                    f"{label}={count} but dense LD has {ld.shape[0]} rows")
        return count
    if isinstance(weights_ld, np.ndarray):
        if weights_ld.ndim != 2:
            raise ValueError(f"{label} weights must be two-dimensional")
        return int(weights_ld.shape[0])
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"{label} LD matrix must be square")
        return int(ld.shape[0])
    if isinstance(ld, (list, tuple)) and ld:
        last = np.asarray(ld[-1][1], dtype=np.int64).ravel()
        if last.size:
            return int(last[-1]) + 1
    return None


def _score_cross_moment(weights_gwas, z, n_scores, label):
    """Compute ``W_gwas' z`` on that GWAS's own standardized-genotype basis."""
    rows, cols, vals, m, k = _weight_columns(weights_gwas, int(z.size))
    if m != z.size:
        raise ValueError(f"{label} weights cover {m} variants but z covers "
                         f"{z.size}")
    if k != n_scores:
        raise ValueError(f"{label} weights describe {k} scores but the LD "
                         f"weights describe {n_scores}; score identity and "
                         "column order must agree")
    c = np.zeros(k, dtype=float)
    np.add.at(c, cols, vals * z[rows])
    return c, int(vals.size), m


def _range_basis(gram):
    """Cache the positive-eigenvalue basis of one PSD quadratic form."""
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0,
                np.finfo(float).tiny)
    keep = values > 1e-10 * scale
    return vectors[:, keep], values[keep], scale


def _range_basis_from_factor(factor):
    """Recover the cached eigenbasis carried by _validate_moments' factor."""
    factor = np.asarray(factor, dtype=float)
    values = np.sum(factor * factor, axis=0)
    scale = max(float(np.max(values)) if values.size else 0.0,
                np.finfo(float).tiny)
    keep = values > 1e-10 * scale
    values = values[keep]
    vectors = (factor[:, keep] / np.sqrt(values)[None, :]
               if values.size else np.zeros((factor.shape[0], 0)))
    return vectors, values, scale


def _range_projection(basis, r):
    """Range membership and minimum-norm solve from a cached eigenbasis."""
    vectors, values, scale = basis
    coordinates = vectors.T @ r
    projected = vectors @ coordinates
    residual = float(np.linalg.norm(r - projected))
    tolerance = 1e-8 * max(float(np.linalg.norm(r)), np.sqrt(scale),
                           np.finfo(float).tiny)
    solution = (vectors @ (coordinates / values)
                if values.size else np.zeros_like(r))
    return residual <= tolerance, residual, solution


def _boundedness_context(gram, pf, *, base_basis=None):
    """Cache the two null-space checks needed by every delta/path point."""
    free = np.flatnonzero(pf <= 0.0)
    penalized = np.flatnonzero(pf > 0.0)
    return {"gram": gram,
            "base_basis": (_range_basis(gram)
                           if base_basis is None else base_basis),
            "free": free, "penalized": penalized,
            "free_basis": (_range_basis(gram[np.ix_(free, free)])
                           if free.size else None)}


def _bounded_path_mask(context, r, pf, alpha, lambdas, delta):
    """Conservative boundedness certificate without refactoring ``G``."""
    in_range, residual, _ = _range_projection(context["base_basis"], r)
    if in_range:
        return np.ones(lambdas.size, dtype=bool), residual, 0.0
    free = context["free"]
    beta_free = np.zeros_like(r)
    if free.size:
        free_ok, free_residual, free_solution = _range_projection(
            context["free_basis"], r[free])
        beta_free[free] = free_solution
    else:
        free_ok, free_residual = True, 0.0
    # Positive delta or elastic-net L2 is coercive on every penalized score.
    # The only remaining null directions live wholly in the unpenalized block,
    # whose compatibility was checked once in _boundedness_context.
    if (delta > 0.0 or alpha < 1.0) and free_ok:
        return np.ones(lambdas.size, dtype=bool), residual, free_residual
    if not free_ok or alpha <= 0.0:
        return np.zeros(lambdas.size, dtype=bool), residual, free_residual
    # At and above this KKT threshold the unpenalized-only solution is optimal.
    # Lower pure-lasso objectives may be bounded too, but certifying them needs
    # a null-space LP; reject those points instead of trusting a stalled solve.
    gradient = r - context["gram"] @ beta_free
    penalized = context["penalized"]
    threshold = float(np.max(np.abs(gradient[penalized]) / pf[penalized]))
    safe = lambdas * alpha >= threshold * (1.0 - 1e-12)
    return safe, residual, free_residual

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

    # Parsed once and used for both the Gram and the fit's weight digest: at
    # genome-wide scale these three arrays are the largest allocation here.
    parsed_ld = _weight_columns(weights_ld, n_variants_ld)
    rows_ld, cols_ld, vals_ld, m_ld, _ = parsed_ld
    gram_raw, score_var = _score_gram_from_coo(parsed_ld, ld)
    k = gram_raw.shape[0]

    wz, n_weight_entries_gwas, m_gwas = _score_cross_moment(
        weights_gwas, z, k, "weights_gwas")
    gram_raw, gram_factor_raw, coherence = _validate_moments(
        wz, gram_raw, var_y, label="fitting")
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
        wz_v, n_weight_entries_gwas_valid, m_gwas_valid = _score_cross_moment(
            weights_gwas_valid, z_valid, k, "weights_gwas_valid")
        n_variants_ld_valid = _ld_variant_count(
            weights_ld_valid, ld_valid, n_variants_ld_valid,
            "n_variants_ld_valid")
        parsed_ld_valid = _weight_columns(weights_ld_valid, n_variants_ld_valid)
        _, _, vals_ld_valid, m_ld_valid, k_valid = parsed_ld_valid
        gram_v = _score_gram_from_coo(parsed_ld_valid, ld_valid)[0]
        if k_valid != k:
            raise ValueError(
                f"weights_ld_valid describes {k_valid} scores but weights_ld "
                f"describes {k}; score identity and column order must agree")
        gram_v, gram_v_factor, validation_coherence = _validate_moments(
            wz_v, gram_v, var_y, label="tuning")
        # The TRAINING scale, deliberately. beta comes off a path fitted in
        # training-standardized coordinates, so both halves of
        # (beta'r)^2 / (beta'G beta) must use those same coordinates.
        sel_gram = gram_v * np.outer(scale, scale)
        sel_r_observed = wz_v * scale
        tuning_basis = _range_basis_from_factor(
            gram_v_factor * scale[:, None])[0]
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

    def _score_path(path_d, gram_fit, delta_value):
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
            mse = np.where(valid, mse, np.nan)
            return r2, mse, int(np.sum(~valid))
        per_repeat_r2 = np.full((len(splits), path_d.shape[0]), np.nan)
        per_repeat_mse = np.full((len(splits), path_d.shape[0]), np.nan)
        safe_every_repeat = np.ones(path_d.shape[0], dtype=bool)
        for rep, (c_tr, c_val, _) in enumerate(splits):
            range_vectors = boundedness["base_basis"][0]
            r_train_observed = c_tr * scale
            r_train = range_vectors @ (range_vectors.T @ r_train_observed)
            safe_split, _, _ = _bounded_path_mask(
                boundedness, r_train, pf, alpha, lambdas_d, delta_value)
            safe_split = safe_split[:path_d.shape[0]]
            safe_indices = np.flatnonzero(safe_split)
            if safe_indices.size:
                fitted, n_rep = _coord.enet_path_gaussian(
                    gram_fit, r_train, pf=pf, alpha=alpha,
                    lambdas=lambdas_d[safe_indices], tol=tol,
                    max_iter=max_iter)
                safe_indices = safe_indices[:n_rep]
            else:
                fitted = np.zeros((0, k))
                safe_indices = np.zeros(0, dtype=int)
            fitted_here = np.zeros(path_d.shape[0], dtype=bool)
            r_val_observed = c_val * scale
            r_val = range_vectors @ (range_vectors.T @ r_val_observed)
            if safe_indices.size:
                fitted = fitted[:safe_indices.size]
                split_r2, quadratic = _pseudo_r2_batch(
                    fitted, gram, r_val, var_y)
                fitted_here[safe_indices] = True
                per_repeat_r2[rep, safe_indices] = split_r2
                per_repeat_mse[rep, safe_indices] = (
                    var_y - 2.0 * (fitted @ r_val) + quadratic)
            safe_every_repeat &= fitted_here
        per_repeat_r2[:, ~safe_every_repeat] = np.nan
        per_repeat_mse[:, ~safe_every_repeat] = np.nan
        r2_mean = np.full(path_d.shape[0], np.nan)
        r2_count = np.sum(np.isfinite(per_repeat_r2), axis=0)
        np.divide(np.nansum(per_repeat_r2, axis=0), r2_count, out=r2_mean,
                  where=r2_count > 0)
        mse_mean = np.full(path_d.shape[0], np.nan)
        mse_mean[safe_every_repeat] = np.mean(
            per_repeat_mse[:, safe_every_repeat], axis=0)
        physically_valid = (
            safe_every_repeat
            & (~np.isfinite(r2_mean) | (r2_mean <= 1.0 + 1e-8))
            & (mse_mean >= -1e-8 * var_y))
        r2_mean[~physically_valid] = np.nan
        mse_mean[~physically_valid] = np.nan
        return (r2_mean, mse_mean,
                int(np.sum(~physically_valid)))

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
        path_d, n_fitted = _coord.enet_path_gaussian(
            gram_d, r, pf=pf, alpha=alpha, lambdas=lambdas_d, tol=tol,
            max_iter=max_iter)
        path_d = path_d[:n_fitted]
        lambdas_d = lambdas_d[:n_fitted]
        r2_d, mse_d, n_selection_invalid = _score_path(
            path_d, gram_d, float(delta))
        if np.all(np.isnan(mse_d)):
            delta_audit.append({"delta": float(delta),
                                "n_fitted": int(n_fitted),
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
        raise ValueError(
            "no bounded objective was available on any requested path. Add a "
            "positive ld_shrinkage, remove an incompatible unpenalized score, "
            "or repair the score/LD/GWAS alignment.")
    delta, index, path, lambdas, scores_path, mse_path = best

    log = {"n_scores": k, "n_variants_ld": m_ld,
           "n_variants_gwas": m_gwas, "selection": selection,
           "n_dead": int(np.sum(dead)), "n_lambda": int(lambdas.size),
           "alpha": float(alpha), "var_y": float(var_y),
           "n_weight_entries_ld": int(vals_ld.size),
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
                    "n_weight_entries_ld_valid": int(vals_ld_valid.size),
                    "n_weight_entries_gwas_valid":
                        n_weight_entries_gwas_valid,
                    "tuning_discarded_ld_null_c_norm":
                        tuning_discarded_norm,
                    "tuning_discarded_ld_null_c_fraction":
                        tuning_discarded_fraction,
                    "training_discarded_by_tuning_ld_c_norm":
                        tuning_discarded_training_norm,
                    "training_discarded_by_tuning_ld_c_fraction":
                        tuning_discarded_training_fraction})
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
        weights_ld_digest=_weight_digest(
            rows_ld, cols_ld, vals_ld, m_ld, k), n_variants_ld=m_ld)


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

    Draws ``(c_train, c_val)`` from a joint-Gaussian/CLT plug-in approximation
    to score-space statistics from disjoint subsets of the same ``n``
    individuals. Under joint Gaussian score/phenotype fourth moments, the
    per-individual covariance of ``v_i = s_i y_i`` is

    .. math::

        V_S = \\mathrm{var}(y)\\,G + c c^\\top
            = W^\\top(\\mathrm{var}(y) D + z z^\\top) W,

    This is not an exact identity for arbitrary discrete genotypes and traits;
    it is the Gaussian fourth-moment model with observed ``c`` and external
    ``G`` plugged in. Conditional on that model, drawing directly in ``K``
    dimensions is exactly equivalent to drawing its ``m``-dimensional analogue
    and projecting. That is what makes the approximation cheap.

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
    factor, base_log = _prepare_subsample_score_moments(
        c, gram, var_y, check=check)
    return _draw_subsample_score_moments(
        np.asarray(c, dtype=float).ravel(), n, n_train, float(var_y), factor,
        base_log, np.random.default_rng(rng))


def _prepare_subsample_score_moments(c, gram, var_y, *, check, factor=None):
    """Validate split moments and factor ``G`` once for any number of draws."""
    c = np.asarray(c, dtype=float).ravel()
    gram = np.asarray(gram, dtype=float)
    k = c.size
    if gram.shape != (k, k):
        raise ValueError(f"c is length {k}, so gram must be ({k}, {k}), got "
                         f"{gram.shape}")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(gram)):
        raise ValueError("c and gram must be finite")

    diag = np.diag(gram)
    log = {"var_y": var_y}
    if check:
        # This cheap componentwise diagnostic often identifies a scale mistake,
        # but an estimated c can exceed its population bound through sampling
        # noise. Retain the evidence without rejecting a valid noisy GWAS.
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.where(diag > 0, c * c / (var_y * diag), 0.0)
        bad = np.flatnonzero(implied > 1.0 + 1e-8)
        if bad.size:
            log["warning"] = (
                f"{bad.size} score(s) have a plug-in single-score R2 above 1 "
                f"(largest {implied.max():.3g}, score index "
                f"{int(np.argmax(implied))}); this can be sampling noise, but "
                "also warrants checking z scaling, var_y, allele alignment, "
                "sample overlap, and the LD reference")
        log["max_implied_r2"] = float(implied.max()) if k else 0.0

    if factor is None and check:
        _, factor, coherence = _validate_moments(
            c, gram, var_y, label="PUMAS")
        rank = factor.shape[1]
        log.update(coherence)
    elif factor is None:
        factor, rank, negative_mass = _psd_sqrt(gram)
        if negative_mass > 1e-8:
            raise ValueError(
                "gram is materially indefinite, so a PUMAS covariance cannot "
                "be sampled safely; rebuild the LD reference at adequate "
                "precision or with an explicit ridge")
    else:
        factor = np.asarray(factor, dtype=float)
        if factor.ndim != 2 or factor.shape[0] != k or \
                not np.all(np.isfinite(factor)):
            raise ValueError("the cached Gram factor is incompatible with c")
        rank = factor.shape[1]
        negative_mass = 0.0
    log["rank"] = int(rank)
    log["n_scores"] = k
    if rank < k:
        log["rank_warning"] = (
            f"the Gram has rank {rank} for {k} scores, so in {k - rank} "
            "direction(s) its var_y * G noise term is zero. The c c' term can "
            "still carry noise along c, but a larger LD reference is "
            "preferable.")
    return factor, log


def _draw_subsample_score_moments(c, n, n_train, var_y, factor, base_log,
                                  rng):
    """Draw one complementary pseudo-split from a prepared score factor."""
    n = float(n)
    n_train = float(n_train)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("n must be finite and positive")
    if not np.isfinite(n_train) or not 0.0 < n_train < n:
        raise ValueError(f"n_train must satisfy 0 < n_train < n = {n}, got "
                         f"{n_train}")
    n_val = n - n_train
    log = dict(base_log)
    log.update({"n": n, "n_train": n_train, "n_val": n_val,
                "var_y": var_y})
    kappa = 1.0 / n_train - 1.0 / n
    # Relative to the Gaussian plug-in covariance, this factor-plus-rank-one
    # draw is exact; V_S itself is never formed.
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
        """Whether this is clean validation after every model choice was fixed."""
        return self.regime == "A"

    @property
    def is_assessment(self):
        """Whether this is clean assessment after every model choice was fixed."""
        return self.regime == "A"

    def summary(self):
        lines = [f"summary-statistic evaluation (regime {self.regime}): "
                 f"R2 {self.r2:.4f}, MSE {self.mse:.4f}",
                 f"  {REGIMES.get(self.regime, 'unknown provenance')}"]
        if self.regime == "B":
            lines.append("  this was used for tuning — do not report it as a "
                         "clean assessment")
        elif not self.is_validation:
            lines.append("  this is not a validation — do not report it as one")
        if self.log.get("warning"):
            lines.append(f"  {self.log['warning']}")
        for key in ("moment_warning", "mse_moment_warning"):
            if self.log.get(key):
                lines.append(f"  {self.log[key]}")
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
        The combination to score, on the raw score scale.
    c_eval, gram_eval : array_like
        Score-space moments from the **evaluation** data: ``W^T z_eval`` and
        ``W^T D_eval W``, on the same score scaling ``beta`` is on.
        The metric projects ``c_eval`` onto the scale-invariant positive range
        of ``gram_eval`` and logs the discarded finite-reference null component.
    var_y : float
        Phenotype variance on the scale ``c_eval`` was formed on. The MSE is
        meaningless if this is wrong, and the R² is off by a constant factor.
    regime : {"A", "B", "C"}, optional
        Declare the provenance. Omitted, it is inferred: identical to the
        fitting moments means regime C.
    fitted_on : array_like, optional
        The observed or identifiable ``c`` the combination was fitted on, so
        that regime C can be detected rather than trusted. A
        :class:`SumstatFit` automatically recognizes both its ``c_raw`` and
        projected ``r``.

    Returns
    -------
    SumstatEval
    """
    if isinstance(beta, SumstatFit):
        if fitted_on is None:
            c_candidate = np.asarray(c_eval, dtype=float)
            fitted_on = beta.c_raw
            if (c_candidate.shape == np.asarray(beta.r).shape
                    and np.array_equal(c_candidate, np.asarray(beta.r))):
                fitted_on = beta.r
        beta = beta.beta
    beta = np.asarray(beta, dtype=float).ravel()
    c_eval = np.asarray(c_eval, dtype=float).ravel()
    gram_eval = np.asarray(gram_eval, dtype=float)
    k = beta.size
    if c_eval.size != k or gram_eval.shape != (k, k):
        raise ValueError(f"beta is length {k}, so c_eval must be ({k},) and "
                         f"gram_eval ({k}, {k}); got {c_eval.shape} and "
                         f"{gram_eval.shape}")
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(c_eval))
            and np.all(np.isfinite(gram_eval))):
        raise ValueError("beta, c_eval and gram_eval must be finite")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")

    same = (fitted_on is not None
            and np.asarray(fitted_on).shape == c_eval.shape
            and np.array_equal(np.asarray(fitted_on, dtype=float), c_eval))
    if regime is None:
        # Only regime C is inferable: it is the one with an observable
        # signature, namely evaluation moments identical to the fitting ones.
        # "Not C" does NOT imply A — a PUMAS pseudo-validation split is also
        # not identical to its pseudo-training split, and defaulting to A there
        # would stamp "clean external validation" on a regime B number. An
        # unprovable label is worse than none, so this refuses instead.
        if not same:
            raise ValueError(
                "cannot infer the evaluation regime: these moments differ from "
                "the ones the combination was fitted on, which rules out C but "
                "does not distinguish A (an untouched assessment GWAS) from B "
                "(tuning or pseudo-validation moments). Pass regime='A' or "
                "regime='B' "
                "explicitly — the two are not interchangeable and the "
                "difference is invisible in the number.")
        regime = "C"
    regime = str(regime).upper()
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got "
                         f"{regime!r}")
    if same and regime in ("A", "B"):
        raise ValueError(
            f"regime {regime} is incompatible with evaluation moments equal "
            "to the fitting moments; this is regime C same-data evaluation")
    if fitted_on is not None and not same and regime == "C":
        raise ValueError(
            "regime C was declared, but c_eval differs from the fitting "
            "moments; declare A or B according to its actual role")

    # A fixed vector uses only one scalar direction. An unused noisy/null score
    # or an indefinite unused subspace must not invalidate that direction.
    observed_beta_c = float(beta @ c_eval)
    c_identifiable, discarded_c_norm, discarded_c_fraction = (
        _project_c_to_gram_range(c_eval, gram_eval))
    num, quad, quad_tol, gram_eval, gram_asymmetry = (
        _directional_score_moments(
            beta, gram_eval, c_identifiable, var_y, "gram_eval"))
    r2 = (float("nan") if quad <= quad_tol
          else (num * num) / (quad * var_y))
    mse = var_y - 2.0 * num + quad

    log = {"n_nonzero": int(np.sum(beta != 0.0)),
           "beta_c": num, "beta_G_beta": quad,
           "observed_beta_c": observed_beta_c,
           "discarded_beta_c_null": observed_beta_c - num,
           "discarded_c_null_norm": discarded_c_norm,
           "discarded_c_null_fraction": discarded_c_fraction,
           "gram_asymmetry": gram_asymmetry,
           "regime_detail": REGIMES[regime]}
    if regime == "C":
        log["warning"] = ("evaluated on the same summary statistics it was "
                          "fitted on; this is an upper bound, not a validation")
    if np.isfinite(r2) and r2 > 1.0:
        log["moment_warning"] = (
            f"plug-in evaluation R2 is {r2:.6g}, above 1; external moments "
            "are noisy, but check scaling, alignment, var_y, and the LD source")
    if mse < 0.0:
        log["mse_moment_warning"] = (
            f"plug-in MSE is {mse:.6g}, below zero; retain it only as a noisy "
            "summary-moment diagnostic")
    if mse > var_y:
        log["mse_warning"] = ("MSE exceeds var(y): this combination predicts "
                              "worse than the mean")
    return SumstatEval(r2=r2, mse=mse, regime=regime, var_y=var_y, n_scores=k,
                       log=log)


def score_moments(weights_ld, z, ld, *, weights_gwas=None,
                  n_variants_ld=None):
    """The score-space moments ``(c, G)`` for one set of summary statistics.

    The pair that :func:`evaluate_sumstat` scores against, and the same pair
    :func:`multi_pgs_sumstats` fits from. Building them for an *evaluation* GWAS
    is how a combination gets an honest regime A number. ``weights_gwas`` and
    ``weights_ld`` represent the same raw component scores, but each is
    multiplied by the empirical genotype SD of its own dataset:
    ``c = W_gwas.T @ z`` and ``G = W_ld.T @ D @ W_ld``. They may cover different
    variant sets; only their score columns must agree.
    """
    if weights_gwas is None:
        raise ValueError(
            "weights_gwas is required separately from weights_ld; pass the "
            "same matrix explicitly only when GWAS and LD genotype scales are "
            "genuinely identical")
    z = np.asarray(z, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")
    n_variants_ld = _ld_variant_count(
        weights_ld, ld, n_variants_ld, "n_variants_ld")
    gram, var = score_gram(weights_ld, ld, n_variants=n_variants_ld)
    c, _, _ = _score_cross_moment(
        weights_gwas, z, gram.shape[0], "weights_gwas")
    return c, gram, var


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_to_reference(scoring_files, variants, *, sd=None, af=None,
                       hwe_genotype_sd=False, drop_ambiguous=True,
                       on_error="raise", progress=None):
    """Align PGS Catalog scoring files to an LD reference's variant table.

    Returns weights on the **standardized** genotype scale, which is what
    :func:`score_gram` needs and what catalog files are *not* on: a catalog
    weight multiplies a raw allele count, and converting requires the empirical
    dosage standard deviation used to construct the LD reference. HWE
    ``sqrt(2 f (1-f))`` is available only as an explicit approximation because
    it ignores imputation uncertainty and departures from HWE.

    Parameters
    ----------
    scoring_files : sequence of str or ScoringFile
    variants : mapping
        The LD reference's variant table, with ``id chrom pos a1 a2``, in the
        reference's own order — the row order of ``D``.
    sd : array_like, optional
        Empirical dosage standard deviation per reference variant. Preferred.
    af : array_like, optional
        Reference allele frequency of ``a1`` per variant. Used only when
        ``hwe_genotype_sd=True``.
    hwe_genotype_sd : bool
        Explicitly approximate dosage SD by ``sqrt(2 f (1-f))`` from ``af``.
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
    scale_sd = None
    scale_source = None
    if sd is not None:
        scale_sd = np.asarray(sd, dtype=float).ravel()
        if scale_sd.size != n_variants:
            raise ValueError(f"sd has {scale_sd.size} entries for {n_variants} "
                             "reference variants")
        if not np.all(np.isfinite(scale_sd)) or np.any(scale_sd < 0.0):
            raise ValueError("sd must be finite and non-negative")
        scale_source = "empirical_sd"
    if af is not None:
        f = np.asarray(af, dtype=float).ravel()
        if f.size != n_variants:
            raise ValueError(f"af has {f.size} entries for {n_variants} "
                             "reference variants")
        if not np.all(np.isfinite(f)) or np.any((f < 0.0) | (f > 1.0)):
            raise ValueError("af must be finite and lie in [0, 1]")
        if hwe_genotype_sd:
            if scale_sd is not None:
                raise ValueError("give sd or request HWE scaling from af, not both")
            scale_sd = np.sqrt(2.0 * f * (1.0 - f))
            scale_source = "hwe_from_af"
    elif hwe_genotype_sd:
        raise ValueError("hwe_genotype_sd=True requires af")
    if af is not None and scale_sd is None:
        raise ValueError("af alone does not define empirical dosage SD; pass sd, "
                         "or set hwe_genotype_sd=True to request the HWE "
                         "approximation explicitly")

    files = list(scoring_files)
    pairs, ids, errors = [], [], {}
    matched = []
    for i, item in enumerate(files):
        label = getattr(item, "pgs_id", None) or str(item)
        try:
            scoring = item if hasattr(item, "weight") else read_scoring_file(item)
            idx, w, log = harmonize_scoring_file(scoring, variants,
                                                 drop_ambiguous=drop_ambiguous)
            if scale_sd is not None:
                # A catalog weight counts alleles; on standardized genotypes the
                # same score is w * sd. A monomorphic reference variant has
                # sd = 0 and contributes nothing, which is the truth here.
                w = w * scale_sd[idx]
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
           "standardized": scale_sd is not None,
           "scale_source": scale_source}
    if matched:
        log["n_matched_median"] = int(np.median(matched))
        log["n_matched_min"] = int(min(matched))
    if errors:
        log["errors"] = errors
    if scale_source == "hwe_from_af":
        log["warning"] = (
            "weights used HWE sqrt(2 f (1-f)) rather than empirical dosage SD; "
            "this is an approximation and may be wrong for imputed variants")
    elif scale_sd is None:
        log["warning"] = (
            "no empirical dosage SD was supplied and no HWE conversion was "
            "requested, so catalog weights were not converted to the "
            "standardized-genotype scale; this is correct only for weights "
            "that were already on it")
    return pairs, ids, log
