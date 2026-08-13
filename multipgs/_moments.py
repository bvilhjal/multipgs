"""Score-space moment validation, spectral projection, and path scoring."""

from __future__ import annotations

import numpy as np

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


def _correlation_factorization(gram):
    """One scale-invariant eigendecomposition of a Gram.

    Work in correlation coordinates so a score's arbitrary units cannot make a
    valid direction look null. Callers that need the projection, the cleaned
    covariance, or the factor all consume this result instead of factorizing
    again.
    """
    symmetric = 0.5 * (np.asarray(gram, dtype=float) + np.asarray(gram, dtype=float).T)
    diagonal = np.diag(symmetric).copy()
    active = diagonal > 0.0
    if not np.any(active):
        empty = np.zeros((0, 0), dtype=float)
        return {
            "symmetric": symmetric, "diagonal": diagonal, "active": active,
            "sd": np.zeros(0, dtype=float), "correlation": empty,
            "values": np.zeros(0, dtype=float),
            "vectors": empty, "keep": np.zeros(0, dtype=bool),
            "scale": np.finfo(float).tiny,
        }
    sd = np.sqrt(diagonal[active])
    correlation = symmetric[np.ix_(active, active)] / np.outer(sd, sd)
    correlation = 0.5 * (correlation + correlation.T)
    correlation[np.diag_indices_from(correlation)] = 1.0
    values, vectors = np.linalg.eigh(correlation)
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0,
                np.finfo(float).tiny)
    keep = values > 1e-10 * scale
    return {
        "symmetric": symmetric, "diagonal": diagonal, "active": active,
        "sd": sd, "correlation": correlation, "values": values,
        "vectors": vectors, "keep": keep, "scale": scale,
    }


def _project_c_to_gram_range(c, gram, *, factor=None):
    """Project ``c`` onto identifiable covariance directions, scale-invariantly."""
    c = np.asarray(c, dtype=float).ravel()
    spec = factor if factor is not None else _correlation_factorization(gram)
    active = spec["active"]
    projected = np.zeros_like(c)
    n_active = int(np.sum(active))
    c_scaled = np.zeros(n_active, dtype=float)
    projected_scaled = np.zeros_like(c_scaled)
    if n_active:
        c_scaled = c[active] / spec["sd"]
        if np.any(spec["keep"]):
            basis = spec["vectors"][:, spec["keep"]]
            projected_scaled = basis @ (basis.T @ c_scaled)
        projected[active] = projected_scaled * spec["sd"]
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
    """Whether a path point has a defined directional plug-in objective."""
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

    A candidate is invalid only when its own quadratic direction is materially
    indefinite, or has zero variance but non-zero covariance.  Sampling noise
    can make an otherwise defined plug-in R2 exceed one or its plug-in MSE fall
    below zero.  Those population-bound violations remain rankable diagnostics;
    rejecting them here used to turn a mildly noisy external moment into the
    null model while :func:`_validate_moments` explicitly promised not to police
    it.
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
    valid = ~impossible
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
    A fixed vector uses the observed scalar ``beta.T @ r`` directly. Components
    of ``r`` attached to unused correlated scores therefore cannot change the
    answer. If the selected score direction itself has numerical zero variance,
    the R2 is undefined regardless of its noisy observed covariance.
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
    observed_num = float(beta @ r)
    # Validate the quadratic direction independently of the noisy covariance.
    # Passing zero here lets a null direction return nan below rather than
    # raising because an external c happens not to share the finite LD nullspace.
    _, den, den_tol, _, _ = _directional_score_moments(
        beta, gram, np.zeros_like(r), var_y, "gram")
    if den <= den_tol:
        return float("nan")
    return (observed_num * observed_num) / (den * var_y)


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

def _validate_moments(c, gram, var_y, *, label, prepared=None):
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
    # Rank and definiteness must not depend on the arbitrary units of a score.
    # Work on correlation coordinates, then map the cleaned covariance and its
    # factor back to raw units. An eigentolerance on raw G would incorrectly
    # call a perfectly valid score null merely because its weights were scaled
    # by (say) 1e-6.
    if prepared is not None and "factorization" in prepared:
        # The caller has proved that this is the same LD-basis Gram (typically
        # training and independent tuning sharing one reference). Its spectral
        # decomposition is independent of c and can be reused exactly.
        spec = prepared["factorization"]
    else:
        spec = _correlation_factorization(gram)
    symmetric = spec["symmetric"]
    diagonal = spec["diagonal"]
    diagonal_scale = max(float(np.max(np.abs(diagonal))),
                         np.finfo(float).tiny)
    materially_negative = diagonal < -1e-12 * diagonal_scale
    if np.any(materially_negative):
        j = int(np.flatnonzero(materially_negative)[0])
        raise ValueError(
            f"{label} gram is materially indefinite: it has negative variance "
            f"{diagonal[j]:.3g} at score {j}")
    near_negative = (diagonal < 0.0) & ~materially_negative
    if np.any(near_negative):
        diagonal = diagonal.copy()
        diagonal[near_negative] = 0.0
        spec = _correlation_factorization(
            np.where(np.eye(diagonal.size, dtype=bool), diagonal, symmetric))
        symmetric = spec["symmetric"]
    if prepared is not None:
        prepared["factorization"] = spec
    active = spec["active"]
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

    sd = spec["sd"]
    correlation = spec["correlation"]
    c_scaled = c[active] / sd if sd.size else np.zeros(0, dtype=float)
    values, vectors = spec["values"], spec["vectors"]
    spectral_scale = spec["scale"]
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
    if np.any(active):
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
