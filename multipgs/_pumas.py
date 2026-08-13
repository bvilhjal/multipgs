"""PUMAS-style score-space subsampling of summary-statistic moments."""

from __future__ import annotations

import numpy as np

from ._moments import _validate_moments

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
