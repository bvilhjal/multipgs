"""Gaussian sufficient statistics for CMSA fold reuse.

Moments are accumulated relative to a fixed origin so a parent Gram and a
held-out Gram can be subtracted without the ``E[X**2] - E[X]**2`` cancellation.
The origin is a function of ``X`` only: ``origin_y`` is always zero, so an
outer-assessment phenotype cannot leak into training statistics through the
numerical origin.
"""

from __future__ import annotations

import numpy as np


def _standardize(X, idx):
    """Column means and standard deviations over rows ``idx``.

    A constant column gets ``scale = 1``, so it standardizes to all-zero and
    can never enter the model. That is the intended handling: a score with no
    variance in the training set carries no information, and dividing by its
    zero standard deviation would poison the whole fit.
    """
    sub = X if idx is None else X[idx]
    center = sub.mean(axis=0)
    scale = sub.std(axis=0)
    dead = scale <= 1e-12
    scale = np.where(dead, 1.0, scale)
    return center, scale, dead


def _gaussian_stats(X, y, idx=None):
    """Stable sufficient statistics for a Gaussian fit on ``idx``."""
    return _gaussian_stats_at_origin(X, y, idx)


def _gaussian_stats_at_origin(X, y, idx=None, *, reference=None):
    """Implementation allowing child statistics to share a parent origin.

    ``origin_x`` is the first row of ``X`` (or the parent's first row) so a
    large column offset does not erase ordinary variance. ``origin_y`` is
    identically zero: a phenotype origin taken from a held-out row would
    cancel out of ``r`` only in exact arithmetic, and can move the last bits
    of a nested-assessment path.
    """
    rows = None if idx is None else np.asarray(idx, dtype=np.int64).ravel()
    n = int(X.shape[0] if rows is None else rows.size)
    if n == 0:
        raise ValueError("cannot form Gaussian statistics from zero rows")
    first = 0 if rows is None else int(rows[0])
    if reference is None:
        origin_x = np.array(X[first], dtype=float, copy=True)
        origin_y = 0.0
    else:
        origin_x = reference["origin_x"]
        origin_y = reference["origin_y"]

    D = X.shape[1]
    sum_xc = np.zeros(D)
    xtx_c = np.zeros((D, D))
    sum_yc = 0.0
    xty_c = np.zeros(D)
    step = max(1, 2_000_000 // max(D, 1))
    for start in range(0, n, step):
        stop = min(start + step, n)
        take = slice(start, stop) if rows is None else rows[start:stop]
        Xc = np.array(X[take], dtype=float, copy=True)
        Xc -= origin_x
        yc = np.asarray(y[take], dtype=float) - origin_y
        sum_xc += np.sum(Xc, axis=0)
        xtx_c += Xc.T @ Xc
        sum_yc += float(np.sum(yc))
        xty_c += Xc.T @ yc
    return {
        "n": n,
        "origin_x": origin_x,
        "sum_xc": sum_xc,
        "xtx_c": (xtx_c + xtx_c.T) * 0.5,
        "origin_y": origin_y,
        "sum_yc": sum_yc,
        "xty_c": xty_c,
    }


def _replace_gaussian_y_moments(stats, X, y, idx):
    """Overwrite phenotype moments from ``idx`` at ``stats``' origin.

    Subtracting ``X'y`` after the held-out phenotype changes is exact only in
    real arithmetic: a large outer ``y`` cancels out of the parent cross-product
    and can move the last bits of training ``r``. The Gram is reused; these
    ``O(n D)`` sums are formed on the training rows so an outer outcome cannot
    leak into the path.
    """
    origin_x = stats["origin_x"]
    origin_y = stats["origin_y"]
    rows = np.asarray(idx, dtype=np.int64).ravel()
    if rows.size != int(stats["n"]):
        raise ValueError("y-moment rows do not match the Gaussian training size")
    D = X.shape[1]
    sum_yc = 0.0
    xty_c = np.zeros(D)
    step = max(1, 2_000_000 // max(D, 1))
    for start in range(0, rows.size, step):
        stop = min(start + step, rows.size)
        take = rows[start:stop]
        Xc = np.array(X[take], dtype=float, copy=True)
        Xc -= origin_x
        yc = np.asarray(y[take], dtype=float) - origin_y
        sum_yc += float(np.sum(yc))
        xty_c += Xc.T @ yc
    stats["sum_yc"] = sum_yc
    stats["xty_c"] = xty_c
    return stats


def _subtract_gaussian_stats(total, held_out):
    """Sufficient statistics for ``total \\ held_out``."""
    n = int(total["n"] - held_out["n"])
    if n <= 0:
        raise ValueError("held-out rows exhaust the Gaussian training set")
    if (not np.array_equal(total["origin_x"], held_out["origin_x"])
            or total["origin_y"] != held_out["origin_y"]):
        raise ValueError("Gaussian statistics must share a centering origin")
    return {
        "n": n,
        "origin_x": total["origin_x"],
        "sum_xc": total["sum_xc"] - held_out["sum_xc"],
        "xtx_c": total["xtx_c"] - held_out["xtx_c"],
        "origin_y": total["origin_y"],
        "sum_yc": float(total["sum_yc"] - held_out["sum_yc"]),
        "xty_c": total["xty_c"] - held_out["xty_c"],
    }


def _gaussian_system(stats):
    """Standardization, Gram and response covariance from raw statistics."""
    n = stats["n"]
    mean_xc = stats["sum_xc"] / n
    center = stats["origin_x"] + mean_xc
    var = np.diag(stats["xtx_c"]) / n - mean_xc * mean_xc
    # Subtraction of two large cross-products can leave tiny negative round-off.
    scale0 = np.sqrt(np.maximum(var, 0.0))
    dead = scale0 <= 1e-12
    scale = np.where(dead, 1.0, scale0)
    cov = stats["xtx_c"] / n - np.outer(mean_xc, mean_xc)
    G = cov / scale[:, None] / scale[None, :]
    mean_yc = stats["sum_yc"] / n
    ybar = stats["origin_y"] + mean_yc
    r = (stats["xty_c"] / n - mean_xc * mean_yc) / scale
    if dead.any():
        G[dead, :] = 0.0
        G[:, dead] = 0.0
        r[dead] = 0.0
    G = (G + G.T) * 0.5
    return center, scale, dead, G, r, float(ybar)
