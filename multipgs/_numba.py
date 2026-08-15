"""Optional Numba decorators used by Multipgs' own numerical kernels."""

from __future__ import annotations

import warnings

__all__ = ["HAVE_NUMBA", "_jit_nogil", "_jit_parallel", "prange",
           "warn_no_numba"]

_warned_no_numba = False


def warn_no_numba():
    """Warn once per process that the pure-Python solver fallback is in force.

    Without Numba the coordinate-descent and Gram kernels fall back to NumPy
    twins that compute the same answers, just much slower, and nothing else
    signals the difference. The public fit entries call this so an accidental
    no-Numba run announces itself.
    """
    global _warned_no_numba
    if HAVE_NUMBA or _warned_no_numba:
        return
    _warned_no_numba = True
    warnings.warn(
        "Numba is not installed, so multipgs' solver kernels run their NumPy "
        "fallbacks; large stacking fits and Gram builds can be much slower. "
        "Install Numba (the [fast] extra) to restore the compiled paths.",
        RuntimeWarning, stacklevel=2)

try:
    from numba import njit as _njit, prange

    HAVE_NUMBA = True

    def _jit_nogil(func):
        return _njit(cache=True, nogil=True)(func)

    def _jit_parallel(func):
        return _njit(cache=True, parallel=True)(func)

except ImportError:  # pragma: no cover - exercised by the no-Numba CI leg
    HAVE_NUMBA = False
    prange = range

    def _jit_nogil(func):
        return func

    def _jit_parallel(func):
        return func
