"""Optional Numba decorators used by Multipgs' own numerical kernels."""

from __future__ import annotations

__all__ = ["HAVE_NUMBA", "_jit_nogil", "_jit_parallel", "prange"]

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
