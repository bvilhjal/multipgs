"""Optional Numba decorators used by Multipgs' own numerical kernels.

The decorators, ``prange`` and ``HAVE_NUMBA`` are LDpred3's, imported from
its published ``ldpred3.shim`` code-level surface: one implementation across
the sibling packages instead of a private copy of the same try/except shim.
``warn_no_numba`` stays local — its message names multipgs' own kernels and
extras — but shares the once-per-process semantics.
"""

from __future__ import annotations

import warnings

from ldpred3.shim import HAVE_NUMBA, _jit_nogil, _jit_parallel, prange

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
