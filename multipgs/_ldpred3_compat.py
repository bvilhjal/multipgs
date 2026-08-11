"""Lazy compatibility seam for the pinned :mod:`ldpred3` dependency.

The names here are private helpers that multipgs deliberately shares with the
ldpred3 version pinned in ``pyproject.toml``. Keeping the imports in one module
makes the next dependency review a single explicit audit, and keeps
``import multipgs`` from pulling in ldpred3's Numba kernels: nothing is
imported until it is used.

Public ldpred3 names (``harmonize``, ``read_plink``, ``run_ldpred3_prs``, ...)
are imported directly from :mod:`ldpred3` by their consumers and are *not*
routed through here — this module is only for the private surface.
"""

from __future__ import annotations

import importlib


_MODULE_NAMES = {
    # The JIT shim. multipgs writes its coordinate-descent kernels in the same
    # dual-path style as ldpred3 (explicit-loop kernel under Numba, vectorised
    # NumPy fallback without it) and reuses the decorators rather than
    # maintaining a second, subtly different import guard.
    "ldpred3._numba": ("HAVE_NUMBA", "_jit_fastmath_nogil", "_jit_nogil"),
    # ldpred3 documents dequantize_ld as "the single place every LD consumer
    # that reads block values as float should route through", but does not list
    # it in ``ld_repr.__all__``. multipgs is such a consumer: score_gram reads a
    # block's low-rank factor directly to skip a back-projection, and must not
    # read an int8 factor at 127x its true magnitude. Routed through the seam
    # rather than imported at the use site, so the next dependency review sees
    # it. :class:`ldpred3.LowRankLD` itself is public and imported normally.
    "ldpred3.ld_repr": ("dequantize_ld",),
}

_NAME_TO_MODULE = {
    name: module for module, names in _MODULE_NAMES.items() for name in names
}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name):
    module = _NAME_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return list(__all__)
