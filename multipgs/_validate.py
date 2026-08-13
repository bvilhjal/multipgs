"""Shared argument validators used by more than one fitting route."""

from __future__ import annotations

import numpy as np


def _positive_integer(value, name):
    """Return an integer-valued public argument without truncating it."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_integer(value, name):
    """Return a non-negative integer-valued public argument."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative integer") from None
    if not np.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)
