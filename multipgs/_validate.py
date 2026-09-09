"""Shared argument validators used by more than one fitting route."""

from __future__ import annotations

import numpy as np

_LABEL = {0: "non-negative", 1: "positive"}


def _integer_argument(value, name, *, minimum):
    """Return an integer-valued public argument without truncating it.

    One implementation for both bounds, and deliberately stricter than
    ``int(value)``: a Boolean is not a count, a size-1 array is not a scalar,
    and a numeric *string* is not an integer. ``float(value)`` alone accepts
    ``"3"``, which then reaches a solver control as a silently coerced value.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a {_LABEL[minimum]} integer")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a {_LABEL[minimum]} integer")
    if raw.dtype.kind == "u" and int(raw) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside int64 range")
    numeric = float(raw)
    if (not np.isfinite(numeric) or numeric < minimum
            or not numeric.is_integer()
            or numeric > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be a {_LABEL[minimum]} integer")
    return int(numeric)


def _positive_integer(value, name):
    """Return a strictly positive integer-valued public argument."""
    return _integer_argument(value, name, minimum=1)


def _nonnegative_integer(value, name):
    """Return a non-negative integer-valued public argument."""
    return _integer_argument(value, name, minimum=0)
