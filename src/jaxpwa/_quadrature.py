"""Bounded host-side reuse of fixed quadrature rules."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only NumPy nodes/weights on [-1, 1], shared by quadrature users.

    Cache only the one-dimensional rule, never a model, device array or full
    integration grid. Scaling to an interval creates fresh arrays at call sites.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights
