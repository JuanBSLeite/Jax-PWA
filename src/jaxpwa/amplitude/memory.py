"""Device-memory-aware sizing of normalization chunks.

With floating dynamics the reverse-mode AD pass through the normalization grid
needs memory proportional to the number of points in one chunk. ``"auto"``
chunk sizing measures that cost instead of guessing it: it compiles (without
running) a small probe of the worst-case program, reads the compile-time
temporary memory XLA reports, and divides the free device memory by the
resulting bytes per point. The scaling is linear in the number of points, so
one probe is enough. The prepared normalization blocks stay resident for the
whole fit whatever the chunk size, so their size is subtracted from the budget
before the temporary buffers are sized.
"""

from __future__ import annotations

import jax

# Share of the *free* device memory (``bytes_limit - bytes_in_use``) that the
# normalization may use: its resident prepared blocks plus the temporary
# buffers of one chunk. The rest is
# headroom for the allocator pool, the CUDA context, autotuning scratch space
# and everything allocated after preparation (data amplitudes, the Hessian).
MEMORY_FRACTION = 0.5

# Points evaluated by the probe; large enough for a stable bytes-per-point
# slope, small enough to compile quickly.
PROBE_POINTS = 20_000

# A sample this small always fits in one block, so no probe is compiled.
SINGLE_BLOCK_POINTS = 20_000

# Never shrink chunks below this: tiny chunks only add scan overhead.
MIN_CHUNK_POINTS = 2_048


def device_memory_budget() -> int | None:
    """Bytes a normalization chunk may use, or ``None`` if it cannot be known.

    ``None`` is returned for devices without memory statistics (the CPU
    backend, or ``XLA_PYTHON_CLIENT_ALLOCATOR=platform``).
    """
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats or "bytes_limit" not in stats:
        return None
    free = int(stats["bytes_limit"]) - int(stats.get("bytes_in_use", 0))
    return max(int(MEMORY_FRACTION * free), 0)


def compiled_temp_bytes(function, *arguments) -> int | None:
    """Temporary bytes of ``function`` as compiled for ``arguments``.

    Nothing is executed. ``None`` if the backend reports no analysis.
    """
    try:
        analysis = jax.jit(function).lower(*arguments).compile().memory_analysis()
    except Exception:
        return None
    if analysis is None:
        return None
    return int(analysis.temp_size_in_bytes)


def tree_nbytes(tree) -> int:
    """Total bytes of the array leaves of ``tree``."""
    return int(
        sum(
            int(getattr(leaf, "nbytes", 0))
            for leaf in jax.tree_util.tree_leaves(tree)
        )
    )


def chunk_points_from_budget(budget: int, bytes_per_point: float, n_points: int) -> int:
    """Largest chunk whose temporary memory fits ``budget``, within bounds."""
    if bytes_per_point <= 0:
        return int(n_points)
    points = int(budget // bytes_per_point)
    return int(min(max(points, MIN_CHUNK_POINTS), n_points))
