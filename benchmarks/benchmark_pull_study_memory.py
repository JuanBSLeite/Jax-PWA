"""Measure the notebook-23 floating-mass CP fit on its actual input files.

Run in a fresh process on the target GPU (no CPU fallback with --require-gpu).
Only the setup cells and build_toy_fit_session definition are executed; the
notebook's environment cell and multi-toy fit/CSV-writing loop are not executed.
The notebook is trusted Python code and must have the notebook-23 cell layout.
The default entry range includes toy 0 and checks that its end was reached.

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
    python benchmarks/benchmark_pull_study_memory.py --require-gpu

The default measures preparation and the value/gradient. Add --hessian for the
actual Minimizer automatic Hessian, or --fit-ncall N for an actual numerical-
HESSE fit (with --hessian it instead uses the automatic Hessian). A call limit
can stop before convergence; this is a memory diagnostic, not a pull result.
Allocator statistics exclude some CUDA/runtime overhead. Host ROOT payload is
reported separately from device allocations. No fit results are written.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import uproot
from benchmark_fit_evaluation import _block_tree

from jaxpwa import read_root_tree

DEFAULT_NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/_data_analyses"
    / "23_b2pipipi_cpvfit_toy_pull_study_SqDP_FreeMasses.ipynb"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--hessian", action="store_true")
    parser.add_argument("--fit-ncall", type=int, default=0)
    parser.add_argument("--entry-stop", type=int, default=200_000)
    parser.add_argument("--normalization-resolution", type=int)
    parser.add_argument(
        "--normalization-chunk-size",
        default="auto",
        help="positive integer or 'auto' (memory-aware, the default)",
    )
    parser.add_argument("--check-hessian", action="store_true")
    args = parser.parse_args()
    if args.fit_ncall < 0 or args.entry_stop < 1:
        parser.error("--fit-ncall must be nonnegative and --entry-stop positive")
    if args.check_hessian and not args.hessian:
        parser.error("--check-hessian requires --hessian")
    device = jax.devices()[0]
    if args.require_gpu and device.platform != "gpu":
        raise RuntimeError(f"GPU required, got {device}")
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default")
    print(
        json.dumps(
            {
                "device": str(device),
                "device_kind": device.device_kind,
                "jax_version": jax.__version__,
                "x64": jax.config.x64_enabled,
                "allocator": allocator,
                "preallocate": os.environ.get(
                    "XLA_PYTHON_CLIENT_PREALLOCATE", "default"
                ),
            }
        ),
        flush=True,
    )
    cells = json.loads(args.notebook.read_text())["cells"]
    scope = {"__name__": "__memory_benchmark__"}
    stages = []
    started = perf_counter()

    def report(stage):
        # Synchronize arrays before reading allocator counters.
        for array in jax.live_arrays():
            array.block_until_ready()
        stats = device.memory_stats()
        payload = {
            "stage": stage,
            "elapsed_seconds": perf_counter() - started,
            "device_memory_stats": stats,
        }
        stages.append(payload)
        print(json.dumps(payload), flush=True)

    def execute(index, expected):
        source = "".join(cells[index]["source"])
        if expected not in source:
            raise ValueError(f"Unexpected notebook layout at cell {index}")
        exec(compile(source, f"{args.notebook}:cell{index}", "exec"), scope)

    execute(2, "from jaxpwa import")
    execute(4, "NORMALIZATION_CONFIG =")
    if args.normalization_resolution is not None:
        scope["NORMALIZATION_CONFIG"]["normalization_resolution"] = (
            args.normalization_resolution
        )
    if args.normalization_chunk_size != "auto":
        scope["NORMALIZATION_CONFIG"]["normalization_chunk_size"] = int(
            args.normalization_chunk_size
        )
    execute(6, "PLUS_CHANNEL =")
    with uproot.open(scope["TOY_FILE"]) as root_file:
        tree = root_file[scope["TOY_TREE"]]
        full_bytes = sum(
            tree[name].array(entry_stop=1, library="np").dtype.itemsize
            * tree.num_entries
            for name in ("m12Sq", "m13Sq", "m23Sq", "charge", "iExpt")
        )
        print(
            json.dumps(
                {
                    "root_entries": tree.num_entries,
                    "full_root_array_payload_bytes": full_bytes,
                    "old_notebook_device_payload_GiB": full_bytes / 1024**3,
                }
            ),
            flush=True,
        )

    def read_host_subset(file_path, tree, branches, **kwargs):
        kwargs.update(library="np", entry_stop=args.entry_stop)
        return read_root_tree(file_path, tree, branches, **kwargs)

    scope["read_root_tree"] = read_host_subset
    execute(8, "def load_toy_charge_sample")
    if scope["_RAW_IEXPT"][0] != 0 or scope["_RAW_IEXPT"][-1] <= 0:
        raise ValueError("Entry range must contain complete toy 0 and its next toy")
    report("toy_0_loaded_host_selection")
    for index, marker in (
        (10, "plus_efficiency ="),
        (12, "_qqbar_plus_pdf ="),
        (14, "COEFFICIENT_SPECS ="),
        (16, "plus_model ="),
        (18, "qqbar_plus_shape ="),
    ):
        execute(index, marker)
    report("models_maps_and_normalization_grids")
    source = "".join(cells[20]["source"])
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_toy_fit_session"
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "build_toy_fit_session",
            "exec",
        ),
        scope,
    )
    session, dropped_plus, dropped_minus = scope["build_toy_fit_session"](0)
    _block_tree((session.plus_cache, session.minus_cache))
    report("prepared_amplitude_caches")
    minimizer = session.minimizer(
        hessian="jax" if args.hessian else "numerical",
        verbose=1,
    )
    free, names, fcn, grad, hessian = minimizer._backend()
    point = np.asarray([parameter.value for parameter in free])
    print(
        json.dumps(
            {
                "events": [session.plus_data.size, session.minus_data.size],
                "normalization_points_per_charge": (
                    session.plus_model.normalization_sample.size
                ),
                "free_parameters": len(names),
                "dropped_zero_efficiency": [dropped_plus, dropped_minus],
            }
        ),
        flush=True,
    )
    value = fcn(*point)
    gradient = grad(*point)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("Non-finite likelihood or gradient")
    print(f"NLL: {value}; gradient norm: {np.linalg.norm(gradient)}", flush=True)
    report("value_gradient_compiled_and_evaluated")
    if args.hessian:
        report("jax_hessian_starting")
        matrix = hessian(*point)
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("Non-finite Hessian")
        print(f"Hessian norm: {np.linalg.norm(matrix)}", flush=True)
        report("jax_hessian_compiled_and_evaluated")
        if args.check_hessian:
            finite = np.empty_like(matrix)
            for i in range(point.size):
                delta = np.zeros_like(point)
                delta[i] = 1e-5
                finite[:, i] = (grad(*(point + delta)) - grad(*(point - delta))) / 2e-5
            relative = np.linalg.norm(matrix - finite) / np.linalg.norm(finite)
            print(
                f"Hessian vs gradient finite differences: relative error {relative}",
                flush=True,
            )
            if relative > 1e-5:
                raise RuntimeError("Hessian disagrees with gradient finite differences")
            report("jax_hessian_verified")
    if args.fit_ncall:
        report("fit_starting")
        result = minimizer.fit(strategy=1, hesse=True, ncall=args.fit_ncall)
        print(
            json.dumps(
                {
                    "fit_valid": bool(result.valid),
                    "nll": float(result.fval),
                    "edm": float(result.fmin.edm),
                    "accurate_covariance": bool(result.fmin.has_accurate_covar),
                    "hesse_failed": bool(result.fmin.hesse_failed),
                    "nfcn": result.nfcn,
                    "ngrad": result.ngrad,
                }
            ),
            flush=True,
        )
        report("fit_completed")
        del result
    del free, names, fcn, grad, hessian, minimizer, session
    gc.collect()
    report("session_released")
    print("PULL_STUDY_MEMORY_JSON=" + json.dumps(stages), flush=True)


if __name__ == "__main__":
    main()
