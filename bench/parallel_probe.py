"""Size sweep + worker-count sweep for the parallel transform/beam-length kernels.

Builds bench/parallel_probe.mojo once, then for each size reports best-of-N wall time
for the serial kernel and the `max.algorithm` parallel kernel at a list of worker
counts. Every parallel result is also checked against the serial output.

Usage: pixi run python bench/parallel_probe.py [--workers 1,2,4,8] [--repeat 15]
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I = ctypes.c_int64
F = ctypes.c_double


def build() -> ctypes.CDLL:
    out = os.path.join(tempfile.gettempdir(), "libparallel_probe.so")
    env = dict(os.environ)
    if "MODULAR_HOME" not in env:
        mojo = subprocess.run(
            ["bash", "-lc", "which mojo"], capture_output=True, text=True, check=True
        ).stdout.strip()
        env["MODULAR_HOME"] = os.path.join(
            os.path.dirname(os.path.dirname(mojo)), "share", "max"
        )
    subprocess.run(
        [
            "mojo", "build", "--emit", "shared-lib",
            os.path.join(ROOT, "bench", "parallel_probe.mojo"), "-o", out,
        ],
        check=True,
        env=env,
    )
    return ctypes.CDLL(out)


def best(fn, repeat: int) -> float:
    result = math.inf
    for _ in range(repeat):
        time.sleep(0.002)  # let the worker pool settle before timing
        start = time.perf_counter()
        fn()
        result = min(result, time.perf_counter() - start)
    return result


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    workers = [int(w) for w in args.workers.split(",")]
    lib = build()
    for name in ("serial", "parallel"):
        pass
    lib.probe_transform_serial.argtypes = [I, I, I, I]
    lib.probe_transform_parallel.argtypes = [I, I, I, I, I]
    lib.probe_beam_lengths_serial.argtypes = [I, I, I, I, I, F]
    lib.probe_beam_lengths_parallel.argtypes = [I, I, I, I, I, F, I]

    rng = np.random.default_rng(7)
    matrix = np.array(
        [[1.2, 0.1, 0.2, 4], [0.3, 0.9, 0.1, 5], [0.1, 0.2, 1.1, 6], [0, 0, 0, 1]],
        np.float32,
    )
    print("## transform (m3mf_transform_vertices_f32 shape)")
    header = "| verts | MB | serial ms | " + " | ".join(
        f"{w}w ms | {w}w x" for w in workers
    ) + " |"
    print(header)
    print("| ---: | ---: | ---: | " + " | ".join("---: | ---:" for _ in workers) + " |")
    for count in (1_000, 10_000, 100_000, 250_000, 500_000, 750_000, 1_000_000,
                  1_500_000, 2_000_000, 3_000_000, 4_000_000, 8_000_000,
                  16_000_000):
        vertices = np.ascontiguousarray(rng.normal(size=(count, 3)), np.float32)
        result = np.empty_like(vertices)
        expect = np.empty_like(vertices)
        lib.probe_transform_serial(addr(vertices), addr(matrix), addr(expect), count)
        serial = best(
            lambda: lib.probe_transform_serial(
                addr(vertices), addr(matrix), addr(result), count
            ),
            args.repeat,
        )
        cells = []
        for w in workers:
            result.fill(np.nan)
            lib.probe_transform_parallel(
                addr(vertices), addr(matrix), addr(result), count, w
            )
            assert np.array_equal(result, expect), f"transform mismatch n={count} w={w}"
            t = best(
                lambda: lib.probe_transform_parallel(
                    addr(vertices), addr(matrix), addr(result), count, w
                ),
                args.repeat,
            )
            cells.append(f"{t * 1e3:.3f} | {serial / t:.2f}x")
        print(
            f"| {count:,} | {count * 12 / 2**20:.1f} | {serial * 1e3:.3f} | "
            + " | ".join(cells)
            + " |"
        )
        sys.stdout.flush()

    print()
    print("## beam lengths (m3mf_beam_lengths_f32 shape)")
    print(header.replace("verts", "beams"))
    print("| ---: | ---: | ---: | " + " | ".join("---: | ---:" for _ in workers) + " |")
    for count in (1_000, 10_000, 100_000, 250_000, 500_000, 750_000, 1_000_000,
                  1_500_000, 2_000_000, 3_000_000, 4_000_000):
        vertices = np.ascontiguousarray(rng.normal(size=(count + 1, 3)), np.float32)
        beams = np.ascontiguousarray(
            np.column_stack(
                (
                    np.arange(count, dtype=np.int32),
                    np.arange(1, count + 1, dtype=np.int32),
                )
            )
        )
        lengths = np.empty(count, np.float64)
        expect_len = np.empty(count, np.float64)
        expect_short = np.empty(count, np.int32)
        short = np.empty(count, np.int32)
        lib.probe_beam_lengths_serial(
            addr(vertices), addr(beams), addr(expect_len), addr(expect_short),
            count, 0.0001,
        )
        serial = best(
            lambda: lib.probe_beam_lengths_serial(
                addr(vertices), addr(beams), addr(lengths), addr(short), count, 0.0001
            ),
            args.repeat,
        )
        cells = []
        for w in workers:
            lib.probe_beam_lengths_parallel(
                addr(vertices), addr(beams), addr(lengths), addr(short), count,
                0.0001, w,
            )
            assert np.array_equal(lengths, expect_len), f"lengths mismatch n={count}"
            assert np.array_equal(short, expect_short), f"short mismatch n={count}"
            t = best(
                lambda: lib.probe_beam_lengths_parallel(
                    addr(vertices), addr(beams), addr(lengths), addr(short), count,
                    0.0001, w,
                ),
                args.repeat,
            )
            cells.append(f"{t * 1e3:.3f} | {serial / t:.2f}x")
        print(
            f"| {count:,} | {(count * 12 + count * 8) / 2**20:.1f} | "
            f"{serial * 1e3:.3f} | " + " | ".join(cells) + " |"
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
