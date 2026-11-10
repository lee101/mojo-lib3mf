"""Serial vs parallel equivalence for the gated kernels.

The transform kernel is gated at 4,000,000 vertices and beam lengths at 3,000,000
beams (src/lib3mf.mojo). Each case below runs the same input through the serial and
the gated parallel path and requires bit-identical output, including on sizes that
straddle the thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_lib3mf as m3
from mojo_lib3mf._lib import addr, lib

TRANSFORM_PARALLEL_MIN = 4_000_000
BEAM_LENGTHS_PARALLEL_MIN = 3_000_000


def _matrix() -> np.ndarray:
    return np.array(
        [[1.2, 0.1, 0.2, 4], [0.3, 0.9, 0.1, 5], [0.1, 0.2, 1.1, 6], [0, 0, 0, 1]],
        np.float32,
    )


@pytest.mark.parametrize(
    "count",
    [TRANSFORM_PARALLEL_MIN - 1, TRANSFORM_PARALLEL_MIN, TRANSFORM_PARALLEL_MIN + 1],
)
def test_transform_parallel_matches_serial(count):
    matrix = _matrix()
    vertices = np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 17
    vertices[:, 1] *= -3
    parallel = np.empty_like(vertices)
    serial = np.empty_like(vertices)
    kernels = lib()
    kernels.m3mf_transform_vertices_f32(
        addr(vertices), addr(matrix), addr(parallel), count
    )
    kernels.m3mf_transform_vertices_f32_serial(
        addr(vertices), addr(matrix), addr(serial), count
    )
    np.testing.assert_array_equal(parallel, serial)
    # The public path uses the gated kernel; it must agree with the serial kernel.
    np.testing.assert_array_equal(
        m3.Mesh(vertices, []).transformed(matrix).vertices, serial
    )


@pytest.mark.parametrize(
    "count",
    [BEAM_LENGTHS_PARALLEL_MIN - 1, BEAM_LENGTHS_PARALLEL_MIN, BEAM_LENGTHS_PARALLEL_MIN + 1],
)
def test_beam_lengths_parallel_matches_serial(count):
    rng = np.random.default_rng(count)
    vertices = rng.normal(size=(count + 1, 3)).astype(np.float32)
    beams = np.column_stack(
        (np.arange(count, dtype=np.int32), np.arange(1, count + 1, dtype=np.int32))
    )
    minimum = 1.5
    kernels = lib()
    lengths = np.empty(count, np.float64)
    too_short = np.empty(count, np.int32)
    kernels.m3mf_beam_lengths_f32(
        addr(vertices), addr(beams), addr(lengths), addr(too_short), count, minimum
    )
    reference_vertices = vertices.reshape(count + 1, 3).astype(np.float64)
    difference = reference_vertices[beams[:, 0]] - reference_vertices[beams[:, 1]]
    expected_lengths = np.sqrt(np.sum(difference * difference, axis=1))
    # Same tolerance as benchmark.numpy_beam_lengths / test_kernels.py.
    np.testing.assert_allclose(lengths, expected_lengths, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(too_short, (expected_lengths < minimum).astype(np.int32))
    # The public path uses the gated kernel and returns identical values.
    public_lengths, public_short = m3.Mesh(
        vertices, [], m3.BeamLattice(beams, minimum_length=minimum)
    ).beam_lengths()
    np.testing.assert_allclose(public_lengths, expected_lengths, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(public_short, expected_lengths < minimum)
