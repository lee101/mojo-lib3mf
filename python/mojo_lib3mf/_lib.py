"""ctypes bindings for the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.environ.get("MOJO_LIB3MF_LIBRARY") or os.path.join(
    ROOT, "dist", "libmojo-lib3mf.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "m3mf_transform_vertices_f32": ([I, I, I, I], None),
    "m3mf_matrix_multiply_f32": ([I, I, I], None),
    "m3mf_matrix_is_identity_f32": ([I], I),
    "m3mf_matrix_is_planar_f32": ([I], I),
    "m3mf_triangle_normals_f32": ([I, I, I, I], None),
    "m3mf_mesh_sanity": ([I] * 8, I),
    "m3mf_manifold_oriented": ([I] * 5, I),
    "m3mf_beam_lengths_f32": ([I, I, I, I, I, F], None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    if not os.path.exists(LIBRARY):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)
