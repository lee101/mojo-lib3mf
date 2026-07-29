"""Benchmarks against lib3mf itself and source-equivalent NumPy kernels."""

from __future__ import annotations

import io
import math
import os
import platform
import sys
import time

import lib3mf
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_lib3mf as m3  # noqa: E402


def best(function, repeat: int = 4) -> float:
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        result = min(result, time.perf_counter() - start)
    return result


def numpy_transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x, y, z = vertices.T
    result = np.empty_like(vertices)
    result[:, 0] = (
        matrix[0, 0] * x
        + matrix[0, 1] * y
        + matrix[0, 2] * z
        + matrix[0, 3]
    )
    result[:, 1] = (
        matrix[1, 0] * x
        + matrix[1, 1] * y
        + matrix[1, 2] * z
        + matrix[1, 3]
    )
    result[:, 2] = (
        matrix[2, 0] * x
        + matrix[2, 1] * y
        + matrix[2, 2] * z
        + matrix[2, 3]
    )
    return result


def numpy_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    u = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    v = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    result = np.cross(u, v)
    lengths = np.sqrt(np.sum(result * result, axis=1))
    valid = lengths > np.float32(1e-10)
    result[valid] /= lengths[valid, None]
    result[~valid] = 0
    return result


def numpy_beam_lengths(
    vertices: np.ndarray, beams: np.ndarray, minimum: float
) -> tuple[np.ndarray, np.ndarray]:
    difference = vertices[beams[:, 0]] - vertices[beams[:, 1]]
    lengths_f32 = np.sqrt(np.sum(difference * difference, axis=1))
    return lengths_f32.astype(np.float64), lengths_f32 < minimum


def repeated_tetrahedra(count: int) -> tuple[np.ndarray, np.ndarray]:
    base_vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32
    )
    base_faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32
    )
    vertices = np.tile(base_vertices, (count, 1))
    vertices[:, 0] += np.repeat(np.arange(count, dtype=np.float32) * 2, 4)
    offsets = np.repeat(np.arange(count, dtype=np.int32) * 4, 4)[:, None]
    triangles = np.tile(base_faces, (count, 1)) + offsets
    return vertices, triangles


def reference_mesh(vertices: np.ndarray, triangles: np.ndarray):
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    mesh = model.AddMeshObject()
    positions = []
    for coordinates in vertices:
        position = lib3mf.Position()
        position.Coordinates[:] = coordinates
        positions.append(position)
    faces = []
    for indices in triangles:
        triangle = lib3mf.Triangle()
        triangle.Indices[:] = indices
        faces.append(triangle)
    mesh.SetGeometry(positions, faces)
    return wrapper, model, mesh


def run() -> list[tuple[str, float, float, str]]:
    rng = np.random.default_rng(7)
    rows: list[tuple[str, float, float, str]] = []

    vertices = np.ascontiguousarray(rng.normal(size=(1_000_000, 3)), np.float32)
    matrix = np.array(
        [[1.2, 0.1, 0.2, 4], [0.3, 0.9, 0.1, 5], [0.1, 0.2, 1.1, 6], [0, 0, 0, 1]],
        np.float32,
    )
    mesh = m3.Mesh(vertices, [])
    mesh.transformed(matrix)
    rows.append(
        (
            "transform 1M vertices",
            best(lambda: mesh.transformed(matrix)),
            best(lambda: numpy_transform(vertices, matrix)),
            "NumPy port of fnMATRIX3_apply",
        )
    )

    normal_vertices = np.ascontiguousarray(
        rng.normal(size=(750_000, 3)), np.float32
    )
    triangles = np.arange(750_000, dtype=np.int32).reshape(-1, 3)
    normal_mesh = m3.Mesh(normal_vertices, triangles)
    normal_mesh.triangle_normals()
    rows.append(
        (
            "normals 250k triangles",
            best(normal_mesh.triangle_normals),
            best(lambda: numpy_normals(normal_vertices, triangles)),
            "NumPy port of fnVEC3_calcTriangleNormal",
        )
    )

    beam_vertices = np.ascontiguousarray(
        rng.normal(size=(1_000_001, 3)), np.float32
    )
    beams = np.column_stack(
        (
            np.arange(1_000_000, dtype=np.int32),
            np.arange(1, 1_000_001, dtype=np.int32),
        )
    )
    beam_mesh = m3.Mesh(
        beam_vertices, [], m3.BeamLattice(beams, minimum_length=0.0001)
    )
    beam_mesh.beam_lengths()
    rows.append(
        (
            "lengths 1M beams",
            best(beam_mesh.beam_lengths),
            best(lambda: numpy_beam_lengths(beam_vertices, beams, 0.0001)),
            "NumPy port of BeamLattice1702_Beams",
        )
    )

    manifold_vertices, manifold_faces = repeated_tetrahedra(5_000)
    manifold_mesh = m3.Mesh(manifold_vertices, manifold_faces)
    _, _, upstream_mesh = reference_mesh(manifold_vertices, manifold_faces)
    manifold_mesh.is_manifold_and_oriented()
    upstream_mesh.IsManifoldAndOriented()
    rows.append(
        (
            "manifold 20k triangles",
            best(manifold_mesh.is_manifold_and_oriented, 3),
            best(upstream_mesh.IsManifoldAndOriented, 3),
            "lib3mf 2.5.0",
        )
    )
    return rows


def machine() -> str:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu}; {platform.system()} {platform.release()}"


def main() -> None:
    print(f"Machine: {machine()}")
    print()
    print("| kernel | mojo-lib3mf | reference | speedup | reference implementation |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, mojo_time, reference_time, reference_name in run():
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{reference_time * 1e3:.3f} ms | "
            f"{reference_time / mojo_time:.2f}x | {reference_name} |"
        )


if __name__ == "__main__":
    main()
