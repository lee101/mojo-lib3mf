"""Flat Python model types backed by the ported Mojo geometry kernels."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ._lib import addr, lib

CAP_MODES = ("sphere", "hemisphere", "butt")
BALL_MODES = ("none", "mixed", "all")
OBJECT_TYPES = ("model", "solidsupport", "support", "surface", "other")
UNITS = ("micron", "millimeter", "centimeter", "inch", "foot", "meter")


def _array(value, dtype, columns: int, name: str) -> np.ndarray:
    source = np.asarray(value)
    target = np.dtype(dtype)
    if source.size == 0:
        return np.empty((0, columns), dtype=target)
    if target.kind == "i":
        if source.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integers")
        if source.size:
            limits = np.iinfo(target)
            minimum, maximum = int(source.min()), int(source.max())
            if minimum < limits.min or maximum > limits.max:
                raise OverflowError(f"{name} values do not fit {target.name}")
    elif source.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers")
    elif target == np.dtype(np.float32) and isinstance(value, np.ndarray):
        if source.dtype != target:
            raise TypeError(f"{name} NumPy arrays must have dtype float32")
    array = np.ascontiguousarray(source, dtype=target)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (n, {columns})")
    return array


def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def transform_from_3mf(text: str) -> np.ndarray:
    fields = text.split()
    if len(fields) != 12:
        raise ValueError("a 3MF transform must contain exactly 12 numbers")
    try:
        values64 = np.asarray([float(field) for field in fields], dtype=np.float64)
    except ValueError as error:
        raise ValueError("a 3MF transform contains an invalid number") from error
    if not np.isfinite(values64).all() or np.any(
        np.abs(values64) > np.finfo(np.float32).max
    ):
        raise ValueError("a 3MF transform contains a non-finite or out-of-range number")
    values = values64.astype(np.float32)
    matrix = identity_transform()
    for column in range(4):
        for row in range(3):
            matrix[row, column] = values[row + column * 3]
    return matrix


def transform_to_3mf(transform) -> str:
    matrix = _array4(transform)
    return " ".join(
        format(float(matrix[row, column]), ".9g")
        for column in range(4)
        for row in range(3)
    )


def _array4(value) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if source.dtype.kind not in "fiu":
        raise TypeError("transform must contain real numbers")
    if isinstance(value, np.ndarray) and source.dtype != np.dtype(np.float32):
        raise TypeError("transform NumPy arrays must have dtype float32")
    if not np.isfinite(source).all() or np.any(
        np.abs(source) > np.finfo(np.float32).max
    ):
        raise ValueError("transform values must be finite and fit Float32")
    matrix = np.ascontiguousarray(source, dtype=np.float32)
    if not np.array_equal(matrix[3], np.array([0, 0, 0, 1], np.float32)):
        raise ValueError("3MF transforms must be affine with last row [0, 0, 0, 1]")
    return matrix


def multiply_transforms(first, second) -> np.ndarray:
    a = _array4(first)
    b = _array4(second)
    result = np.empty((4, 4), dtype=np.float32)
    lib().m3mf_matrix_multiply_f32(addr(a), addr(b), addr(result))
    return result


def is_identity_transform(transform) -> bool:
    matrix = _array4(transform)
    return bool(lib().m3mf_matrix_is_identity_f32(addr(matrix)))


def is_planar_transform(transform) -> bool:
    matrix = _array4(transform)
    return bool(lib().m3mf_matrix_is_planar_f32(addr(matrix)))


@dataclass
class BeamLattice:
    beams: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    radii: np.ndarray | None = None
    caps: tuple[tuple[str, str], ...] | Iterable[Iterable[str]] | None = None
    minimum_length: float = 0.0001
    default_radius: float = 1.0
    ball_mode: str = "none"
    default_ball_radius: float = 0.0
    ball_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    ball_radii: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.beams = _array(self.beams, np.int32, 2, "beams")
        count = len(self.beams)
        if self.radii is None:
            self.radii = np.full((count, 2), self.default_radius, np.float64)
        self.radii = _array(self.radii, np.float64, 2, "radii")
        if len(self.radii) != count:
            raise ValueError("radii must have one row per beam")
        if self.caps is None:
            self.caps = tuple(("sphere", "sphere") for _ in range(count))
        else:
            self.caps = tuple(tuple(pair) for pair in self.caps)
        if len(self.caps) != count or any(len(pair) != 2 for pair in self.caps):
            raise ValueError("caps must contain two modes per beam")
        if any(cap not in CAP_MODES for pair in self.caps for cap in pair):
            raise ValueError(f"cap modes must be one of {CAP_MODES}")
        if self.ball_mode not in BALL_MODES:
            raise ValueError(f"ball_mode must be one of {BALL_MODES}")
        ball_indices = np.asarray(self.ball_indices)
        if ball_indices.ndim != 1:
            raise ValueError("ball_indices must be one-dimensional")
        self.ball_indices = _array(
            ball_indices.reshape(-1, 1),
            np.int32,
            1,
            "ball_indices",
        ).reshape(-1)
        if self.ball_radii is None:
            self.ball_radii = np.full(
                len(self.ball_indices), self.default_ball_radius, np.float64
            )
        self.ball_radii = np.ascontiguousarray(self.ball_radii, dtype=np.float64)
        if self.ball_radii.shape != self.ball_indices.shape:
            raise ValueError("ball_radii must have one value per ball")


@dataclass
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray
    beam_lattice: BeamLattice | None = None

    def __post_init__(self) -> None:
        self.vertices = _array(self.vertices, np.float32, 3, "vertices")
        self.triangles = _array(self.triangles, np.int32, 3, "triangles")
        if self.beam_lattice is not None and not isinstance(
            self.beam_lattice, BeamLattice
        ):
            raise TypeError("beam_lattice must be a BeamLattice")

    def transformed(self, transform) -> "Mesh":
        matrix = _array4(transform)
        result = np.empty_like(self.vertices)
        if len(self.vertices):
            lib().m3mf_transform_vertices_f32(
                addr(self.vertices), addr(matrix), addr(result), len(self.vertices)
            )
        return Mesh(result, self.triangles.copy(), copy.deepcopy(self.beam_lattice))

    def triangle_normals(self) -> np.ndarray:
        result = np.empty((len(self.triangles), 3), dtype=np.float32)
        if len(result):
            if not self._indices_in_range(self.triangles):
                raise ValueError("triangle index out of range")
            lib().m3mf_triangle_normals_f32(
                addr(self.vertices),
                addr(self.triangles),
                addr(result),
                len(self.triangles),
            )
        return result

    def _indices_in_range(self, indices: np.ndarray) -> bool:
        return bool(
            indices.size == 0
            or (int(indices.min()) >= 0 and int(indices.max()) < len(self.vertices))
        )

    def check_sanity(self) -> bool:
        lattice = self.beam_lattice
        beams = (
            lattice.beams
            if lattice is not None
            else np.empty((0, 2), dtype=np.int32)
        )
        balls = (
            lattice.ball_indices
            if lattice is not None
            else np.empty(0, dtype=np.int32)
        )
        return bool(
            lib().m3mf_mesh_sanity(
                addr(self.vertices),
                addr(self.triangles),
                addr(beams),
                addr(balls),
                len(self.vertices),
                len(self.triangles),
                len(beams),
                len(balls),
            )
        )

    def is_manifold_and_oriented(self) -> bool:
        if not self.check_sanity():
            return False
        edge_count = 3 * len(self.triangles)
        workspace = np.empty(2 * edge_count, dtype=np.int64)
        return bool(
            lib().m3mf_manifold_oriented(
                addr(self.triangles),
                addr(workspace),
                addr(workspace) + edge_count * workspace.itemsize,
                len(self.vertices),
                len(self.triangles),
            )
        )

    def beam_lengths(self) -> tuple[np.ndarray, np.ndarray]:
        if self.beam_lattice is None:
            empty = np.empty(0, dtype=np.float64)
            return empty, np.empty(0, dtype=bool)
        beams = self.beam_lattice.beams
        if not self._indices_in_range(beams):
            raise ValueError("beam index out of range")
        lengths = np.empty(len(beams), dtype=np.float64)
        flags = np.empty(len(beams), dtype=np.int32)
        if len(beams):
            lib().m3mf_beam_lengths_f32(
                addr(self.vertices),
                addr(beams),
                addr(lengths),
                addr(flags),
                len(beams),
                self.beam_lattice.minimum_length,
            )
        return lengths, flags.astype(bool)


@dataclass
class Component:
    object_id: int
    transform: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.object_id <= 0:
            raise ValueError("object_id must be positive")
        if self.transform is not None:
            self.transform = _array4(self.transform)


@dataclass
class MeshObject:
    object_id: int
    mesh: Mesh
    name: str = ""
    object_type: str = "model"

    def __post_init__(self) -> None:
        if self.object_id <= 0:
            raise ValueError("object_id must be positive")
        if self.object_type not in OBJECT_TYPES:
            raise ValueError(f"object_type must be one of {OBJECT_TYPES}")


@dataclass
class ComponentsObject:
    object_id: int
    components: list[Component]
    name: str = ""
    object_type: str = "model"

    def __post_init__(self) -> None:
        if self.object_id <= 0:
            raise ValueError("object_id must be positive")
        if self.object_type not in OBJECT_TYPES:
            raise ValueError(f"object_type must be one of {OBJECT_TYPES}")


@dataclass
class BuildItem:
    object_id: int
    transform: np.ndarray | None = None
    part_number: str = ""

    def __post_init__(self) -> None:
        if self.object_id <= 0:
            raise ValueError("object_id must be positive")
        if self.transform is not None:
            self.transform = _array4(self.transform)


@dataclass
class Model:
    objects: list[MeshObject | ComponentsObject] = field(default_factory=list)
    build_items: list[BuildItem] = field(default_factory=list)
    unit: str = "millimeter"
    language: str = "en-US"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object IDs must be unique")
        known = set(ids)
        if any(item.object_id not in known for item in self.build_items):
            raise ValueError("build item refers to an unknown object")
        for obj in self.objects:
            if isinstance(obj, ComponentsObject):
                if any(component.object_id not in known for component in obj.components):
                    raise ValueError("component refers to an unknown object")

    def write(self, destination, *, compression: int = 8) -> None:
        from .io import write

        write(self, destination, compression=compression)

    @classmethod
    def read(cls, source) -> "Model":
        from .io import read

        return read(source)
