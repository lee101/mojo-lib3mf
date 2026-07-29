from __future__ import annotations

import lib3mf
import numpy as np
import pytest

import mojo_lib3mf as m3


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    triangles = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32
    )
    return vertices, triangles


def reference_mesh(vertices: np.ndarray, triangles: np.ndarray):
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    mesh = model.AddMeshObject()
    positions = []
    for coordinates in vertices:
        position = lib3mf.Position()
        position.Coordinates[:] = [float(value) for value in coordinates]
        positions.append(position)
    faces = []
    for indices in triangles:
        face = lib3mf.Triangle()
        face.Indices[:] = [int(value) for value in indices]
        faces.append(face)
    mesh.SetGeometry(positions, faces)
    return mesh


def test_transform_vertices_matches_upstream_matrix_apply_reference():
    vertices, triangles = tetrahedron()
    matrix = np.array(
        [[2, 0.5, 0, 4], [0, -1, 0.25, 3], [0.5, 0, 3, -2], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    actual = m3.Mesh(vertices, triangles).transformed(matrix).vertices
    expected = np.empty_like(vertices)
    for i, (x, y, z) in enumerate(vertices):
        expected[i] = [
            matrix[0, 0] * x
            + matrix[0, 1] * y
            + matrix[0, 2] * z
            + matrix[0, 3],
            matrix[1, 0] * x
            + matrix[1, 1] * y
            + matrix[1, 2] * z
            + matrix[1, 3],
            matrix[2, 0] * x
            + matrix[2, 1] * y
            + matrix[2, 2] * z
            + matrix[2, 3],
        ]
    np.testing.assert_array_equal(actual, expected)


def test_matrix_multiply_and_3mf_wire_order():
    first = np.array(
        [[1, 2, 0, 3], [0, 1, 4, 5], [2, 0, 1, 7], [0, 0, 0, 1]],
        np.float32,
    )
    second = np.array(
        [[2, 0, 1, 2], [1, 3, 0, 4], [0, 2, 1, 6], [0, 0, 0, 1]],
        np.float32,
    )
    np.testing.assert_allclose(
        m3.multiply_transforms(first, second), first @ second, rtol=1e-6
    )
    text = m3.transform_to_3mf(first)
    assert text.split() == [
        "1",
        "0",
        "2",
        "2",
        "1",
        "0",
        "0",
        "4",
        "1",
        "3",
        "5",
        "7",
    ]
    np.testing.assert_array_equal(m3.transform_from_3mf(text), first)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.0, True), (0.0001, True), (0.001, False)],
)
def test_identity_uses_upstream_squared_delta_threshold(delta, expected):
    matrix = m3.identity_transform()
    matrix[0, 0] += np.float32(delta)
    assert m3.is_identity_transform(matrix) is expected


@pytest.mark.parametrize(
    ("z_skew", "expected"), [(0.0, True), (5e-8, True), (2e-7, False)]
)
def test_planar_uses_upstream_epsilon(z_skew, expected):
    matrix = m3.identity_transform()
    matrix[2, 0] = np.float32(z_skew)
    assert m3.is_planar_transform(matrix) is expected


def test_normals_match_upstream_cross_normalize_and_degenerate_zero():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [2, 0, 0]], np.float32
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3]], np.int32)
    normals = m3.Mesh(vertices, triangles).triangle_normals()
    np.testing.assert_array_equal(normals, [[0, 0, 1], [0, 0, 0]])


@pytest.mark.parametrize(
    ("vertices", "triangles", "expected"),
    [
        (np.empty((0, 3)), np.empty((0, 3)), True),
        ([[0, 0, 0]], np.empty((0, 3)), True),
        ([[0, 0, 0], [1, 0, 0]], [[0, 0, 1]], False),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]], False),
        ([[1.1e9, 0, 0]], np.empty((0, 3)), False),
    ],
)
def test_mesh_sanity_edge_cases(vertices, triangles, expected):
    mesh = m3.Mesh(vertices, triangles)
    assert mesh.check_sanity() is expected


def test_duplicate_positions_zero_area_and_unreferenced_vertices_are_sane():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [99, 99, 99]],
        np.float32,
    )
    mesh = m3.Mesh(vertices, [[0, 1, 2], [0, 1, 3]])
    assert mesh.check_sanity()
    np.testing.assert_array_equal(mesh.triangle_normals(), np.zeros((2, 3)))


def test_manifold_parity_with_same_upstream_library():
    vertices, triangles = tetrahedron()
    ours = m3.Mesh(vertices, triangles)
    reference = reference_mesh(vertices, triangles)
    assert ours.is_manifold_and_oriented()
    assert reference.IsManifoldAndOriented()

    open_triangles = triangles[:-1]
    ours_open = m3.Mesh(vertices, open_triangles)
    reference_open = reference_mesh(vertices, open_triangles)
    assert ours_open.is_manifold_and_oriented() is False
    assert reference_open.IsManifoldAndOriented() is False


def test_non_manifold_edge_is_rejected():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        np.float32,
    )
    triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], np.int32)
    assert not m3.Mesh(vertices, triangles).is_manifold_and_oriented()


def test_manifold_radix_sort_handles_multiple_key_bytes():
    vertices, triangles = tetrahedron()
    copies = 100
    repeated_vertices = np.tile(vertices, (copies, 1))
    repeated_vertices[:, 0] += np.repeat(
        np.arange(copies, dtype=np.float32) * 2, len(vertices)
    )
    offsets = np.repeat(
        np.arange(copies, dtype=np.int32) * len(vertices), len(triangles)
    )[:, None]
    repeated_triangles = np.tile(triangles, (copies, 1)) + offsets
    assert m3.Mesh(repeated_vertices, repeated_triangles).is_manifold_and_oriented()


def test_manifold_radix_sort_rejects_same_orientation_pair():
    vertices, triangles = tetrahedron()
    reversed_face = triangles.copy()
    reversed_face[0] = reversed_face[0, ::-1]
    assert not m3.Mesh(vertices, reversed_face).is_manifold_and_oriented()


def test_beam_lengths_and_strict_minimum_warning():
    vertices = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 0.00005]], np.float32)
    lattice = m3.BeamLattice([[0, 1], [0, 2]], minimum_length=0.0001)
    lengths, too_short = m3.Mesh(vertices, [], lattice).beam_lengths()
    np.testing.assert_allclose(lengths, [5.0, 0.00005], rtol=1e-6)
    np.testing.assert_array_equal(too_short, [False, True])


def test_beam_sanity_rejects_self_edge_and_bad_ball():
    vertices = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    self_edge = m3.BeamLattice([[0, 0]])
    assert not m3.Mesh(vertices, [], self_edge).check_sanity()
    bad_ball = m3.BeamLattice(
        [[0, 1]], ball_indices=np.array([2]), ball_radii=np.array([0.1])
    )
    assert not m3.Mesh(vertices, [], bad_ball).check_sanity()


def test_index_arrays_reject_nonintegers_and_int32_overflow_before_ffi():
    with pytest.raises(TypeError, match="integers"):
        m3.Mesh([[0, 0, 0]], np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(OverflowError, match="Int32|int32"):
        m3.Mesh([[0, 0, 0]], np.array([[0, 1, 2**32]], dtype=np.int64))
    with pytest.raises(OverflowError, match="Int32|int32"):
        m3.BeamLattice(np.array([[0, 2**32]], dtype=np.uint64))
    with pytest.raises(OverflowError, match="Int32|int32"):
        m3.BeamLattice([], ball_indices=np.array([2**32], dtype=np.uint64))


def test_float32_ffi_inputs_do_not_silently_narrow_numpy_arrays():
    with pytest.raises(TypeError, match="float32"):
        m3.Mesh(np.zeros((1, 3), dtype=np.float64), [])
    with pytest.raises(TypeError, match="float32"):
        m3.is_identity_transform(np.eye(4, dtype=np.float64))


def test_transform_rejects_nonfinite_overflow_and_trailing_text():
    matrix = m3.identity_transform()
    matrix[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        m3.is_identity_transform(matrix)
    with pytest.raises(ValueError, match="exactly 12"):
        m3.transform_from_3mf("1 0 0 0 1 0 0 0 1 0 0 0 trailing")
