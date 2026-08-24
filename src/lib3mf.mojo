"""Compute kernels ported from lib3mf's mesh and fixed-size math code."""

from max.algorithm import parallelize
from std.math import sqrt
from std.memory import stack_allocation

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime TRANSFORM_PARALLEL_THRESHOLD = 4_000_000
comptime TRANSFORM_TASKS = 8


def f32p(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def f64p(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


@always_inline
def transform_vertices_range(
    vertices: F32Ptr,
    matrix: F32Ptr,
    result: F32Ptr,
    start: Int,
    end: Int,
):
    for i in range(start, end):
        var x = vertices[3 * i]
        var y = vertices[3 * i + 1]
        var z = vertices[3 * i + 2]
        result[3 * i] = (
            matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
        )
        result[3 * i + 1] = (
            matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
        )
        result[3 * i + 2] = (
            matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]
        )


# lib3mf: Source/Common/Math/NMR_Matrix.cpp fnMATRIX3_apply
@export("m3mf_transform_vertices_f32")
def transform_vertices_f32(
    vertices_addr: Int, matrix_addr: Int, result_addr: Int, count: Int
) abi("C"):
    var vertices = f32p(vertices_addr)
    var matrix = f32p(matrix_addr)
    var result = f32p(result_addr)
    if count < TRANSFORM_PARALLEL_THRESHOLD:
        transform_vertices_range(vertices, matrix, result, 0, count)
        return

    @parameter
    def work(task: Int):
        var start = count * task // TRANSFORM_TASKS
        transform_vertices_range(
            vertices,
            matrix,
            result,
            start,
            count * (task + 1) // TRANSFORM_TASKS,
        )

    parallelize[work](TRANSFORM_TASKS, TRANSFORM_TASKS)


@export("m3mf_transform_vertices_f32_serial")
def transform_vertices_f32_serial(
    vertices_addr: Int, matrix_addr: Int, result_addr: Int, count: Int
) abi("C"):
    transform_vertices_range(
        f32p(vertices_addr), f32p(matrix_addr), f32p(result_addr), 0, count
    )


# lib3mf: Source/Common/Math/NMR_Matrix.cpp fnMATRIX3_multiply
@export("m3mf_matrix_multiply_f32")
def matrix_multiply_f32(
    first_addr: Int, second_addr: Int, result_addr: Int
) abi("C"):
    var first = f32p(first_addr)
    var second = f32p(second_addr)
    var result = f32p(result_addr)
    for i in range(4):
        for j in range(4):
            result[4 * i + j] = (
                first[4 * i] * second[j]
                + first[4 * i + 1] * second[4 + j]
                + first[4 * i + 2] * second[8 + j]
                + first[4 * i + 3] * second[12 + j]
            )


# lib3mf: Source/Common/Math/NMR_Matrix.cpp fnMATRIX3_isIdentity
@export("m3mf_matrix_is_identity_f32")
def matrix_is_identity_f32(matrix_addr: Int) abi("C") -> Int:
    var matrix = f32p(matrix_addr)
    var delta = Float64(0.0)
    for i in range(4):
        for j in range(4):
            var expected = Float32(1.0) if i == j else Float32(0.0)
            var d = Float64(matrix[4 * i + j] - expected)
            delta += d * d
    return 1 if delta < 0.0000001 else 0


# lib3mf: Source/Common/Math/NMR_Matrix.cpp fnMATRIX3_isplanar
@export("m3mf_matrix_is_planar_f32")
def matrix_is_planar_f32(matrix_addr: Int) abi("C") -> Int:
    var matrix = f32p(matrix_addr)
    var eps = Float32(1.0e-7)
    return 1 if (
        abs(matrix[8]) < eps
        and abs(matrix[9]) < eps
        and abs(matrix[10] - 1.0) < eps
        and abs(matrix[2]) < eps
        and abs(matrix[6]) < eps
    ) else 0


# lib3mf: Source/Common/Math/NMR_Vector.cpp fnVEC3_calcTriangleNormal
@export("m3mf_triangle_normals_f32")
def triangle_normals_f32(
    vertices_addr: Int, triangles_addr: Int, result_addr: Int, face_count: Int
) abi("C"):
    var vertices = f32p(vertices_addr)
    var triangles = i32p(triangles_addr)
    var result = f32p(result_addr)
    for i in range(face_count):
        var a = Int(triangles[3 * i])
        var b = Int(triangles[3 * i + 1])
        var c = Int(triangles[3 * i + 2])
        var ux = vertices[3 * b] - vertices[3 * a]
        var uy = vertices[3 * b + 1] - vertices[3 * a + 1]
        var uz = vertices[3 * b + 2] - vertices[3 * a + 2]
        var vx = vertices[3 * c] - vertices[3 * a]
        var vy = vertices[3 * c + 1] - vertices[3 * a + 1]
        var vz = vertices[3 * c + 2] - vertices[3 * a + 2]
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        var length = sqrt(nx * nx + ny * ny + nz * nz)
        if length > 1.0e-10:
            result[3 * i] = nx / length
            result[3 * i + 1] = ny / length
            result[3 * i + 2] = nz / length
        else:
            result[3 * i] = 0.0
            result[3 * i + 1] = 0.0
            result[3 * i + 2] = 0.0


# lib3mf: Source/Common/Mesh/NMR_Mesh.cpp CMesh::checkSanity
@export("m3mf_mesh_sanity")
def mesh_sanity(
    vertices_addr: Int,
    triangles_addr: Int,
    beams_addr: Int,
    balls_addr: Int,
    vertex_count: Int,
    face_count: Int,
    beam_count: Int,
    ball_count: Int,
) abi("C") -> Int:
    if (
        vertex_count > 2147483647
        or face_count > 2147483647
        or beam_count > 2147483647
        or ball_count > 2147483647
    ):
        return 0
    if vertex_count > 0:
        var vertices = f32p(vertices_addr)
        for i in range(vertex_count * 3):
            if abs(vertices[i]) > 1000000000.0:
                return 0
    if face_count > 0:
        var triangles = i32p(triangles_addr)
        for i in range(face_count):
            var a = Int(triangles[3 * i])
            var b = Int(triangles[3 * i + 1])
            var c = Int(triangles[3 * i + 2])
            if a < 0 or b < 0 or c < 0:
                return 0
            if a >= vertex_count or b >= vertex_count or c >= vertex_count:
                return 0
            if a == b or a == c or b == c:
                return 0
    if beam_count > 0:
        var beams = i32p(beams_addr)
        for i in range(beam_count):
            var a = Int(beams[2 * i])
            var b = Int(beams[2 * i + 1])
            if (
                a < 0
                or b < 0
                or a >= vertex_count
                or b >= vertex_count
                or a == b
            ):
                return 0
    if ball_count > 0:
        var balls = i32p(balls_addr)
        for i in range(ball_count):
            var a = Int(balls[i])
            if a < 0 or a >= vertex_count:
                return 0
    return 1


# lib3mf: Source/Model/Classes/NMR_ModelMeshObject.cpp CModelMeshObject::isManifoldAndOriented
def radix_sort_edges(keys: I64Ptr, scratch: I64Ptr, count: Int):
    var counts = stack_allocation[256, DType.int64]()
    for pass_index in range(8):
        for bucket in range(256):
            counts[bucket] = 0
        var source = keys if pass_index % 2 == 0 else scratch
        var target = scratch if pass_index % 2 == 0 else keys
        var shift = Int64(8 * pass_index)
        var mask = Int64(127 if pass_index == 7 else 255)
        for i in range(count):
            var bucket = Int((source[i] >> shift) & mask)
            counts[bucket] += 1
        var offset = Int64(0)
        for bucket in range(256):
            var bucket_count = counts[bucket]
            counts[bucket] = offset
            offset += bucket_count
        for i in range(count):
            var value = source[i]
            var bucket = Int((value >> shift) & mask)
            target[Int(counts[bucket])] = value
            counts[bucket] += 1


# lib3mf: Source/Model/Classes/NMR_ModelMeshObject.cpp CModelMeshObject::isManifoldAndOriented
@export("m3mf_manifold_oriented")
def manifold_oriented(
    triangles_addr: Int,
    keys_addr: Int,
    directions_addr: Int,
    vertex_count: Int,
    face_count: Int,
) abi("C") -> Int:
    if vertex_count < 3 or face_count < 3:
        return 0
    var triangles = i32p(triangles_addr)
    var keys = i64p(keys_addr)
    var scratch = i64p(directions_addr)
    var direction_mask = Int64(-9223372036854775807) - 1
    for face in range(face_count):
        for edge in range(3):
            var a = Int64(triangles[3 * face + edge])
            var b = Int64(triangles[3 * face + (edge + 1) % 3])
            if (
                a < 0
                or b < 0
                or a >= Int64(vertex_count)
                or b >= Int64(vertex_count)
                or a == b
            ):
                return 0
            var lo = a if a <= b else b
            var hi = b if a <= b else a
            var key = (lo << 32) | hi
            keys[3 * face + edge] = (
                key if a <= b else key | direction_mask
            )
    var edge_count = 3 * face_count
    radix_sort_edges(keys, scratch, edge_count)
    var key_mask = Int64(9223372036854775807)
    var i = 0
    while i < edge_count:
        if i + 1 >= edge_count:
            return 0
        var first = keys[i]
        var second = keys[i + 1]
        var key = first & key_mask
        if (
            key != (second & key_mask)
            or (first < 0) == (second < 0)
            or (
                i + 2 < edge_count
                and key == (keys[i + 2] & key_mask)
            )
        ):
            return 0
        i += 2
    return 1


# lib3mf: Source/Model/Reader/BeamLattice1702/NMR_ModelReaderNode_BeamLattice1702_Beams.cpp OnNSChildElement
@export("m3mf_beam_lengths_f32")
def beam_lengths_f32(
    vertices_addr: Int,
    beams_addr: Int,
    lengths_addr: Int,
    too_short_addr: Int,
    beam_count: Int,
    min_length: Float64,
) abi("C"):
    var vertices = f32p(vertices_addr)
    var beams = i32p(beams_addr)
    var lengths = f64p(lengths_addr)
    var too_short = i32p(too_short_addr)
    for i in range(beam_count):
        var a = Int(beams[2 * i])
        var b = Int(beams[2 * i + 1])
        var dx = vertices[3 * a] - vertices[3 * b]
        var dy = vertices[3 * a + 1] - vertices[3 * b + 1]
        var dz = vertices[3 * a + 2] - vertices[3 * b + 2]
        var length = sqrt(dx * dx + dy * dy + dz * dz)
        lengths[i] = Float64(length)
        too_short[i] = 1 if Float64(length) < min_length else 0
