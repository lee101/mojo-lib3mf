"""Probe kernels for the `parallel` phase: serial vs max.algorithm parallelize.

Built by bench/parallel_probe.py. Not part of the shipped library; it exists so the
size threshold and worker count are chosen from measurement on this host.
"""

from max.algorithm import parallelize
from std.math import sqrt

comptime F32Ptr = Pointer[Float32, AnyOrigin[mut=True]]


def f32p(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def transform_range(
    vertices: F32Ptr, matrix: F32Ptr, result: F32Ptr, start: Int, end: Int
):
    for i in range(start, end):
        var x = vertices[3 * i]
        var y = vertices[3 * i + 1]
        var z = vertices[3 * i + 2]
        result[3 * i] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
        result[3 * i + 1] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
        result[3 * i + 2] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]


@export("probe_transform_serial")
def probe_transform_serial(
    vertices_addr: Int, matrix_addr: Int, result_addr: Int, count: Int
) abi("C"):
    transform_range(f32p(vertices_addr), f32p(matrix_addr), f32p(result_addr), 0, count)


@export("probe_transform_parallel")
def probe_transform_parallel(
    vertices_addr: Int, matrix_addr: Int, result_addr: Int, count: Int, workers: Int
) abi("C"):
    if workers <= 1 or count <= 0:
        transform_range(
            f32p(vertices_addr), f32p(matrix_addr), f32p(result_addr), 0, count
        )
        return
    var vertices = f32p(vertices_addr)
    var matrix = f32p(matrix_addr)
    var result = f32p(result_addr)
    def work(item: Int) {imm}:
        transform_range(
            vertices,
            matrix,
            result,
            count * item // workers,
            count * (item + 1) // workers,
        )
    parallelize(work, workers, workers)


def beam_lengths_range(
    vertices: F32Ptr,
    beams: Pointer[Int32, AnyOrigin[mut=True]],
    lengths: Pointer[Float64, AnyOrigin[mut=True]],
    too_short: Pointer[Int32, AnyOrigin[mut=True]],
    min_length: Float64,
    start: Int,
    end: Int,
):
    for i in range(start, end):
        var a = Int(beams[2 * i])
        var b = Int(beams[2 * i + 1])
        var dx = vertices[3 * a] - vertices[3 * b]
        var dy = vertices[3 * a + 1] - vertices[3 * b + 1]
        var dz = vertices[3 * a + 2] - vertices[3 * b + 2]
        var length = sqrt(dx * dx + dy * dy + dz * dz)
        lengths[i] = Float64(length)
        too_short[i] = 1 if Float64(length) < min_length else 0


@export("probe_beam_lengths_serial")
def probe_beam_lengths_serial(
    vertices_addr: Int,
    beams_addr: Int,
    lengths_addr: Int,
    too_short_addr: Int,
    beam_count: Int,
    min_length: Float64,
) abi("C"):
    beam_lengths_range(
        f32p(vertices_addr),
        Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=beams_addr),
        Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=lengths_addr),
        Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=too_short_addr),
        min_length,
        0,
        beam_count,
    )


@export("probe_beam_lengths_parallel")
def probe_beam_lengths_parallel(
    vertices_addr: Int,
    beams_addr: Int,
    lengths_addr: Int,
    too_short_addr: Int,
    beam_count: Int,
    min_length: Float64,
    workers: Int,
) abi("C"):
    if workers <= 1 or beam_count <= 0:
        probe_beam_lengths_serial(
            vertices_addr, beams_addr, lengths_addr, too_short_addr, beam_count,
            min_length,
        )
        return
    var vertices = f32p(vertices_addr)
    var beams = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=beams_addr)
    var lengths = Pointer[Float64, AnyOrigin[mut=True]](
        unsafe_from_address=lengths_addr
    )
    var too_short = Pointer[Int32, AnyOrigin[mut=True]](
        unsafe_from_address=too_short_addr
    )
    def work(item: Int) {imm}:
        beam_lengths_range(
            vertices,
            beams,
            lengths,
            too_short,
            min_length,
            beam_count * item // workers,
            beam_count * (item + 1) // workers,
        )
    parallelize(work, workers, workers)
