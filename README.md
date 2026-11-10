# mojo-lib3mf

`mojo-lib3mf` is a focused Mojo port of
[lib3mf](https://github.com/3MFConsortium/lib3mf) for reading, writing, checking,
and transforming common 3MF models. It is a standalone package: lib3mf is used
as the test and benchmark reference, not as a runtime dependency of the public
API.

This is a derived work of lib3mf, which is distributed under the
[BSD 2-Clause License](https://github.com/3MFConsortium/lib3mf/blob/master/LICENSE).
The port itself is MIT licensed; upstream copyright and license text are in
[NOTICE](NOTICE).

## Coverage

Implemented:

- OPC ZIP containers, root relationships, content types, and the root 3D model part
- 3MF Core 2015 model XML: units, language, top-level metadata, mesh objects,
  components objects, and build items
- Float32 vertices, Int32 triangles, triangle normals, upstream mesh sanity rules,
  and oriented-manifold checking
- 3MF column-major 3x4 transform strings, bulk vertex transforms, 4x4 composition,
  identity checking, and planar-transform checking
- Beam Lattice 2017 beams, endpoint radii, cap modes, minimum-length warnings,
  and the 2020 balls extension

Not implemented are beam sets, materials and per-triangle properties, textures, thumbnails
and arbitrary attachments, slices, production UUID/path features, secure content,
volumetric and implicit extensions, triangle sets, and preservation of unknown
extension XML. OPC and XML orchestration use Python's standard library; the
compute-bound geometry paths are the Mojo port.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles the single Mojo unit to
`dist/libmojo-lib3mf.so`. The Python package is placed on `PYTHONPATH` by the
Pixi environment.

## Usage

This complete example creates a tetrahedron, applies a build-item transform,
writes a standards-compliant OPC package, and reads it back:

```python
import numpy as np
import mojo_lib3mf as m3

vertices = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    dtype=np.float32,
)
triangles = np.array(
    [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    dtype=np.int32,
)

mesh = m3.Mesh(vertices, triangles)
assert mesh.check_sanity()
assert mesh.is_manifold_and_oriented()

transform = m3.identity_transform()
transform[:3, 3] = [10, 20, 30]
model = m3.Model(
    objects=[m3.MeshObject(1, mesh, name="tetrahedron")],
    build_items=[m3.BuildItem(1, transform)],
)
model.write("tetrahedron.3mf")

loaded = m3.Model.read("tetrahedron.3mf")
print(loaded.objects[0].mesh.triangle_normals())
```

The same code is checked in and can be run with:

```bash
pixi run python examples/basic.py
```

It writes `tetrahedron.3mf` in the current directory.

Beam lattices attach directly to a mesh:

```python
lattice = m3.BeamLattice(
    beams=[[0, 1], [1, 2]],
    radii=[[0.10, 0.15], [0.15, 0.15]],
    caps=[("butt", "hemisphere"), ("sphere", "sphere")],
    default_radius=0.10,
)
beam_mesh = m3.Mesh(vertices, triangles, lattice)
lengths, below_minimum = beam_mesh.beam_lengths()
```

## How it works

Hot data stays in flat, C-contiguous structure-of-arrays-compatible buffers:
vertices are `(n, 3)` Float32, triangles and beams are Int32 index arrays, radii
are Float64, and transforms are row-major `(4, 4)` Float32 matrices. Python owns
all buffers. Their integer addresses cross a small `ctypes` C ABI into one Mojo
shared library, so the Mojo side neither allocates Python-visible memory nor
retains pointers.

NumPy vertex and transform inputs must already be `float32`; they are not
silently narrowed. Integer index arrays are range-checked before conversion to
`int32`. Python sequences are converted to these documented storage types.
All buffers are made C-contiguous and remain owned and live for the full
synchronous FFI call.

The OPC layer follows lib3mf's package constants and relationship traversal.
The XML layer follows its Core 2015 and Beam Lattice readers and writers,
including radius fallback rules, cap aliases, coordinate bounds, transform wire
order, the `1e-10` normal cutoff, `1e-7` planar and identity choices, and the
`0.0001` beam minimum default. Oriented-manifold checking ports
`CModelMeshObject::isManifoldAndOriented`: every undirected edge must occur once
in each direction.

Vertex transforms and beam lengths use the compiler-vectorized serial loop below
a measured size threshold; at and above it they split across a real
`max.algorithm.parallelize` worker pool, so smaller inputs avoid pool overhead.
The gates are 4,000,000 vertices and 3,000,000 beams with 4 workers each; the
measurements behind them, and the sizes where the gain flattens, are in
[Parallelism](#parallelism). Explicit strided SIMD gathers were benchmarked but
not retained because they were slower than Mojo's vectorization of the
contiguous AoS loop on this machine.

Hand-written SIMD was evaluated again and again not retained; see
[SIMD and fusion evaluation](#simd-and-fusion-evaluation). The contiguous
transform loop compiles to a scalar loop of 3 loads, 3 `vfmadd231ss`, 3
`vaddss` and 3 stores per vertex: one vector register's worth of work per
vertex, with no widening pass at all.

## Correctness

The tests install the official `lib3mf` 2.5.0 Python binding. Packages are
cross-read in both directions, and meshes, build/component transforms, beam
radii, cap modes, balls, and manifold results are compared against that binding.
The binding does not expose lib3mf's internal bulk matrix, normal, or beam-length
functions; those kernels use NumPy references transcribed from the cited
upstream source functions, plus exact epsilon boundary and degenerate-input
tests.

The suite covers empty meshes, a single triangle, duplicate positions,
zero-area faces, repeated face indices, out-of-range indices, non-manifold
edges, unreferenced vertices, self-edge beams, and invalid balls.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz (2 x 18
cores, 72 logical), Linux 6.8.0-147-generic, while the shared host carried a load
average of 15-19:

| kernel | mojo-lib3mf | reference | speedup | reference implementation |
| --- | ---: | ---: | ---: | --- |
| transform 1M vertices | 3.511 ms | 13.508 ms | 3.85x | NumPy port of `fnMATRIX3_apply` |
| transform kernel 4M vertices | 10.622 ms | 16.287 ms | 1.53x | serial Mojo kernel |
| normals 250k triangles | 2.443 ms | 42.230 ms | 17.29x | NumPy port of `fnVEC3_calcTriangleNormal` |
| lengths 1M beams | 5.085 ms | 67.814 ms | 13.34x | NumPy port of `BeamLattice1702_Beams` |
| manifold 20k triangles | 2.737 ms | 15.839 ms | 5.79x | lib3mf 2.5.0 |

Times are best of four runs, except manifold checking which is best of three.
Input and reference setup is outside each timed call; public operations that
return arrays allocate those results inside the timing. The explicitly labeled
4M kernel row uses the same preallocated buffers for both paths to isolate the
parallel crossover. These are CPU and workload-specific results; rerun the
locked benchmark task on the target machine rather than treating the ratios as
universal.

No GPU path is provided. The bulk numeric kernels move substantially more than
one byte per two floating-point operations, while manifold checking is an
integer sort with irregular memory access. All are below the arithmetic
intensity needed to repay device transfer and launch overhead.

## Parallelism

Two kernels are size-gated onto `max.algorithm.parallelize` with an explicit
worker count (`std.algorithm`/`std.threading` forms do not compile on this
toolchain). Everything else stays serial, and sizes below the gate never touch
the pool.

`bench/parallel_probe.py` builds `bench/parallel_probe.mojo`, which carries the
same two loops next to a `max.algorithm` variant, and sweeps size x worker count
(1, 2, 4, 8), checking every parallel result against the serial output. It runs
best-of-15 per cell. Rerun it with `pixi run python bench/parallel_probe.py
--workers 1,2,4,8`.

**Transform, crossover** (best-of-25, speedup over the serial kernel). Repeat
sweeps drift by 5-10% in either direction because the host is shared; the shape
is what matters:

| vertices | serial | 2 workers | 4 workers | 8 workers |
| ---: | ---: | ---: | ---: | ---: |
| 1,500,000 | 5.47 ms | 0.65x | 0.82x | 0.94x |
| 2,000,000 | 7.16 ms | 0.72x | 0.99x | 1.13x |
| 2,500,000 | 10.13 ms | 0.89x | 1.22x | 1.33x |
| 3,500,000 | 12.82 ms | 1.05x | 1.38x | 1.53x |
| 4,000,000 | 14.32 ms | 1.03x | 1.39x | 1.61x |

The parallel path loses below ~2M vertices (0.82-1.05x at 1-1.5M, and at 100k-1M
it is 0.29-0.64x), turns over between 2M and 3M, and is consistently ahead from
2.5M up. The gate is **4,000,000 vertices**, about 1.3x above the crossover, which
is also the size `pixi run bench` already exercises. Beam lengths cross later,
about 2M beams:

| beams | serial | 2 workers | 4 workers | 8 workers |
| ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 4.76 ms | 0.69x | 0.81x | 0.92x |
| 1,500,000 | 6.72 ms | 0.80x | 0.98x | 1.00x |
| 2,000,000 | 8.68 ms | 0.90x | 1.09x | 1.20x |
| 3,000,000 | 12.66 ms | 0.97x | 1.28x | 1.22x |
| 4,000,000 | 17.53 ms | 1.07x | 1.53x | 2.06x |

The 2M row sits inside the noise (separate repeats gave 0.81-1.09x), so the gate
is **3,000,000 beams**, where it measured >=1.28x. The bench's 1M-beam row stays
on the serial path and is expected to be unchanged.

**Worker cap: 4.** Four threads are used for both kernels. Transform scales
2.59x at 16M vertices on 4 workers and 2.85x on 8, a flattening gain for twice
the contention; beam lengths is the erratic one (8 workers gave 2.06x at 4M in
one repeat and 1.80x at 3M in the next). The cap is not raised.

**Before and after**, `pixi run bench`, same host and harness:

| kernel | before this phase | after | note |
| --- | ---: | ---: | --- |
| transform 1M vertices | 3.601 ms | 3.511 ms | below the gate, still serial (noise) |
| transform kernel 4M vertices | 14.625 ms / 14.982 ms serial | 10.622 ms / 16.287 ms serial | 1.02x -> 1.53x |
| lengths 1M beams | 5.032 ms | 5.085 ms | below the gate, still serial (noise) |

"Before" is the shipped code, whose `m3mf_transform_vertices_f32` split the
vertices into eight *serial* chunks in one call, labelled parallel but measured
at 0.86-1.05x of the serial kernel in `bench/analysis.md`. That chunking is
replaced by the real pool, not kept alongside it.

Gain flattens on this host because both kernels are memory-bound (24 MB per 1M
transform, measured at ~8 GB/s single-threaded) and the host is shared; two
sockets add a NUMA penalty on the split buffers. `m3mf_transform_vertices_f32_serial`
and the `TRANSFORM_PARALLEL_MIN`/`BEAM_LENGTHS_PARALLEL_MIN`/`PARALLEL_WORKERS`
constants in `src/lib3mf.mojo` are where to change the gates.

One caveat from the sweep above: with the pool warmed, a 3M-beam call measured
0.67x of serial in one repeat, so the beam gate can still lose when the host is
contended. If that pattern repeats on the target machine, raise
`BEAM_LENGTHS_PARALLEL_MIN` or drop that gate; the transform at 4M stayed
>=1.35x in every repeat.

## SIMD and fusion evaluation

`bench/analysis.md` ranked the hot contiguous loops and concluded manual SIMD
was not justified because the kernels sit at the machine's memory ceiling. That
claim was tested directly on this host (`simd_width_of[DType.float32]()` is 8;
the CPU has AVX2/FMA and no AVX-512). Every candidate was compiled as an extra
`@export` next to the existing kernels and checked against them on the same
buffers before any timing:

| kernel | candidate | result |
| --- | --- | --- |
| `transform_vertices_f32_serial` | one 8-, 16- or 64-lane contiguous vector per iteration over the flattened `(n, 3)` buffer (`v * plane + bias`) | wrong values in rows the vector pass wrote, so not timed; the shipped scalar loop already uses about one vector register per vertex, so there is no widening to win |
| `triangle_normals_f32` | not attempted | per-face gather through the Int32 index array; the analysis measured it as gather-latency bound (3 MB index + 9 MB gathered vertices per 250k faces) |
| `beam_lengths_f32` | flags already fused into the length pass; a widened variant was abandoned | its strided index gather cannot be widened, and the length is `sqrt(Float32)` promoted to `Float64`, so the probe computed values in a different type rather than like-for-like |
| `transform_vertices_f32_serial` | software pipelining, 2 and 4 vertices in flight, identical per-vertex expressions (bit-identical results, asserted at 0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 20, 21, 1001 and 4099 vertices) | slower: pipe-2 0.89-0.99x and pipe-4 0.86-0.90x of serial in 5 of 6 runs (1M and 4M vertices, best of 20/8) |

`pixi run bench` before and after the evaluation, four runs each, with `src/`
checked out and rebuilt from the same commit (the host was under unrelated
load, so these are the same code measured twice):

| kernel | before (4 runs) | after (4 runs) |
| --- | ---: | ---: |
| transform 1M vertices | 3.525, 3.554, 3.804, 4.453 ms | 3.666, 4.383, 3.578, 3.626 ms |
| transform kernel 4M vertices | 16.139, 13.793, 15.149, 15.237 ms | 16.061, 16.291, 16.040, 15.367 ms |
| normals 250k triangles | 2.441, 2.485, 2.465, 2.476 ms | 2.396, 2.538, 2.380, 2.882 ms |
| lengths 1M beams | 5.104, 5.422, 5.140, 5.100 ms | 5.092, 7.097, 5.117, 5.219 ms |
| manifold 20k triangles | 2.740, 2.724, 2.718, 2.734 ms | 2.579, 2.816, 2.851, 2.843 ms |

None of these is retained; the kernels, their results and the benchmark numbers
are unchanged. The pipelined variant was rejected on measurement, the
`transform` vector variants on correctness, and the remaining loops were not
candidates because their memory access is strided. Length-coverage tests for a
vector tail (`0, 1, W-1, W, W+1`, large odd) are only meaningful with a retained
SIMD path; with none, the existing tests are the coverage. Revisit SIMD only
with a kernel that is compute-bound rather than moving 24 MB per call.
