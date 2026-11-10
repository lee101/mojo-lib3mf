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
tests. There is no test today that fetches or executes the upstream C++ excerpts
that `tests/upstream_fixtures.py` carries, so they are documentation of the
intended reference, not an executed oracle.

The suite covers empty meshes, a single triangle, duplicate positions,
zero-area faces, repeated face indices, out-of-range indices, non-manifold
edges, unreferenced vertices, self-edge beams, and invalid balls. It runs 53
tests in about 9 s on this host, none skipped.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz (2 x 18
cores, 72 logical), Linux 6.8.0-147-generic, while the shared host carried a load
average of 30-42. The table is the minimum over seven `pixi run bench` runs on
unchanged code; see `bench/verify.md` for every raw run, including runs where
co-tenant load inflated one side by up to 2x:

| kernel | mojo-lib3mf | reference | speedup | reference implementation |
| --- | ---: | ---: | ---: | --- |
| transform 1M vertices | 3.284 ms | 12.223 ms | 3.72x | NumPy port of `fnMATRIX3_apply` |
| transform kernel 4M vertices | 9.549 ms | 15.006 ms | 1.57x | serial Mojo kernel |
| normals 250k triangles | 2.212 ms | 40.757 ms | 18.43x | NumPy port of `fnVEC3_calcTriangleNormal` |
| lengths 1M beams | 4.506 ms | 63.954 ms | 14.19x | NumPy port of `BeamLattice1702_Beams` |
| manifold 20k triangles | 2.660 ms | 15.369 ms | 5.78x | lib3mf 2.5.0 |

Each run is itself best of four calls, except manifold checking which is best of
three. Input and reference setup is outside each timed call; public operations that
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

**Transform, crossover** (best-of-15 per cell in each of three sweeps, speedup
over the serial kernel; the table shows the observed range across those sweeps).
The host is shared and sweeps drift by up to 20% either way, so the range is what
reproduces, not any single cell:

| vertices | serial | 2 workers | 4 workers | 8 workers |
| ---: | ---: | ---: | ---: | ---: |
| 1,500,000 | 5.16-5.92 ms | 0.67-0.71x | 0.76-0.84x | 0.79-0.90x |
| 2,000,000 | 7.04-7.66 ms | 0.73-0.79x | 0.88-0.95x | 0.96-1.07x |
| 2,500,000 | 8.97 ms (one sweep) | 0.81x | 0.98x | 1.12x |
| 3,500,000 | 12.78 ms (one sweep) | 1.00x | 1.26x | 1.24x |
| 4,000,000 | 13.99-14.95 ms | 0.95-1.13x | 1.43-1.54x | 1.40-1.49x |

The parallel path loses below ~2M vertices (0.67-0.90x at 1-1.5M), turns over
between 2M and 3M, and is ahead from ~3M up (1.24-1.61x on 4-8 workers). At
100k-1M vertices on 4 or 8 workers it is 0.04-0.67x of serial: the pool adds about
4.1-5.3 ms per call at these sizes, so 100k vertices runs about 25x slower on the
parallel path. The gate is **4,000,000 vertices**, which keeps those sizes on the
serial path and is also the size `pixi run bench` exercises. Beam lengths cross
later, about 2M beams:

| beams | serial | 2 workers | 4 workers | 8 workers |
| ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 4.28-4.65 ms | 0.57-0.60x | 0.56-0.74x | 0.59-0.78x |
| 1,500,000 | 6.45-6.58 ms | 0.66-0.76x | 0.73-0.92x | 0.79-0.97x |
| 2,000,000 | 8.95-9.21 ms | 0.86-0.89x | 1.02-1.06x | 0.98-1.15x |
| 3,000,000 | 13.10-14.90 ms | 1.08-1.19x | 1.37-1.45x | 1.38-1.53x |
| 4,000,000 | 18.50-19.16 ms | 1.19x | 1.68-1.73x | 1.72-1.86x |

The 2M row sits inside the noise (repeats gave 0.98-1.15x on 8 workers, 1.02-1.06x
on 4), so the gate is **3,000,000 beams**, where 4 workers measured 1.37-1.45x and
8 workers 1.38-1.53x. The bench's 1M-beam row stays on the serial path, where the
probe measured 0.56-0.78x for the parallel path it does not use.

**Worker cap: 4.** Four threads are used for both kernels. Transform scales
2.28-2.46x at 16M vertices on 4 workers and 2.50-2.88x on 8, a flattening gain for
twice the contention; beam lengths is the erratic one (8 workers gave 1.72-1.86x at
4M and 1.38-1.53x at 3M across sweeps). The cap is not raised.

**Before and after this phase** was measured in the phase that introduced the pool;
those runs are not from this verification session and are therefore no longer
quoted here. What the table above shows is the current split: the gated kernel at
4M vertices measured 9.55-23.07 ms against 15.01-22.49 ms for the serial kernel
over seven runs, a best case of 1.96x and a worst case of 0.65x when a co-tenant
stole the workers (6 of the 7 runs were 1.43x or better). "Before" that phase, `m3mf_transform_vertices_f32` split the
vertices into eight *serial* chunks in one call, labelled parallel but measured
at 0.86-1.05x of the serial kernel in `bench/analysis.md`. That chunking is
replaced by the real pool, not kept alongside it.

Gain flattens on this host because both kernels are memory-bound (24 MB per 1M
transform; a NumPy `add` over two 32 MB arrays measured 8.72 GB/s single-threaded
this session, and the 1M-vertex transform moves 24 MB in ~3.3 ms, ~7.3 GB/s) and
the host is shared; two sockets add a NUMA penalty on the split buffers. `m3mf_transform_vertices_f32_serial`
and the `TRANSFORM_PARALLEL_MIN`/`BEAM_LENGTHS_PARALLEL_MIN`/`PARALLEL_WORKERS`
constants in `src/lib3mf.mojo` are where to change the gates.

The phase that set the beam gate also recorded one 3M-beam call at 0.67x of
serial, which this session did not reproduce: three sweeps measured 1.37-1.45x on
4 workers and 1.38-1.53x on 8 at that size. The gate is therefore unchanged. The
sweeps do show that the beam kernel loses below 2M beams (0.56-0.97x across 1-1.5M
and 1-8 workers) and only clears 1.0x at 2-3M with 4-8 workers (0.98-1.15x at 2M,
1.08-1.53x at 3M), so a target machine with a slower pool can still lose here; if
that happens, raise `BEAM_LENGTHS_PARALLEL_MIN` or drop that gate.
The transform at 4M stayed at 1.43-1.54x on 4 workers in every sweep of this
session, but fell to 0.65x of serial in one of seven `pixi run bench` runs under
heavy co-tenant load, so it is not immune either.

## SIMD and fusion evaluation

`bench/analysis.md` ranked the hot contiguous loops and concluded manual SIMD
was not justified because the kernels sit at the machine's memory ceiling. That
claim was tested in the phase that produced this section (`simd_width_of[DType.float32]()`
is 8; the CPU has AVX2/FMA and no AVX-512). Its candidates were compiled as extra
`@export` entries next to the existing kernels and checked against them on the
same buffers before any timing. Those experiment kernels are not in this
commit, so the results below are that phase's record, not measurements from this
session:

| kernel | candidate | result |
| --- | --- | --- |
| `transform_vertices_f32_serial` | one 8-, 16- or 64-lane contiguous vector per iteration over the flattened `(n, 3)` buffer (`v * plane + bias`) | wrong values in rows the vector pass wrote, so not timed; the shipped scalar loop already uses about one vector register per vertex, so there is no widening to win |
| `triangle_normals_f32` | not attempted | per-face gather through the Int32 index array; the analysis treats it as gather-latency bound (3 MB index + 9 MB gathered vertices per 250k faces) |
| `beam_lengths_f32` | flags already fused into the length pass; a widened variant was abandoned | its strided index gather cannot be widened, and the length is `sqrt(Float32)` promoted to `Float64`, so the probe computed values in a different type rather than like-for-like |
| `transform_vertices_f32_serial` | software pipelining, 2 and 4 vertices in flight, identical per-vertex expressions (bit-identical results, asserted at 0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 20, 21, 1001 and 4099 vertices) | slower: pipe-2 0.89-0.99x and pipe-4 0.86-0.90x of serial in 5 of 6 runs (1M and 4M vertices, best of 20/8) |

The before/after `pixi run bench` table that this section originally carried was
measured in that phase, and its "before" column came from a tree that is not
checked out in this commit, so it is not reproduced here. What this session
measured is seven runs of the unchanged code; the ranges are in the main
benchmark table above (`transform 1M` 3.28-8.24 ms, `4M` 9.55-23.07 ms parallel
over 15.01-22.49 ms serial, `normals` 2.21-2.66 ms, `lengths` 4.51-7.12 ms,
`manifold` 2.66-3.09 ms). None of the candidates is retained; the kernels, their
results and the benchmark numbers are unchanged. The pipelined variant was
rejected on measurement, the
`transform` vector variants on correctness, and the remaining loops were not
candidates because their memory access is strided. Length-coverage tests for a
vector tail (`0, 1, W-1, W, W+1`, large odd) are only meaningful with a retained
SIMD path; with none, the existing tests are the coverage. Revisit SIMD only
with a kernel that is compute-bound rather than moving 24 MB per call.
