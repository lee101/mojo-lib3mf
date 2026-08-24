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

Vertex transforms use the compiler-vectorized serial loop below 4 million
vertices. At and above that measured crossover they split into eight independent,
balanced CPU tasks; smaller inputs avoid thread-pool launch overhead. Explicit
strided SIMD gathers were benchmarked but not retained because they were slower
than Mojo's vectorization of the contiguous AoS loop on this machine.

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

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic:

| kernel | mojo-lib3mf | reference | speedup | reference implementation |
| --- | ---: | ---: | ---: | --- |
| transform 1M vertices | 3.817 ms | 21.619 ms | 5.66x | NumPy port of `fnMATRIX3_apply` |
| transform kernel 4M vertices | 10.615 ms | 15.648 ms | 1.47x | serial Mojo kernel |
| normals 250k triangles | 2.794 ms | 59.910 ms | 21.44x | NumPy port of `fnVEC3_calcTriangleNormal` |
| lengths 1M beams | 5.146 ms | 66.455 ms | 12.91x | NumPy port of `BeamLattice1702_Beams` |
| manifold 20k triangles | 2.835 ms | 16.129 ms | 5.69x | lib3mf 2.5.0 |

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
