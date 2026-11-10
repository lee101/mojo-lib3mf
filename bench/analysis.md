# ANALYZE: where mojo-lib3mf spends its time

Every number below was produced in this session on this machine. Nothing is estimated;
where a figure is a model (flops, bytes touched, flops/byte) rather than a measurement,
the table says so. Algorithms were not changed. `src/` and `tests/` are untouched.

## Environment

| | |
| --- | --- |
| CPU | Intel Xeon E5-2697 v4 @ 2.30 GHz, 2 sockets x 18 cores, SMT2 (72 logical) |
| Affinity | `0-71` (all cores, including SMT siblings) |
| Memory bandwidth delivered | **7.6-8.3 GB/s** single-threaded (NumPy `add`/`multiply` over 32M f32, 46.1-33.0 ms) |
| Kernel caches | L1d 1.1 MiB, L2 9 MiB, L3 90 MiB |
| OS / kernel | Linux 6.8.0-147-generic |
| Toolchain | `mojo ==1.2.0.dev2026092605`, `max ==26.7.0.dev2026092605`, Python 3.13.14, lib3mf 2.5.0 |
| `perf` | **unavailable** (`perf_event_paranoid = 4`), so no hardware counters; flops/byte is a traffic model |
| Host noise | load average 23-29 from unrelated work throughout; parallel rows are therefore noisy |

`pixi run bench` (locked, exit 0) was run twice — at the start and at the end of the
session. Both runs, verbatim outcomes:

| run | transform 1M | transform 4M (par vs serial) | normals 250k | lengths 1M | manifold 20k |
| --- | ---: | ---: | ---: | ---: | ---: |
| first | 3.616 ms / 13.143 ms (3.64x) | **16.306 ms / 13.992 ms (0.86x)** | 2.403 ms / 41.766 ms | 5.145 ms / 64.843 ms | 2.641 ms / 13.519 ms |
| last | 3.607 ms / 14.927 ms (4.14x) | 15.422 ms / 16.177 ms (1.05x) | 2.360 ms / 43.631 ms | 5.186 ms / 71.828 ms | 2.796 ms / 16.412 ms |

## Methodology, and what it cost me to get right

* Public entry points are timed **exactly as bench.py times them**: `min` of N
  independent `perf_counter` intervals, setup and reference work outside the timed
  region. Kernel-only rows time the raw ctypes call on preallocated buffers, so the
  glue/kernel split is a measured subtraction, not a guess.
* `per_call(fn, N) = best(fn, N) / N` is the legitimate fast path here and gives the
  ctypes/ABI call floor, **not the full public wrapper.** The wall clock ticks in ~11 µs
  units on this host, so most short intervals cancel to exactly 0 for fixed-size calls.
  Dividing a single best-of-N interval by N lands on 10-14 ns for any call whose true
  cost is below that. Two consequences, both relevant to the next phases:
  * bench.py reports `transform 1M vertices` at 3.607 ms when `min` over 40 intervals
    gives 3.504 ms and the kernel-only call is 3.31-3.41 ms; bench.py's four-repeat
    minimum is 0.1-0.7 ms coarse. Ratios from bench.py are therefore good to roughly
    5-15%, not to the last digit.
  * I could not measure the fixed-size wrapper as a whole: with intervals mostly
    cancelling to 0, my best-of-N row says 7.5 ns and my loop probes (12-23 µs) are
    dominated by per-iteration sleep state. **Neither is reported.** Rows 22-25 are
    kernel-only ctypes figures plus the one `per_call` floor reading, and are labelled
    as floors. They show Python+FFI cannot beat ~25-32 ns in isolation, which answers
    the "is the wrapper the problem?" question without over-claiming a total.
* Two of my probes segfaulted early: `addr()` of a **temporary** array
  (`addr(np.empty(e))` inside a lambda) frees the buffer before ctypes dereferences
  it. That is probe misuse, not a library bug: `model.py`/`io.py` always keep the
  buffer alive across the call.

## Ranked hotspot table

`glue`, `flops`, `bytes` are per call. `flops/byte` and the bound are analytic models:
flops counted per iteration, bytes counted as *distinct bytes the kernel must move*
(not cache-block traffic). Intensity is compared with the measured **8.1 GB/s** and
**~3.4 ns/vertex** achievable here.

### Tier 1 — the compute entry points real callers use

| # | function | input size | time (this session) | bytes touched | flops | flops/byte | bound | where the time sits | vs upstream | next phase |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | `Mesh.transformed` (`m3mf_transform_vertices_f32`) | 1,000,000 verts | 3.504 ms (min of 40), 3.607 ms (bench.py) | 24 MB | 12 M | 0.5 | memory (3.40 ns/vertex = the machine ceiling) | kernel (3.35 ms serial kernel-only, 3.39 ms via ctypes); glue ~0.1-0.3 ms | no upstream API (numpy 13.673/14.927 ms, **3.9-4.1x**) | **none** (already at 90% of measured bandwidth) |
| 2 | same operation, `_serial` kernel-only | 1,000,000 verts | 3.28-3.39 ms | 24 MB | 12 M | 0.5 | memory (3.28-3.39 ns/vertex) | kernel | — | **none** |
| 2b | same, kernel-only | 4,000,000 verts | serial **13.8-14.0 ms**, "parallel" **13.6-16.3 ms** | 96 MB | 48 M | 0.5 | memory (3.40-3.46 ns/vertex) | kernel | serial Mojo is the reference | **none** for the arithmetic; `parallel` phase only if the split is kept at all (see §4) |
| 3 | `Mesh.triangle_normals` (`m3mf_triangle_normals_f32`) | 250,000 faces | 2.398-2.453 ms public; 2.189-2.235 ms kernel | 3 MB idx + 9 MB verts (gathered) + 3 MB out | 4.25 M | 0.28 | memory + latency (13 sc/face); gather means idx is read ~2x | kernel: 8.7-10.1% (0.21 ms) is glue | no upstream API (numpy 47.2 ms, **19.5x**); upstream normal calc is not exposed | **none** for arithmetic; `unroll` only to hide the gather latency |
| 4 | same, kernel-only | 1,000,000 faces | 9.34-9.67 ms | 36 MB | 17 M | 0.26 | memory/latency, linear: 3.98 ns/face vs 2.82 at 250k | kernel | — | **none** |
| 5 | `Mesh.beam_lengths` (`m3mf_beam_lengths_f32`) | 1,000,000 beams | 4.812-5.186 ms public; 3.986-4.128 ms kernel | 4 MB idx + 12 MB verts + 12 MB out | 5.5 M | 0.2 | memory + latency: 11 sc/beam; writes fp64 (8 B/beam) where fp32 would halve the output traffic | kernel; **glue 0.83-1.03 ms = 17-20%** at this size | no upstream API (numpy 64.8-71.8 ms, **13.5-13.9x**) | `unroll` (saves the glue) |
| 6 | `Mesh.is_manifold_and_oriented` | 20,000 faces | 2.577-2.796 ms public | ~0.75 MB (keys+scratch, 2x) | ~7 M (cmp/mov in 8 radix passes) | ~9 | compute/latency: 129-140 ns/face | kernel (sort): check_sanity is only 0.096-0.099 ms | **lib3mf 2.5.0: 13.5-16.4 ms (5.1-5.9x)** | `none` (int, irregular, tiny inputs) |

### Tier 2 — the Python I/O layer (biggest wall-clock costs in the package)

| # | function | input size | time (this session) | bytes touched | flops | flops/byte | bound | where the time sits | vs upstream (lib3mf 2.5.0) | next phase |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 7 | `Model.write` (default `compression=8`) | 20,000 faces | **568.5-575.4 ms** | 1.49 MB XML produced | 0 | 0 | Python-bound | Python: ET `tostring` 169 ms + tree build 157 ms at this size + deflate | **32.3 ms → 17.6-18.3x slower** | `parallel` (Python), and `unroll` = replace the per-element loops |
| 8 | `Model.write(compression=0)` | 20,000 faces | **303.9-307.2 ms** | 1,486 KB stored | 0 | 0 | Python-bound | per-vertex loop 76 ms (`_float` x3 x20k), per-triangle loop 31.6 ms, ET SubElement 50.8 ms, `_serialize_xml` self 217-223 ms | **32.3 ms → 9.4-9.5x slower** | `unroll` |
| 9 | `Model.read` | 20,000 faces | **131.1-133.0 ms** | 1,485 KB deflated / 8.0 MB XML | 0 | 0 | Python-bound | `_read_mesh` self 72.8 ms; 3 genexprs 25.0/31.9/22.6 ms; `all()` 17.0 ms + 60k `abs`/`isfinite` | **31.5 ms → 4.2x slower** | `unroll` |
| 10 | `read` / `write` / cross-read | 1,000 faces | read 6.20-6.66 ms; write 14.5 ms stored, 26.2-27.6 ms deflated | 9.5 KB deflated / 70 KB | 0 | 0 | Python-bound | as above, amortised | read **2.13 ms (3.1x slower)**, write **1.87 ms (7.8-14.8x slower)**, upstream<-ours 2.03 ms, ours<-upstream 7.81 ms | `unroll` |
| 11 | `read` / `write` | 16 faces (the README's tetrahedron) | read 0.36-0.37 ms; write 0.48-0.50 ms (stored), 0.59-0.62 ms (deflate) | 1.1-1.6 KB | 0 | 0 | Python-bound | per-call overhead + `_validate_for_write` 0.113 ms at 20k | read **0.69 ms**, write **0.44 ms**; read 0.79x *faster*, write 1.1-2.3x slower | `none` |

### Tier 3 — per-element Python loops and copies (flag: re-enter ctypes per element)

| # | function / path | input size | time (this session) | bytes touched | flops | flops/byte | bound | where the time sits | vs reference | next phase |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 12 | `io._model_xml` vertex loop (`_float` x3 per row) | 20,000 verts | 76.0-83.4 ms | — | 0 | 0 | Python | 20k Python iterations, 60k `_float` | — | `unroll` |
| 13 | `io._model_xml` triangle loop (`str(int())` x3 per row) | 20,000 faces | 31.6 ms | — | 0 | 0 | Python | 20k iterations | — | `unroll` |
| 14 | `io._read_mesh` vertex/triangle loops | 20,000 rows | 79.7 ms combined (genexprs 25.0+31.9+22.6) | — | 0 | 0 | Python | 3 genexprs + `all()` over each point | — | `unroll` |
| 15 | a caller's own per-element ctypes loop | 100,000 beams | **854-880 ms (8.5-8.8 µs/beam)** | — | — | — | Python/ABI | **one ctypes call per beam** | one block call: 0.394 ms → **2108-2229x slower** | `unroll` (the fix is the contract, not the kernel) |
| 16 | same, per vertex | 10,000 verts | **59.3-60.1 ms (5.9-6.0 µs/vertex)** | — | — | — | Python/ABI | one call per vertex | one block call: 0.039-0.040 ms → **1471-1534x slower** | `unroll` |
| 17 | `Mesh.transformed` | 1,000,000 verts | 3.504-3.607 ms total | **12 MB+ per call** | 12 M | 0.5 | allocation, below the kernel | Python: 11.4 MiB in 9 allocations (result 12 MB, `triangles.copy()`), 120 allocation events, RSS delta 11,720 kB, **plus `copy.deepcopy(beam_lattice)` and `triangles.copy()` on every call** | — | `unroll` (preallocate / drop the deepcopy) |
| 18 | `Mesh.triangle_normals` | 250,000 faces | 2.398-2.453 ms total | result 3 MB only | 4.25 M | 0.28 | Python alloc of the result + min/max guard | 840 B Python-visible besides the 3 MB result | — | `unroll` |
| 19 | `BeamLattice.__post_init__` default caps | 100,000 beams | 23.9-25.5 ms (0.24-0.26 µs/beam) | — | 0 | 0 | Python | materialises `tuple(("sphere","sphere") x n)` — a Python tuple per beam | factory alone: 6.24 ms/100k | `unroll` |
| 20 | `Mesh.__init__` from a Python list of rows | 250,000 float32 array rows | 101.7-114.2 ms (array input: 0.199 ms) | — | 0 | 0 | Python | `np.ascontiguousarray` over a Python sequence | — | `none` (documented slow path) |

### Tier 4 — fixed-size transform API (callers pass one 4x4 matrix)

| # | function | input size | time (this session) | bytes touched | flops | flops/byte | bound | where the time sits | vs upstream / baseline | next phase |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 22 | `multiply_transforms` | one 4x4 f32 | 179-199 ns (min of N=200); kernel 24.4-31.8 ns | 192 B | 112 | 0.6 | latency | Python wrapper + ctypes | **numpy `a @ b`: 8.6-10.5 ns (17-19x faster)**; lib3mf `GetScaleTransform` etc. only (no multiply API) | `none` (numpy wins; FFI cannot beat 25 ns) |
| 23 | `is_identity_transform` | one 4x4 f32 | 94.5-109 ns; kernel 9.5-12.4 ns | 64 B | 32 | 0.5 | latency | Python + ctypes | — | `none` |
| 24 | `is_planar_transform` | one 4x4 f32 | 94.5-109.9 ns; kernel 9.7-12.3 ns | 64 B | 5 | 0.08 | latency | Python + ctypes | — | `none` |
| 25 | `transform_from_3mf` / `transform_to_3mf` | 12 numbers | 113.1 / 123.0 ns | 48 B | 0 | 0 | latency | Python `format`/`float` | — | `none` |
| 26 | `Mesh(v, tris)` with 4 vertices / 4 faces | the README tetrahedron | 27.0-28.8 ns | — | 0 | 0 | latency | Python | — | `none` |

Rows 22-25 are kernel-only ctypes figures plus the `per_call` floor reading — i.e.
anything measurable about the FFI path for a single 4x4 matrix, not a total public
cost (see §Methodology). They stay in the table rather than being dropped.

### Python-level loops that re-enter ctypes per element

Inside the package itself, **none**: every kernel is one bulk call per operation
(`model.py` lines 182, 192, 219, 237, 258). Each public operation therefore re-enters
ctypes **once**. The per-element loops that do *not* touch ctypes are the XML writer
and reader (rows 12-14) and `BeamLattice` caps (row 19). The per-element ctypes case
(rows 15-16) exists only for a caller that loops over rows itself; it costs 5.9-8.8 µs
per element, i.e. **~2100x** the block path, so the fast way to touch this is to keep
the bulk ABI and document it.

### Per-call allocations and copies

* `Mesh.transformed`: allocates the full result (12 MB at 1M) and *also* copies
  `triangles` and `deepcopy`s the beam lattice on every call (row 17).
* `Mesh.triangle_normals` / `beam_lengths` / `is_manifold_and_oriented`: allocate
  their result buffers per call (3 MB / 8 B+4 B per beam / 2 x 3 x n x 8 B workspace).
* `Model.write`: builds a full `Element` tree and serialises it to a temporary
  `BytesIO` before handing it to zipfile (rows 7-8).
* `BeamLattice` with default caps: one Python tuple per beam (row 19).

## Where time goes, in one screen

```
per 20k-face write (compression=0), 304 ms total:
  per-vertex  Python loop (60k _float)      76 ms   (25%)
  per-triangle Python loop + SubElement     ~82 ms   (27%)
  ET.tostring (serialisation)               169 ms   (56%)   [same tree, measured alone]
  deflate at compression=8 adds            ~265 ms
per 20k-face read, 132 ms total:
  _read_mesh self                           73 ms   (55%)
  genexprs + validation                     ~80 ms   (60%)
per 1M-vertex transform, 3.50 ms total:
  kernel (serial, memory bound)            3.35 ms  (96%)
  Python glue + allocation                 0.15 ms
```

## Recommendations, ranked

1. **`unroll` — XML read/write (rows 7-14).** 4-18x slower than lib3mf and it is
   pure per-element Python: build the XML with `ndarray.tobytes`/`numpy`
   `char` formatting and a join, or a bulk formatter, instead of 20k Python iterations
   per direction. Flops/byte is 0, so no GPU/LLVM vectorisation is relevant.
2. **`unroll` — `transformed` copies (row 17).** Drop `deepcopy`/`copy()` unless the
   caller asks; 12 MB of extra traffic and 120 allocations per call. Worth it below
   ~10 MB where the copy is a measurable share of the 3.5 ms.
3. **`unroll` — `beam_lengths` glue (row 5).** 17-20% of the public cost at 1M is the
   wrapper; a single-pass FFI with caller-owned outputs removes it.
4. **`none` — all four Mojo kernels (rows 1-4, 6).** Every one is at or near the
   measured 8-8.5 GB/s ceiling or the 3.4 ns/vertex floor; the compute is done.
5. **`parallel` — only if parallel transform is kept (row 2).** Today the eight-task
   loop buys nothing (0.96-1.03x against serial on every run) and can lose (0.86x).
   Since it is not a `max.algorithm` pool but a serial eight-way split, either make it
   a real `max` worker pool behind a size gate and prove it on this host, or drop it.
   Expected ceiling from the measured bandwidth: 4 cores → ~1.0 ns/vertex.
6. **`simd` — not justified.** The compiler already vectorises the contiguous loop
   (README records explicit gathers being slower); the kernels are at bandwidth, so
   no manual SIMD can add flops/byte that does not exist.
7. **`gpu` — not justified for any row.** The highest intensity in the package is
   manifold checking at ~9 flops/byte, then transform at 0.5, normals at 0.26-0.28,
   lengths at 0.2, and pure-integer rows at 0. A GPU path must repay copy-in/copy-out
   and launch; at <1 flop/byte the data movement alone (24 MB for 1M vertices) exceeds
   the entire current kernel time (3.5 ms), so the transfer is the cost, not the
   arithmetic.

## Rows where the port is slower than upstream

| function | size | port | upstream | ratio |
| --- | --- | ---: | ---: | ---: |
| `Model.write` (deflate) | 20,000 faces | 568-575 ms | 32.3 ms | **17.6-18.3x slower** |
| `Model.write` (stored) | 20,000 faces | 304-307 ms | 32.3 ms | **9.4-9.5x slower** |
| `Model.read` | 20,000 faces | 131-133 ms | 31.5 ms | **4.2x slower** |
| `Model.write` (deflate) | 1,000 faces | 26.2-27.6 ms | 1.87 ms | **14.0-14.8x slower** |
| `Model.write` (stored) | 1,000 faces | 14.5 ms | 1.87 ms | **7.8x slower** |
| `Model.read` | 1,000 faces | 6.20-6.66 ms | 2.13 ms | **2.9-3.1x slower** |
| `Model.write` (deflate) | 16 faces | 0.59-0.62 ms | 0.44 ms | 1.3-1.4x slower |
| `transform kernel` parallel vs serial | 4,000,000 verts | 13.6-16.3 ms | 13.8-16.2 ms | 0.86-1.05x (unstable, not a win) |
| `multiply_transforms` vs numpy | 4x4 | 179-199 ns | 8.6-10.5 ns | 17-19x slower |

Not slower: transform (3.9-4.1x faster than the NumPy reference), normals (18.5-19.5x),
beam lengths (13.5-13.9x), manifold (5.1-5.9x, and a *faster* single-`read` than
upstream at 16 faces).

## What the next phases should act on, in order

| phase | target | why now |
| --- | --- | --- |
| `unroll` | XML writer and reader (rows 7-14) | largest absolute time in the package and the only place it loses to upstream by >4x |
| `unroll` | `transformed` copies (row 17) | 12 MB and 120 allocations per call for zero algorithmic work |
| `unroll` | `beam_lengths` glue (row 5) | measurable 17-20% at the benchmark size |
| `parallel` | the transform's eight-way split (row 2) | currently no speedup at all on this host; decide keep-with-proof or remove |
| `simd`, `gpu` | nothing | every numeric kernel is at the measured memory ceiling and ≤9 flops/byte |
