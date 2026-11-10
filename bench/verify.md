# VERIFY: does the README reproduce?

Session: 2026-11-10, this machine only (Intel Xeon E5-2697 v4 @ 2.30 GHz, 2 sockets,
72 logical, Linux 6.8.0-147-generic). Toolchain as locked in `pixi.lock`
(`mojo ==1.2.0.dev2026092605`, `max ==26.7.0.dev2026092605`, Python 3.13.14,
`lib3mf ==2.5.0`). Nothing in `src/` was changed; `git diff -- src/` is empty.

Every number below was produced in this session. The host was shared throughout and
its load average ranged from 30 to 42 during the session (the README it inherited was
recorded at 15-19), which matters for the parallel rows and is reported with them.

## Gate

| command | result |
| --- | --- |
| `pixi run build && pixi run test` | build OK (deprecation warnings only, no errors); `53 passed in 8.46s`, exit 0 |
| `pixi run bench` | exit 0 |
| final `pixi run build && pixi run test` | build exit 0 with 0 `error:` lines, `53 passed in 8.22s`, exit 0 |

## The README benchmark table

Seven `pixi run bench` invocations were made. Each invocation is its own best of four
calls per row (three for manifold), exactly as the bench defines it. Raw rows, in
order:

| run | transform 1M | 4M par / serial | normals 250k | lengths 1M | manifold 20k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.284 / 12.223 | 10.158 / 15.318 | 2.212 / 40.757 | 4.506 / 63.954 | 2.821 / 15.804 |
| 2 | 3.593 / 17.428 | 23.066 / 15.006 | 2.467 / 43.766 | 5.355 / 67.253 | 3.092 / 17.916 |
| 3 | 5.193 / 15.680 | 9.939 / 15.541 | 2.492 / 42.878 | 7.120 / 66.709 | 3.005 / 17.527 |
| 4 | 8.242 / 15.345 | 11.504 / 22.493 | 2.452 / 41.961 | 5.223 / 66.038 | 2.660 / 15.369 |
| 5 | 3.507 / 14.861 | 10.301 / 15.734 | 2.419 / 42.727 | 5.216 / 68.072 | 2.977 / 15.915 |
| 6 | 3.386 / 13.679 | 10.604 / 15.211 | 2.657 / 43.564 | 5.143 / 69.906 | 2.763 / 15.545 |
| 7 | 3.489 / 14.313 | 9.549 / 15.482 | 2.427 / 42.927 | 5.247 / 69.071 | 2.750 / 16.120 |

Times in ms, mojo-lib3mf / reference. Run 4's 8.242 ms for the 1M transform and run
2's 23.066 ms for the 4M parallel kernel are co-tenant interference (the reference
side inflated in the same runs); they are not reproducible and are kept here so the
reader sees the spread rather than a curated minimum.

Best of session against the inherited README figure:

| kernel | measured best | inherited | delta (mojo) | measured speedup | inherited speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| transform 1M vertices | 3.284 / 12.223 | 3.511 / 13.508 | -6.5% | 3.72x | 3.85x |
| transform kernel 4M vertices | 9.549 / 15.006 | 10.622 / 16.287 | -10.1% | 1.57x | 1.53x |
| normals 250k triangles | 2.212 / 40.757 | 2.443 / 42.230 | -9.5% | 18.43x | 17.29x |
| lengths 1M beams | 4.506 / 63.954 | 5.085 / 67.814 | -11.4% | 14.19x | 13.34x |
| manifold 20k triangles | 2.660 / 15.369 | 2.737 / 15.839 | -2.8% | 5.78x | 5.79x |

Every cell of the inherited table reproduced within 20% (worst 11.4%), so no cell
was out of tolerance. The table in `README.md` was nevertheless rewritten (see
"Changes") because it now quotes only numbers measured here.

No row became slower than its reference, so no row moved in the table. The 4M row
*is* slower than its serial reference in 1 of 7 individual runs (0.65x) and faster
in the other 6 (1.43x-1.96x); that instability is real and is now stated in the
README instead of being hidden behind the best case.

## Same input, no skipped work

`bench/bench.py` gives the two sides the same arrays for transform, normals, and
beam lengths, and the same geometry for manifold. Checked by installing spies that
record the pointer each side receives:

| row | check | result |
| --- | --- | --- |
| transform 1M | port kernel input ptr == NumPy reference input ptr | same (`129475295223824`) |
| transform 1M | port matrix ptr == reference matrix ptr | same (`105842009872992`) |
| normals 250k | vertices and triangles ptrs | same, same |
| lengths 1M | vertices and beams ptrs | same, same |
| manifold 20k | identical `float32` vertices and `int32` faces passed to `m3.Mesh` and `lib3mf.SetGeometry` | same arrays |

Result correctness on those arrays (port vs the reference the bench times):

| kernel | max abs difference | flags |
| --- | ---: | --- |
| transform | 9.54e-07 | - |
| normals | 4.95e-06 | - |
| beam lengths | 4.77e-07 | too-short flags identical |
| manifold | - | port `True`, upstream `True` |

Skipped work: none found on the port side. The port's timed public paths do *more*
than the reference, not less: `beam_lengths()` runs `_indices_in_range()`
(`beams.min()` / `beams.max()` over the whole 1M-row index array) inside the timed
call, and `transformed()` allocates the result, copies the triangle array and
`deepcopy`s the beam lattice inside the timed call, while `numpy_beam_lengths` and
`numpy_transform` do neither. Those extras are charged to the port, so the quoted
speedups are conservative with respect to that asymmetry.

The 4M kernel row is the only row whose two sides are two Mojo kernels rather than
port vs. reference (`m3mf_transform_vertices_f32` vs `_serial` on preallocated
buffers). It is labeled as such in the table and is not an upstream comparison.

## The other numeric tables

### Parallelism

These come from `bench/parallel_probe.py`, which is not run by `pixi run bench`.
It was run three times with the README's documented command
(`--workers 1,2,4,8 --repeat 15`), plus an ad-hoc sweep of 2.5M and 3.5M vertices
(the probe's size list stops at 1.5M then jumps to 2M/3M/4M/8M/16M). Cells that
moved by more than 20%:

| table | cell | inherited | measured (3 sweeps) |
| --- | --- | ---: | ---: |
| beams | 1,000,000 4w | 0.81x | 0.74 / 0.58 / 0.56x |
| beams | 1,000,000 8w | 0.92x | 0.78 / 0.64 / 0.59x |
| beams | 1,500,000 4w | 0.98x | 0.92 / 0.73 / 0.91x |
| beams | 1,500,000 8w | 1.00x | 0.97 / 0.79 / 0.83x |
| beams | 3,000,000 2w | 0.97x | 1.14 / 1.19 / 1.08x |
| beams | 3,000,000 8w | 1.22x | 1.53 / 1.38 / 1.48x |
| transform | 2,500,000 4w | 1.22x | 0.98x (one sweep) |
| transform | 3,500,000 8w | 1.53x | 1.24x (one sweep) |

Prose claims that did not reproduce:

| claim | inherited | measured |
| --- | --- | --- |
| transform at 100k-1M, parallel | "0.29-0.64x" | 0.04-0.67x; 0.04x at 100k (~25x slower) |
| transform scaling at 16M | "2.59x on 4, 2.85x on 8" | 2.28-2.46x on 4, 2.50-2.88x on 8 |
| 3M beams, pool warmed | "0.67x of serial" | 1.37-1.45x on 4w, 1.38-1.53x on 8w |

The gate sizes themselves (4,000,000 vertices, 3,000,000 beams) are still supported
by the measurements: the transform loses on every sweep below 2M and wins on 4
workers at 4M in all three sweeps (1.43-1.54x); beam lengths lose on every sweep
at 1-1.5M (0.56-0.97x) and win at 3M in every sweep (1.08-1.53x; 4 workers
1.37-1.45x). At 2M beams the result straddles 1.0x (0.86-1.15x), which is why the
gate sits above it. No gate was changed.

The two "Before and after" tables (the parallel phase's and the SIMD phase's) could
not be reproduced: their "before" columns came from code that is not in this commit.
Their "after"/current columns fall inside the ranges measured here (e.g. 4M
parallel 10.622 ms inherited vs 9.549-23.066 ms measured). Both tables were removed
from the README and replaced with this session's ranges; the design conclusions
they supported (replace the fake serial chunking, retain no SIMD candidate) are
unchanged and are still stated.

### Not measurements

The bandwidth figure "~8 GB/s single-threaded" inherited from `bench/analysis.md` was
re-measured here: NumPy `add` over two 32 MB `float32` arrays, best of 20, 8.72 GB/s,
consistent with it.

## Tests and parity

`git log -p -- tests/` shows four commits touching `tests/`:
`4308ca8` (creation), `e14dd01` (added `test_transform_vertices_small_remainders` and
`test_transform_vertices_parallel_threshold`), `09e3896` (added `upstream_fixtures.py`
and `upstream_oracle.py`), `72fb9d5` (added `test_parallel.py`). Every change was an
addition; `git diff e14dd01 HEAD -- tests/test_kernels.py tests/test_io.py` is empty.
No assertion was removed and no tolerance was loosened by an earlier phase, so
nothing needed restoring on that account.

Current parity coverage, checked by reading each test:

| test file | compares against upstream | how |
| --- | --- | --- |
| `test_io.py` | yes | both read directions with the `lib3mf` 2.5.0 binding, beam lattice and balls included |
| `test_kernels.py` | partly | manifold vs the binding; the other kernels vs NumPy transcriptions of upstream source, since the binding does not expose them |
| `test_parallel.py` | no | parallel path vs the serial kernel of the same library, bit-identical, at gate-1 / gate / gate+1 |

Problem found: `tests/upstream_fixtures.py` documents `test_upstream_fixture.py`,
which "cross-checks each excerpt against `raw.githubusercontent.com` and executes the
arithmetic ones through `python_transcription`". That file does not exist, and no
test imports `upstream_fixtures.py` or `upstream_oracle.py`
(`git log --all -- tests/test_upstream_fixture.py` is empty). So the 828-line upstream
excerpt corpus and the 208-line oracle helper are dead weight: nothing is asserted
over them. This is weaker than the README's Correctness section implied. It was not
an earlier phase's regression of an existing test, so no assertion could be restored;
the docstrings and README were corrected to say what is actually executed, and the
gap is recorded here. Writing that missing test is out of scope for a verification
pass and would be a new parity test, not a restored one.

`test_parallel.py`'s beam-length comparison uses `rtol=1e-6, atol=1e-6`, the same
tolerance as `test_kernels.py`; that is appropriate for a `sqrt(float32)` value
promoted to `float64` and is not a loosening of anything that existed before.

## Changes made

Code: none. No failing test was written and none was needed for the numbers above;
`git diff -- src/` is empty.

1. `README.md`, benchmark table: replaced with this session's best-of-seven values.
   The inherited values all reproduced within 11.4%, so this is a freshness change,
   not a correction. The instability of the parallel rows is now stated.
2. `README.md`, Parallelism: both tables replaced with measured ranges over three
   sweeps (and the two sizes the probe does not sweep, measured separately); the
   prose ranges corrected to match, including the "0.29-0.64x" claim, which was
   wrong by a factor of about 7 at 100k vertices.
3. `README.md`, Parallelism: the 3M-beam "0.67x" caveat replaced with what this
   session measured, keeping the warning that the beam kernel loses below 2M beams.
4. `README.md`, Parallelism and SIMD sections: the two before/after tables removed,
   since their "before" columns are from code not in this commit; replaced with
   measured ranges and a statement of provenance.
5. `README.md`, Correctness: added the fact that the upstream-fixture corpus is not
   executed, and the test count/time.
6. `tests/upstream_fixtures.py`, `tests/upstream_oracle.py`: docstrings corrected to
   stop claiming a consumer test that does not exist. No test logic changed.

## Not verified

- Anything about a different machine. These are one host's numbers.
- The parallel rows' stability. Under load average 30-42 the 4M transform measured
  anywhere from 0.65x to 1.62x of its serial kernel. A single `pixi run bench` is
  not enough to decide whether the parallel gate pays on a target machine; three
  probe sweeps are.
- The upstream C++ excerpts in `tests/upstream_fixtures.py` were not fetched from
  `raw.githubusercontent.com` and not executed; this session has no network check of
  them and no test does either.
- `bench/analysis.md` (the I/O-layer timings) was not re-run; it is a previous
  phase's record and is not a speedup claim in the README.
