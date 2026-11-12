# STC selected-target MP2

See the [root-owned DF/SCF optimization guide](../../README_SERIAL_DF_SCF.md) for
shared-buffer activation, preconditioning, thread settings and staged OOC use.

STC owns Boys selection, exact-frame grouping, weak-pair correction, restart
serialization, three-tensor preparation and molecular response. Production
modules and examples do not import native DLNO. Saved orbital records and
`frag_lolist` retain a complete singleton partition. `boys_grouping` separately
chooses physical CC fragments: `singleton` uses one original active occupied
Boys orbital; `atom` uses the existing Löwdin assignment and hydrogen-to-heavy
atom groups; `molecule` puts all active occupied Boys orbitals in one physical
fragment and automatically enables full molecular domains.
`DomainSelections.physical_groups` stores ordered orbital-record
membership, and its position is the stable physical fragment ID. Internal
`DomainGroup` entries merge preparation only when saved supports, candidate
ordering, retained virtual ranks and PAO anchors are identical.

The CCSD entry points `dlno_stc.ccsd.kernel` and `dlno_stc.ccsd.value_and_grad`
default to `lis_aux_svd_tol=1e-5`. Each physical fragment compresses its LIS
auxiliary factors by retaining squared singular values `s**2 > lis_aux_svd_tol`.
Pass `lis_aux_svd_tol=None` or `0` to disable compression. This applies to all
physical grouping modes; domain MP2, D/W matrices and orbital selection retain
their existing auxiliary spaces.

Compression uses NumPy's reduced SVD (`full_matrices=False`). Its NumPy/SciPy
forward algebra honors `PYSCFAD_DENSE_BLAS_THREADS` through a temporary BLAS
scope, restoring the
previous limits on exit. This does not configure JAX/XLA threads or change the
custom reverse.

```python
from pyscfad import dlno_stc
static = dlno_stc.build_domain_selections(mf, frozen=4)
energy = dlno_stc.kernel(mf, static, scope="domain", controls={"mode": "deterministic"})
energy, gradient = dlno_stc.value_and_grad(
    mol, build_mf, build_static=lambda mf: dlno_stc.build_domain_selections(mf, frozen=4),
    scope="domain", controls={"mode": "deterministic"},
    include_hf=True,
)
```

For gradients, ``build_static`` selects domains from the result of one SCF
performed under ``jax.vjp``. The driver retains that SCF pullback for the final
nuclear response, so no preliminary SCF is needed. The callback runs outside
differentiation on a detached reference, protecting the live SCF object's
structure from selection-only DF caches. Alternatively, pass existing
``DomainSelections`` as ``static``; do not supply both. System gradients need
only ``mol`` and ``build_mf`` with ``scope="system"`` and a frozen-space setting.

Pass `correlation_auxbasis="cc-pvdz-ri"` to use RIFIT for correlation while
`build_mf` (or the supplied `mf`) continues to use its existing JKFIT setup:

```python
energy, gradient = dlno_stc.value_and_grad(
    mol, build_mf,
    build_static=lambda mf: dlno_stc.build_domain_selections(mf, frozen=4),
    controls={"mode": "deterministic"}, include_hf=True,
    correlation_auxbasis="cc-pvdz-ri",
)
```

The same argument is accepted by MP2 and CCSD `kernel`/`value_and_grad`,
domain/LIS selection builders, and MP2 export/replay helpers. Its default,
`None`, preserves the existing SCF fitting configuration and workflow.
Callbacks invoked through `build_static` inherit the run's correlation fit;
when selecting or exporting separately, pass the same argument at each step.
Exact pair screening and CC D/W selections record the fitting basis and
reject reuse with a different fit. Multipole screening remains independent
of the correlation fit. Restart and packet identities include the override.

With an explicit override, each domain constructs its own auxiliary molecule,
metric and factors in the chosen basis. Local factors are not slices of a
global fitted tensor. HF, AO Fock and implicit SCF response continue using
`mf.with_df`, including its existing global JKFIT file. Correlation coordinate
response uses the local RIFIT metric/three-center derivatives; its orbital
and Fock response still closes through the single JKFIT SCF pullback.
The storage policies below are unchanged: this option does not create a
global RIFIT disk file. A distinct system fit uses all atoms in the direct
producer and retains the transformed correlation factors in RAM. An explicit
basis identical to the SCF configuration reuses compatible cached SCF factors
in system preparation and exact pair screening. Memory
estimates use the chosen correlation auxiliary dimension; there is no new
disk fallback for an oversized local factor, metric, H cache or adjoint.

Standalone domain MP2 and direct system MP2 retain one auxiliary metric,
Cholesky factor and derivative tape through each preparation/pullback pair,
including MPI owner jobs and saved-packet replay. Positive definiteness is
checked by the retained factorization, without a separate validation build.
Cached system factors use a dimension check to reject rank-reduced fitting
sources. Domain MP2 now uses the same `half`/`direct` cache setting as CCSD(T),
including MPI owner jobs and saved-packet replay. The default `direct` mode
regenerates local integrals without retaining H. Set
`PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE=half` to retain H through the coefficient
pullback, reducing ordinary three-center integral generation from two passes
to one per energy/gradient group. H is released before the deferred geometry
contraction. System MP2 retains its existing cached-factor/direct
streaming policy without an H cache. Local integral response finishes before
common/Fock/SCF response in both modes.

`mp2.imported_value_and_grad` uses the production driver's single-SCF
initialization, final overlap/Fock outputs, common response and final SCF
response scopes. It also shares global DF geometry work and releases completed
tapes in the same order. Packet identities, effective sampling budgets and
partial-import HF/weak restrictions are unchanged. The portable packet format
does not require a scalar standard error; replay logs unavailable uncertainty
as NaN rather than treating it as zero.

`python examples/dlno_stc/validate_water4_mp2.py --xyz-file water4.xyz --output results/water4`
compares direct, half and saved-packet gradients with separate JKFIT/RIFIT,
checks integral-pass receipts, and validates a fixed-selection directional
finite difference. It defaults to cc-pVDZ and retains results and packets in
the new output directory. It uses a Boys gradient tolerance of `1e-9` so
localization convergence error does not dominate the finite difference;
production defaults are unchanged. Set numerical thread limits before launch.

Nuclear gradients require coordinate-only molecule leaves
(`mol.build(trace_exp=False, trace_ctr_coeff=False)`). Selections, domains,
Laplace grid, retained ranks and PAO anchors remain fixed through response;
orbitals, Fock blocks, RI factors, Boys localization and SCF are differentiated.
Finite domains and system scope project saved anchors into their retained
virtual spaces and apply metric Cholesky orthonormalization. Pivoted QR selects
the discrete anchors once during concrete setup; preparation reuses them.
Weak corrections
retain the original singleton topology. `scope="system"` retains the complete
active-space energy and system sampling policy. Its independent controls and
canonical specialization remain available through the low-level full solver.

Local-DF stripes use the smaller of `mf.with_df.max_memory` and half the
remaining process allowance under `mf.mol.max_memory`, after reserving outputs
that are about to be allocated. There is no separate fixed stripe cap. These
are workspace estimates; indivisible auxiliary shells and runtime allocations
mean they are not a hard process RSS limit. Retained H tensors have their own
admission check and remain in RAM.

SCF disk storage contains fitted `j3c` factors only. Raw `(P|mu nu)` inputs
are generated locally in bounded shell tiles and are never cached on disk.
Fragment DF has two modes, selected by
`PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE` (benchmark: `--fragment-df-mode`):

- `half` (opt-in, option 2): one integral-generation pass per domain group.
  Retain `H_ED[P,mu,p] = sum_nu (P|mu nu) C_ED[nu,p]` and fitted `B_ED,ov`.
  Each LIS lies in its occupied/virtual ED parent span, so `H_LIS = H_ED U`.
  Its fitted oo/ov/vv blocks and both coefficient-response legs use this H.
  The current LIS H/output is released after its correction; ED H and MP2
  factors remain live through the W/MP2 pullback. No `A_ED` is retained.
- `direct` (default, option 1): regenerate integrals for each ED/LIS forward transform
  and coefficient pullback. One ED and one LIS require four passes for an
  energy plus gradient; a group with `k` LIS solves requires `2*(1+k)`.
  Forward-only execution needs `1+k` passes.

These counts refer to ordinary three-center inputs. Gradient runs still
compute derivative integrals. Both modes cache one metric, Cholesky factor
and response tape per group, and combine all local three-center/metric
coordinate responses into one contraction. ED/LIS coefficient pullbacks
remain separate and feed the W/D and orbital responses normally. This works
with deterministic and stochastic triples, independently of their integral
rotation option. Use `single_pass=True` for W to avoid a selection prepass.

For `m` orbital columns, H uses `8*naux*nao*m` bytes; overlapping left/right
slices are stored only once. The ED cache and current LIS cache coexist.
The resident-memory estimate uses `mol.max_memory` (`--max-memory-mb`), current
RSS, and the optional advisory cache limit `PYSCFAD_LNO_HALF_DF_CACHE_MB` (decimal MB).
It includes output-conversion space and budgets half the current RSS headroom
for subsequent work. Exceeding this headroom estimate or the cache limit emits
a `RuntimeWarning` and continues with resident allocation.
`mf.with_df.max_memory` (`--df-memory-mb`) limits
workspace, not resident H storage. Generation and reverse contractions use
bounded tiles under the workspace allowance, RSS headroom and local integral
tile target. Insufficient allowance for even one processing tile still raises
`MemoryError`, and actual allocation failures propagate. Select `direct` for
lower resident input memory. These
allocation budgets do not impose a hard RSS cap on the entire CC solver.

Deferred ED/LIS MO integral adjoints always remain in RAM until the domain's
combined derivative contraction finishes; there is no local adjoint disk
fallback. ED occupied-virtual seeds retain their original storage. LIS seeds
use packed symmetry for whitening and metric products, then expand once to
full symmetric orbital pairs shared by the coefficient and geometry pullbacks.
Their storage is `8*naux*nocc_ED*nvir_ED` and `8*naux*nmo_LIS**2` bytes,
respectively, plus coefficient snapshots. All domain-group seeds are released
after its derivative contraction or on failure. Resident adjoints add to the
memory occupied by H and other solver data; contraction scratch remains tiled.
Neither raw AO inputs nor H tensors are spilled. Both modes retain the
packed-density optimization below. Receipts report `adjoint_resident_bytes`,
`adjoint_peak_bytes` (including coefficient snapshots), `symmetric_expanded_bytes`,
`cache_mode`, `raw_integral_builds`,
`raw_source_passes`, `half_builds`, `half_rotations`, `half_cache_hits`,
`half_pullbacks`, `half_releases`, `half_resident_bytes` and `half_peak_bytes`.
The raw-storage helpers/switches, projected cache, and option 3's `A_ED` cache
and combined ED-pair adjoint have been removed.

Production domain setup builds one live AO Fock matrix in `common.fock`.
ED and LIS Fock blocks are orbital projections of this resident matrix;
fragments do not load an AO Fock matrix from disk or rebuild AO J/K for each
projection. The initial shared Fock build may read fitted SCF DF factors.
Fragment Fock cotangents accumulate before the common AO-Fock pullback.

Serial and root-owned MPI `value_and_grad` calls also combine the global
AO-Fock and implicit SCF DF geometry responses when the out-of-core DF source
is built or attached
inside the traced SCF builder. A producer boundary returns the summed seed
through the original AO/auxiliary geometry maps. Density and orbital response
remain immediate; compatible coordinate seeds share one derivative-integral
pass and one metric/Cholesky tape. This works with either fragment DF mode.
`Shared global SCF DF reverse` reports consumers, seed groups, disk
contributions, and geometry/metric passes. Set `PYSCFAD_STC_COMBINE_SCF_DF=0`
before tracing for the separate-response reference path. With explicit
`parallel_scf_jk=False`, MPI fragment owners reduce completed cotangents to
root, which closes the combined shared response once. Explicit distributed
SCF and externally staged DF sources retain their existing ownership rules.

Local DF geometry responses fill the derivative contractor's packed AO-pair
buffer directly, including the correct diagonal weights. They avoid a full
AO-square symmetric-sum array, and shared consumers accumulate in packed
space. `PYSCFAD_DF_MO_DENSITY_BACKEND=auto` selects an AO-tiled native
projection for at least 512 AOs and orbital contraction rank at most 128;
Larger orbital ranks use batched GEMM followed by direct symmetric packing.
The tiled projection uses up to 16 OpenMP workers, each with a 128-by-128
tile, and serial BLAS inside that parallel region. Its scratch is reused
across auxiliary blocks and capped by the caller's existing conservative
density-workspace reservation. Auxiliary reads remain bounded; neither raw
integrals nor the complete packed AO-adjoint tensor are retained by packing.
Set the backend to `gemm`, `tiled`, or `legacy` for controlled comparisons.
An older `pyscfadlib` without the tiled symbol falls back to GEMM packing;
rebuild that library to enable tiling. The derivative-integral kernel and
ED/LIS cache choices are independent of this option.
`examples/dlno_stc/benchmark_df_density_packing.py` compares the original,
GEMM-packed and automatic paths on representative water-32 dimensions.

Within that root-owned response, occupied-factor caching is enabled by default.
Its shared allowance with the preconditioner uses half the current RSS
headroom under `mol.max_memory` (or explicit `precond_total_memory`). Global
geometry panels use the same headroom policy after the root cache is freed.
Explicit `PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB`,
`PYSCFAD_PRECOND_CACHE_MEMORY_MB`, and `PYSCFAD_FACTOR_PANEL_MEMORY_MB` values
remain upper bounds in MiB; `PYSCFAD_SCF_OCCUPIED_CACHE=0` disables the occupied
panel. This removes the implicit 1 GiB/256 MiB cache ceilings from this path
without changing the user's total memory allowance or DF workspace limit.
`SCF response factor cache` receipts report reservations, hits, and rejected
allocations for the completed root solve.

The DF-CCSD response retains update intermediates from its single converged-
amplitude linearization through the final integral adjoint, eliminating a
second forward update replay. Smooth (T) similarly retains orbital-rotation
prefixes for the current Laplace node and releases them in its pullback.
Their allowance is half current RSS headroom under the CC solver's total
memory budget; `PYSCFAD_TRIPLES_DRESSING_CACHE_MB` optionally lowers it (MiB,
zero disables caching). Insufficient headroom selects bounded rematerialization.
Fixed-count stochastic runs prepare all pilots first and reconstruct each
current node's prefixes once when needed; discarded pilots compute no VJP.
Diagnostics report `dressing_cache_peak_bytes`, `dressing_cache_hits`, and
`dressing_cache_rebuilds`. Cached and uncached paths have the same energies,
sampling streams, and full Fock/orbital response.

Domain selection and the domain driver lazily enable persistent JAX compilation
caching in `$XDG_CACHE_HOME/pyscfad/jax`, falling back to `~/.cache/pyscfad/jax`.
The automatic policy retains small kernels for reuse in subsequent processes.
MPI launcher ranks use separate `mpi-rank-N` subdirectories.
Numerical endpoint values and response tapes are rebuilt for each calculation.
This JAX setting is process-wide. Set `JAX_ENABLE_COMPILATION_CACHE=false` to opt
out. An explicit JAX cache directory retains its existing threshold policy;
a custom cache needs `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` and
`JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1` to retain all small kernels.
An unavailable automatic cache warns and allows the calculation to continue.

Weak-pair kernels pad each endpoint's virtual dimension to a multiple of 128
by default. Set `PYSCFAD_WEAK_PAIR_BUCKET_SIZE` to a positive integer before
starting a calculation to override this width in both serial and MPI paths,
for example `export PYSCFAD_WEAK_PAIR_BUCKET_SIZE=64`. Use the same setting on
every MPI rank and keep it fixed throughout the calculation. Larger buckets
reduce the number of compiled pair shapes at the cost of more padded work;
the padding does not change the energy or its response. Explicit
`pad_endpoint(..., bucket_size=...)` arguments override the environment.

Weak response builds each orbital screen once under `jax.vjp`, retaining the
returned screen and its orbital-algebra pullback. Compact transition moments
are shared by all incident pairs. Response replays only AO moments using that
exact screen, then applies the retained orbital pullback. Orbital and energy
response remain differentiated; there is no eigensolver reconstruction or
orbital-sign matching during replay.

For gradient calls using `build_static`, serial MP2 and the CCSD driver now
prepare these endpoints during domain screening. Weak energy and cotangents are
completed after atom-group expansion of the strong mask, before strong-frame
selection, and consumed once at the normal weak-completion boundary. The same
live common-data frame and its pullback are adopted by initialization. Raw AO
moments are shared within each identical ordered primary atom support during
screening. Reverse replay accumulates raw AO-operator cotangents across that
support before applying one shared molecular integral pullback. Each endpoint
still applies its own saved orbital pullback. All direct molecular dependence
of the current endpoint is through those raw AO moments; sharing their response
does not freeze integral or orbital derivatives. No screening threshold or
weak-energy formula changes.

PAO preprojection is also shared across compatible weak endpoints. Concrete
rank selection reuses the parent/support/representation algebra within that
selection call. Differentiable screening shares the same algebra only for
identical ordered parent columns, support and representation AOs, and retained
canonical/overlap/completeness labels. Each target still has its own occupied
projection, final metric rank and virtual semicanonicalization. Reverse mode
sums preprojection cotangents and applies one shared pullback as soon as its
last active endpoint contributes; unused groups release their tapes without
reverse work. Detached selection values never replace the live PAO response.
This changes no BLAS/OpenMP defaults and does not distribute CCSD's weak work
across MPI ranks.

Endpoint PAO construction uses shared reference products `Y = S @ P`,
`G = P.conj().T @ Y`, and `S @ occupied_targets`. Fresh gradient selection
localizes the Boys orbitals once and attaches their checked implicit response
in the reference-aligned rotation chart; it does not run another localization
optimizer. The stationarity check uses the requested tolerance. An unsuitable
frame falls back to ordinary refinement, preserving the existing saddle and
response checks. The PAOs, their normalization products, and the screening
products are built once in the live reference. Concrete selection reads
detached views of those same values. A primary domain
then selects blocks of these products for its metric eigensolves, normalization
and projection; it no longer repeats full-AO products for each endpoint or
preprojection group. Full ordered AO representations retain the direct PAO
identity projection. Standalone fallback calls form products only for their
requested parent columns. Forward-only serial screening and energy sums covering
every endpoint use the same sharing without retaining pullbacks. Single-pair
and partial-endpoint energy queries retain selected-parent preparation so a
small query never triggers an all-system product cache.

Live response accumulates all product cotangents before applying one shared
product pullback. Cached products use the exact transpose of their linear
differential, without recomputing the forward products. The overlap, PAO and independent occupied-target derivatives
are preserved, including directions away from the physical reference. This
removes repeated global arithmetic, not the global reference or its response:
shared products and accumulated cotangents still occupy global-sized storage.
Static products remain available through weak response for subsequent strong
selection, and local pullbacks can temporarily produce global-sized bars.

This reuse is transient: no response tape is serialized into static selections
or checkpoints. Saved-static callers, explicitly supplied target frames, and
callbacks replacing the selected reference retain the ordinary preparation
path. Selection callbacks that change the reference also rebuild its S/F data.
`include_baseline=False` does not prepare weak response. Resumed calls
defer weak work until checkpoint progress is restored, so completed weak terms
are not recomputed. Distributed MP2
retains its existing distributed endpoint executor; CCSD MPI computes and
consumes the reused weak contribution once on root. A completed checkpoint
weak term discards any newly prepared contribution instead of adding it twice.

The inclusive `stc.domain.selection` timer now contains common preparation and
weak response. `stc.domain.selection_weak_response` identifies the nested weak
work. Use the interval-partitioned timing report or subtract this marker before
adding separately displayed weak phases; summing inclusive timers double-counts
work. Pending numerical weak cotangents remain resident until consumption, but
endpoint tapes are released before strong-frame construction.

Only one AO-moment derivative tape lives at a time. Orbital pullbacks retain
local algebra residuals until consumed; their memory can exceed the saved
orbital coefficients alone. Endpoint values, cotangents and both pullbacks
are released progressively, with one explicit garbage collection after the
weak stage. Automatic collection and strong-group cleanup remain active.
The AO-moment consistency check compares each moment order's Cartesian RMS
per virtual column with `atol=rtol=1e-9`; centers and energy gaps retain
componentwise checks.

## Fixed or molecule-fitted Laplace grids

`pyscfad.dlno_stc.laplace.make_laplace_grid` returns independent float64 roots,
weights, and a JSON-compatible diagnostics dictionary. The default remains the
original six-point grid, including its factor of 2.6. Fitting parameters are
ignored for `kind="default"`. `kind="fitted"` fits positive nodes and weights to
the active canonical orbital-energy distribution; fitted values have no extra
2.6 scaling. Fit once on the controlling rank, after SCF and outside JAX tracing:

```python
import numpy as np
from pyscfad.dlno_stc.laplace import make_laplace_grid

laplace_grid = "default"        # default | fitted
laplace_points = 6              # fitted only
laplace_lambda_gradient = 0.1   # fitted only

# mf is the converged restricted DF-SCF reference. These saved full-MO
# indices already exclude frozen occupied and virtual orbitals.
static = dlno_stc.build_domain_selections(mf, frozen=4)
energies = np.asarray(mf.mo_energy)
roots, weights, diagnostics = make_laplace_grid(
    energies[static.active_occ_indices], energies[static.active_vir_indices],
    kind=laplace_grid, points=laplace_points,
    lambda_gradient=laplace_lambda_gradient,
)
grid = dict(laplace_roots=roots, laplace_weights=weights)
controls = dict(mode="deterministic", **grid)
energy = dlno_stc.kernel(mf, static, controls=controls)
system_energy = dlno_stc.kernel(mf, scope="system", frozen=4, controls=controls)
```

Grid choice is independent of deterministic/stochastic solver mode. Existing
callers can continue passing explicit arrays. Pass only the resolved arrays to
solver controls; `laplace_grid`, `laplace_points`, and
`laplace_lambda_gradient` are Python driver options. Arrays remain part of the
existing restart and replay fingerprints. Save them and the diagnostics with
the run, and reuse exactly those arrays for gradients and displaced geometries.
Do not refit inside `build_mf`, a response callback, or individual domains.

For CCSD, supply the same arrays to both controls **before LIS selection**:

```python
from pyscfad.dlno_stc import ccsd

baseline_controls = dict(mode="deterministic", **grid)
matrix_controls = dict(mode="deterministic", **grid)
selected = ccsd.build_static_selections(
    mf, static, matrix_kind="D", thresh_occ=1e-6, thresh_vir=1e-6,
    baseline_controls=baseline_controls, matrix_controls=matrix_controls,
    single_pass=False,
)
energy = ccsd.kernel(mf, selected, baseline_controls=baseline_controls,
                     matrix_controls=matrix_controls)
```

For `matrix_kind="W"`, the CCSD driver uses the energy returned by W
construction for the strong-domain MP2 term. It does not run a separate MP2
energy/gradient solver. With the current normalization,
`E = (trace(Wo) + trace(Wv))/2`; deterministically both traces equal E.
Stochastic E and W use the same production samples, with covariance included
in the reported MP2 energy SE. After the LIS CCSD(T) response, one matrix
pullback combines `energy_bar=1` with the occupied/virtual W cotangents. Direct
Fock and orbital cotangents from the LIS response are accumulated through their
own preparation paths. The extended-domain occupied/virtual frame is built
once per preparation group and shared by the domain DF/Fock and LIS branches.
Preparation exposes both its three input tensors and the frame coefficients
as differentiable outputs. Direct LIS frame cotangents are summed over the
group's physical fragments and seeded alongside the accumulated DF/Fock bars;
one preparation VJP combines them before reversing the frame once. Only the
current group's frame, derivative intermediates and coefficient bars are kept.
Selection also reuses the frame returned by domain preparation, while its
discrete selection pass remains separate from the differentiable evaluation.
The per-fragment `direct_host_bar_norm` reports direct host bars with this
shared frame held fixed; shared frame response is included later in the total
gradient. No energy derivative is computed during W sampling.
Weak-pair MP2 terms are still added separately, once. `include_baseline=False`
sets the W energy seed to zero and returns only the selected CC corrections.
For W, `matrix_controls` determines strong-domain energy precision;
`baseline_controls` remains part of preparation configuration, and the two
Laplace grids must match. D calculations retain their independent MP2 solver.
Rebuild saved CC selections and start a new checkpoint after this merge;
selection schema 9 requires rebuilding earlier LIS selections and checkpoints.

The fit uses 600 single-gap points and three positive least-squares starts.
Diagnostics include fitting seconds, exact molecular denominator bounds,
histogram-weighted energy RMS, Delta-scaled derivative RMS (without lambda),
and maximum relative errors for `1/Delta` and `1/Delta**2` sampled at 10001
equally spaced denominators. `make_laplace_grid()` needs no spectra for the
default grid; numerical diagnostics then contain `None` and fitting time is
zero. Fitted mode rejects invalid spectra, nonpositive gaps, invalid fit
parameters, and failed fits without falling back to the default.

Gradients differentiate the energy with roots and weights held fixed. They do
not differentiate the optimizer or a refitted grid as geometry changes. The
diagnostics measure quadrature errors, not MP2 energy/force error bounds;
stochastic standard errors exclude quadrature bias. For local coverage use
`laplace_grid_diagnostics` with `eigvalsh(foo)` and `eigvalsh(fvv)`, rather than
local diagonal entries. A molecular fit does not guarantee local accuracy.

The workstation acceptance driver is
`TEST_general/dlno_stc_test/fitted_LT_test/test_water4_lt_fit.py` in the parent
project. It compares both choices using one central SCF, cached complete-domain
tensors, and two displaced SCFs. Its `--laplace-grid`, `--laplace-points`, and
`--laplace-lambda-gradient` options also configure the optional
`--ccsd-matrix-kind D|W` calculation before LIS selection. It saves grids,
diagnostics, gradients, matrix seeds, and integration checks in its results
directory; no C++ changes or reference-script runtime imports are required.

## MPI across domains

For `scope="domain"` and a communicator with more than one rank, complete
group jobs are distributed round-robin. Each owner prepares its three tensors,
runs the local STC solver with OpenMP, and applies its saved preparation VJP.
Weak endpoints are prepared and replayed once globally; pair work is separately
partitioned. Cotangents are summed, and communicator rank zero drives the common
and SCF response once, including HF once when requested. Only rank zero returns
the public energy or `(energy, mol_bar)`; other ranks return `None`.

Multi-rank domain calls preserve `parallel_scf_jk=True` by default and
use the MPI DF executor for SCF J/K
contractions, their density response, and
the independent three-center coordinate derivative blocks. Root retains the
SCF iteration and differentiation tape, localization, and coupled metric
response. Common Fock preparation and its pullback use the same executor.
Workers serve these collective phases, then resume their independent domain
work.

MP2 and CC owners call the same domain preparation and solver kernels as
serial execution, including half-transformed DF reuse and retained CC integral
blocks. Both schedulers initialize persistent compilation caches on workers
and reuse root's existing response buffers. The optional distributed SCF
service validates density factors like serial, reuses block scratch during
each density response, and verifies each rank's DF contents once before its
first contraction. It does not yet cache occupied panels across response
iterations or use serial factorized geometry response; these optimizations
are available through the root-owned SCF path selected with `parallel_scf_jk=False`.

Every rank calls the driver with the same molecule and configured MF type.
Only root supplies controls and either `static` or `build_static`. The gradient
builder must support a worker branch that adopts root's canonical reference
without SCF:

```python
def build_mf(mol, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.chkfile = None
    if mo_coeff_init is None:
        mf.kernel()
    else:
        mf.mo_coeff, mf.mo_energy = mo_coeff_init, mo_energy_init
        mf.mo_occ, mf.e_tot = mo_occ_init, e_tot_init
        mf.converged = True
    return mf
```

Workers use root's signed canonical/common arrays and matching basis definitions.
With MPI SCF enabled, the worker builder first receives placeholder canonical
arrays to create an unrun RHF/DF reference. The driver builds its global DF
factors for the service phase; it later calls the builder with root's converged
canonical arrays for domain work. Configure identical auxiliary bases and
real float64 DF factors on every rank. The service packs flattened full
in-core AO-pair factors automatically and verifies factor contents and
molecular definitions before its first contraction. Its private source view distributes coordinate
derivatives for both in-core and out-of-core factors. If attaching an
out-of-core CDERI file, attach it before the worker branch returns and make it
accessible to every rank. With `parallel_scf_jk=False`, workers need no global
CDERI. For energy-only `kernel`, workers supply matching MF skeletons and the
driver installs root's canonical state; its MPI DF service covers common
preparation, while any SCF performed before calling `kernel` is caller-owned.
`comm=None` and communicator size one retain serial domain execution. System
scope and low-level independent-replica MPI retain their existing behavior.
The total sampling tolerance is divided by `sqrt(global_group_count)` once,
and group seeds are independent of the owner rank.

Set numerical thread limits before importing JAX/NumPy. For example, with eight
allocated physical cores:

```bash
OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYSCFAD_DENSE_BLAS_THREADS=2 PJRT_NPROC=2 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
mpiexec -n 4 python examples/dlno_stc/water_tetramer.py \
  --mpi --xyz-file water_4.xyz --basis cc-pvtz --auxbasis cc-pvtz-jkfit \
  --output results/water4.json
```

With JAX 0.9.1, `PJRT_NPROC` sizes the CPU runtime's worker pools before the
backend initializes. OpenMP/BLAS limits and `xla_cpu_multi_thread_eigen=false`
alone do not limit those pools. The water-16 launcher and input set
`PJRT_NPROC` from `THREADS` / `--threads` and record it in `settings.json`.
Compilation and helper threads can still make the process's total thread count
larger than this setting; it is not a process-wide thread or memory limit.

Memory limits apply per process. Round-robin balances group counts rather than
runtime, so large domains can accumulate on one rank; the slowest owner limits
each phase. Rank-specific JAX caches remain persistent, and cold compilation
should be separated from warm scaling measurements.

## Array and native contract

All numerical requests contain exactly `foo(oo)`, `fvv(vv)`, and `B(Pia)` as
real float64 arrays. Fock matrices are full symmetric blocks in the same local
orthonormal frame as `B`, before Fock diagonalization. Selected local occupied
indices are distinct, nonnegative integer labels and travel as metadata.
Results provide a scalar and exactly three unit-seed bars in the input frame.

`_stc_mp2.solve_selected(foo, fvv, B, target_indices, controls, with_grad=True)`
returns the sum of selected target energies with identity
`boys_target_laplace_mp2_v1`. Its sampling error estimates that sum, including
within-group covariance. `_stc_mp2.solve_full` retains the complete-system
algorithm. The old orbital-weight solver and five-array boundary are removed.
Laplace quadrature weights and mathematical sampling probabilities are retained.

## Native D/W matrices and delayed reverse

`backend.solve_matrices` and `backend.pullback_matrices` are explicit host calls.
They use the same three prepared tensors and local occupied/virtual frames as
the energy solver. D is the native MP2 selection density, with its native
normalization; W is a separate signed energy-weighted approximation. Neither
call constructs a relaxed physical one-particle RDM.

Native v2 W energy is `0.5 * (trace(Woo) + trace(Wvv))`. Its reported energy
SE includes covariance between the occupied and virtual trace samples; it is
not obtained by combining their marginal errors independently. Consequently,
an energy-only W seed is equivalent to half-identity seeds on both matrices,
using the same saved production plan. Deterministic W has both traces equal
to its energy within numerical tolerance.

The CCSD driver reads the strong-domain MP2 energy directly from W in both
deterministic and stochastic runs. Every physical fragment contributes its W
energy once, including fragments omitted from the CC correction selection;
there is no separate strong-domain MP2 solve for W. The energy response is
included in the same W pullback as the LIS response. Baseline and matrix
Laplace grids must match; mismatched grids are rejected. The MP2 subtraction
inside the truncated LIS and the weak-pair completion are unchanged.

```python
import jax
import numpy as np
from pyscfad.dlno_stc import adjoint, backend, prepare

# mf, common, static and fragment_index are the existing preparation objects.
inputs, preparation_vjp = jax.vjp(
    lambda current_mf, current_common: prepare.prepare_inputs(
        current_mf, current_common, static, fragment_index), mf, common)
metadata = {"fragment_id": fragment_index, "target_indices": (0,)}
forward = backend.solve_matrices(
    inputs, metadata, {"mode": "deterministic", "target_batch_size": 1},
    matrix_kind="D")
# Supply downstream output bars later, in the same input frame.
input_bars = backend.pullback_matrices(
    inputs, forward, energy_bar=0.,
    occupied_bar=np.eye(inputs["foo"].shape[0]),
    virtual_bar=np.eye(inputs["fvv"].shape[0]))
mf_bar, common_bar = adjoint.apply_input_pullback(
    preparation_vjp, inputs, input_bars)
```

Use the actual saved local target slot(s) in `metadata["target_indices"]`;
`(0,)` above is only an example. Group outputs are sums, not a per-target stack.
The reverse differentiates `energy_bar*E + sum(occupied_bar*Xoo) +
sum(virtual_bar*Xvv)` with the full Frobenius convention, including asymmetric
seeds. Its three input bars are already seeded: do not multiply them by another
energy bar. Numerical inputs, target/frame metadata, grid, resolved native
options, native version and auxiliary offsets are fingerprinted. Reusing a
forward with changed inputs, controls or native version raises an error. Replay
records are owned in memory and are not serializable; v1 records cannot be used
with v2.

Matrix controls are separate from the established stochastic energy controls:

| Control | Matrix behavior |
| --- | --- |
| `mode`, `laplace_roots`, `laplace_weights` | Deterministic or stochastic; the existing six-point grid is the default. Stochastic D/W requires strictly positive weights. |
| `global_seed`, `threads` | Explicit seed and native thread count; matrix default threads is 1. Seeds are derived per fragment and batch. |
| `target_batch_size` | Default `None` shares one native call across all targets in the physical fragment. An explicit positive integer caps the batch for memory constraints; full orbital spaces are retained. |
| `memory_limit_mb` | Optional advisory budget in decimal MB for current RSS plus estimated additional matrix workspace. Exceedance warns and continues. Checks run before both forward and reverse host/native allocations; diagnostics include the breakdown and estimated process peak (unknown when RSS is unavailable). |
| `auxiliary_group_size` | Default 32 rows per group when `aux_offsets` is omitted; an explicit partition takes precedence and is included in memory/replay accounting. |
| `matrix_tolerance` | Stochastic D/W Frobenius SE target for each summed matrix, default `1e-4`. Inspect measured SEs and `target_met`; allocation is not an attainment guarantee. |
| `domain_tau`, `virtual_domain_tau` | Stochastic occupied/virtual exact-core thresholds, defaults `.003` and `.0003`. |
| `pair_pilot_samples`, `min_samples` | Stochastic matrix pilot and production minimum, defaults 100000 and 64. |
| `resident_points` | Stochastic D only: 2 or 3, default 3. W retains its fixed and dressed factors and rejects this ineffective setting. |
| `energy_tolerance`, `system_workload_cutoff` | Stochastic D only, defaults `3e-4` and `.0065`. W's half-trace energy and covariance-aware energy SE come from matrix samples; W has no separate energy allocation target. |

Deterministic matrices reject sampling controls. Matrix calls reject fixed-count
energy options (`production_samples`, `pilot_samples`, production limits) and
unknown settings. Those existing controls continue to work on the separate
stochastic energy solver. New deterministic energy calls accept
`target_batch_size` and optional `threads`; stochastic energy uses the existing
OpenMP settings and rejects these new deterministic-only controls.
For standalone deterministic selected energy, `target_batch_size` bounds the
bare-integral target cache (default 1). All targets share each point's forward
transformation and accumulated reverse regardless of that cache size. Uncached
bare pair integrals are rebuilt in worker-local scratch. Diagnostics expose
transformation/reverse counts and the kernel timing sections.

The STC packet, finite-domain, system and D/W matrix memory guards treat
`max_memory` and `memory_limit_mb` as advisory. They warn when a conservative
estimate exceeds the configured budget, or a process-peak forecast exceeds
90% of a detected planning allowance. Estimates above 100% also continue.
`max_memory` still controls existing block sizes, and native tiling and cache
fallbacks retain their existing behavior. Generic MP2, RCCSD, RPA, Davidson
and DIIS memory checks are outside this policy.

Each public calculation emits at most one warning per stage and rank. Warnings
show the estimate, RSS, configured budget, detected hard limit, advisory
headroom and planning allowance with their sources. Linux detection follows
the process's cgroup v1/v2 membership and visible ancestor hard limits. V1
ancestor limits require verified hierarchical accounting; ambiguous mount
membership, unreadable values and unlimited readings remain unknown. Slurm node or CPU allocations
are divided by known local ranks for planning, never by global MPI size.
Uncertain local-rank mappings supply no Slurm per-rank allowance.

A guard raises `MemoryError` only if a justified unavoidable new array exceeds
a verified finite cgroup memory-plus-swap capacity (v2 RAM plus swap limits,
or the v1 combined `memory.memsw` limit). RAM alone is advisory when swap
capacity is unknown or unlimited. Safety multipliers, RSS plus estimated workspace,
cgroup usage/headroom, host `MemAvailable`, and Slurm allocations cannot prove
an allocation impossible. Packet export lacks an exact local auxiliary size,
so its global auxiliary estimate can only warn. Allocation failures and integer
overflow checks continue to propagate through the existing MPI error paths.

The W adapter explicitly enables virtual `w_virtual_reform`, matching the
supplied v2 driver. Other sampler options retain the supplied defaults,
including occupied stratum mixtures and virtual splitting. Effective options
are reported per batch and included in replay identity; experimental switches
are not exposed as new host controls.

Each stochastic batch receives independent streams and `1/sqrt(batch_count)`
of the requested SE budget. The returned SE is the square root of the sum of
batch variances, and the combined target flags use that measured uncertainty.
Each W batch contributes its full native energy variance, including its
occupied/virtual covariance, before independent batch variances are summed.
No matrix SE or energy SE is a nuclear-gradient uncertainty estimate.

Memory diagnostics include atom-weight arrays, simultaneous B/C proposal
caches, striped locks and bounded reverse update/radix buffers, as well as
factors, compact caches and matrix accumulators. Delayed reverse rechecks
current RSS because downstream arrays may have been allocated after forward.

Deterministic energy uses the supplied Laplace energy-only/fused kernel; it
does not build D/W. Explicit target batches bound its selected pair-Gram cache.
Current QR-anchor/Cholesky orbital construction, weak completion, MPI ownership,
unit-energy scalar bars and the gradient sign convention remain as before.
Adapter/source differences and original hashes are listed in
`cpp/native/SOURCE_MANIFEST.txt`. Existing restart identities are retained:
the scalar Laplace equations are unchanged, while their grids and scientific
controls remain covered by request fingerprints.

## Energy sampling budget

Public `energy_tolerance` is the total requested sampling standard error in Eh.
It is divided exactly once by the square root of the number of actual grouped
strong solver jobs: `group_tolerance = energy_tolerance / sqrt(group_count)`.
This assigns equal variance budgets, whose sum is `energy_tolerance**2` for
independent group streams. Sixteen targets in four groups receive
`0.0003 / sqrt(4) = 0.00015 Eh` per job. With merging disabled, they receive
`0.0003 / sqrt(16) = 0.000075 Eh`. Caller dictionaries are copied. Weak pairs
and MPI replicas do not enter the divisor; fixed-count overrides and replica
reduction retain their previous meaning. A group standard error describes
energy, never force uncertainty. Reused or unverified sampling streams do not
justify summing group variances as an independent combined uncertainty.

## Disk exchange and restart

`mp2.export_run` and `finite.export_finite_run` use one implementation. Export
writes one `fragment_NNNN/input.h5` per group, even for full-support selections.
Packets declare `finite_boys_target_mp2_v1`; the name identifies the selected
finite-frame functional even when its chosen support spans the molecule.
`mp2.solve_request` consumes the saved effective controls and writes `result.h5`.
`mp2.imported_value_and_grad` rebuilds all three tensors, checks their values,
applies imported bars and closes weak, common and SCF response.

```python
from pyscfad.dlno_stc import mp2, backend
mp2.export_run(mf, static, "run", controls=controls)
# One execution per exported group:
mp2.solve_request("run/fragment_0000/input.h5", "run/fragment_0000/result.h5", backend.solve)
static = mp2.load_static_from_run(mf, "run", frozen=4, controls=controls)
energy, gradient = mp2.imported_value_and_grad(
    mol, build_mf, static, lambda i: mp2.load_disk_packet("run", i),
    include_hf=True,
)
```

Use the packet adapter exposed by `backend` for your installed backend API;
`examples/dlno_stc/workflow.py` demonstrates the complete workflow using a tiny
independent deterministic reference. `finite.py` retains thin compatibility
entry points; its replay wrapper defaults to correlation energy (`include_hf=False`).
Partial job replay requires both `include_hf=False` and `include_weak=False`.

Fingerprints cover tensors, controls, molecular preparation, frozen space,
original/global target identities, actual local `target_indices`, ordered Boys
candidates, AO/auxiliary supports, retained PAO ranks and anchors, `local_before_fock_v3`,
group membership and sampling budget. Budget metadata stores `total_tolerance`,
`original_target_count`, `group_count`, and `effective_tolerance`. Replay does
not allocate again. Responses must echo the selected identities and frame.
Saved multi-group packets using the previous linear allocation must be
regenerated before replay; budget validation requires the square-root rule.

The existing STC `RestartManager` stores concrete selections and typed grouping
metadata; its registry accepts only required STC-owned classes. The model tag is
`boys_domain_model="local_selector_v1"`. Native DLNO checkpoints and old
weighted or target-free finite packets are rejected; there is no migration.
Use `merge_identical=False` consistently for diagnostic singleton export/replay.

## Build and checks

From the repository root, with PySCFAD, pybind11, Armadillo and OpenMP available:

```bash
cmake -S pyscfad/dlno_stc/cpp -B build/dlno_stc \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/pyscfad/dlno_stc"
cmake --build build/dlno_stc -j 4
ctest --test-dir build/dlno_stc --output-on-failure
python -m pytest pyscfad/dlno_stc/test/test_selected_kernel.py \
  pyscfad/dlno_stc/test/test_full_kernel.py \
  pyscfad/dlno_stc/test/test_protocol.py \
  pyscfad/dlno_stc/test/test_finite_workflow.py
```

The optional extension links Armadillo and OpenMP; it requires no Q-Chem or MPI
library. Imports do not compile or download anything. Examples include the
small disk workflow, finite `water_tetramer.py`, and unchanged system-scope
`water_tetramer_full.py`.

Scoped BLAS threading for the triples DF contractions is optional and disabled
by default. To enable it, add these options to the CMake configure command:

```bash
  -DPYSCFAD_STC_OPENBLAS_THREAD_CONTROL=ON \
  -DPYSCFAD_STC_OPENBLAS_LIBRARY="/absolute/path/to/pthread/libopenblas.so"
```

Use the same pthread OpenBLAS shared library that Armadillo uses at runtime;
the OpenMP OpenBLAS backend is not supported by this option. At runtime, set
`PYSCFAD_STC_DENSE_BLAS_THREADS=8` while retaining `OPENBLAS_NUM_THREADS=1` for
the surrounding OpenMP work. Only the large forward DF contractions and final
DF-factor adjoints use the requested BLAS count. The previous count is restored
before sampling and on exceptions. Leaving the environment variable unset
preserves the existing BLAS policy; setting it without a supported build raises
an error. BLAS thread limits are process-wide, so concurrent calls that change
this policy are unsupported. The enabled build adds a CTest for scope restoration;
`pyscfad/dlno_stc/test/test_triples_blas_threads.py` checks numerical parity and
observed thread counts.

Restart algorithm ABI 3 and grouping convention `boys_exact_domain_groups_v3`
require fresh checkpoints after restoring the anchored virtual frame. Older
finite packets using `local_before_fock_v1` or `local_before_fock_v2` must be
re-exported; the outer tensor exchange schema and checkpoint container version
remain 1.

## D/W-selected LIS and CCSD(T)

`pyscfad.dlno_stc.ccsd` provides `build_static_selections`, `kernel` and
`value_and_grad`. Each physical Boys fragment has its own retained LIS and
CCSD(T) correction; identical extended domains may share preparation.

Domain-space LIS is the only supported construction. D/W is diagonalized
within the orthonormal occupied/virtual ED frames. Occupied selection retains
the normalized projections of the physical fragment's original Boys targets
and selects LNOs in their orthogonal complement. Virtual selection uses the
ED matrix directly. Zero cutoffs retain the entire corresponding ED space.
The `lis_space` checkpoint marker and optional Python argument accept only
`"domain"`; the benchmark no longer has a `--lis-space` selector.

CCSD receives the explicit local occupied count and only active LIS columns.
Fragment weights use the normalized domain targets. The global SCF AO Fock
is projected into the LIS, including occupied-virtual entries. SCF, Boys/PAO,
ED, D/W and retained-space responses remain differentiated. Smooth frames
and deterministic or stochastic triples use this same domain definition.
Global-space LIS checkpoints are rejected and must be rebuilt. Selection
schema 10 remains valid for existing domain-space checkpoints.

`build_static_selections` defaults to `single_pass=True` for W calculations
and returns a plan without a W prepass. Production
prepares one domain group, selects each fragment's LIS from its W result, and
retains the same input/frame pullback through its CCSD(T)-MP2 correction and
joint W/MP2 reverse. Set `single_pass=False` for two-pass selection, required for
D matrices or callers that need concrete LIS selections, anchors
and grids before production. `benchmark_lno_w_triples.py` also defaults to
single-pass. With MPI, each owner selects and differentiates the LIS for its
assigned domain groups. SCF and common response can stay on root with
`parallel_scf_jk=False`. Both paths use only domain
eigensystems. The former `virtual_eigenspace` Python argument and
`--lis-virtual-eigenspace` CLI switch have been removed.

Single-pass adaptive triples uses `variance_allocation="occupied_count"`:
`w_f = n_occ_fragment / n_active_occ_total` and
`target_SE_f = total_SE * sqrt(w_f)`. These counts refer to original physical
occupied targets, not extended-domain dimensions. Subset calculations keep
the whole-molecule denominator. This replaces the legacy `mp2_energy` proxy
allocation and may change sample counts and finite-sample triples results.
Neither allocation is differentiated. The common-domain pullback remains:
summed fragment adjoints must still pass through Boys/PAO construction,
overlap and Fock data, followed by the separate SCF orbital response.

DF-CCSD uses the improved virtual ladder contractions from upstream PySCFAD.
The default `vvvv_direct=False` caches the symmetry-packed virtual integrals
for repeated CCSD and response iterations when the cache fits within 60% of
remaining `max_memory`; otherwise the solver warns and uses the improved
uncached contraction. Set `vvvv_direct=True` on `ccsd.kernel` or
`ccsd.value_and_grad` to force the lower-memory mode. Canonical and LNO
`RCCSD` objects expose the same attribute, set before `ao2mo`/`kernel`.
Both modes always cache packed `(ov|vv)`, along with the existing `oooo`,
`ovoo`, `ovov`/`ovvo`, and `oovv` blocks. CCSD updates, lambda contractions,
and the analytical response replay read tiles from the packed cache; they
do not rebuild `(ov|vv)` from DF factors. Its float64 storage is
`8 * nocc * nvir * [nvir * (nvir + 1) / 2]` bytes per active fragment.
Smooth triples uses the stable external-leg numerator in both input modes.
The default input mode constructs its integral blocks from DF factors. Set
`triples_controls={"mode": "deterministic", "integral_transform": "rotate"}`
to rotate cached `ovvv`, `ovoo`, and `ovov` blocks at each Laplace point instead.
The same `integral_transform` option is available in stochastic mode. This path
needs no transformed B factors and accumulates the cached-integral response
across points before propagating it to the original DF factors.
The float64 `vvvv` cache uses `16 * [nvir * (nvir + 1) / 2]**2` bytes per active
fragment; concurrent MPI ranks each need their own cache allocation.
The selected mode is included in serial CC checkpoint identity.
The default analytical CCSD response keeps both cache cotangents symbolic.
If `pyscfad_dfccsd_custom_response=False` selects generic implicit response,
JAX can additionally materialize one `vvvv`-cache-sized zero cotangent; use
`vvvv_direct=True` when that extra memory is undesirable.

The retained stochastic implementation uses full occupied/virtual LIS frames,
native asymmetric Laplace triples, and physical selected-space restricted
sampling. There are no internal Fock cluster projectors. Off-diagonal Fock,
physical D/W subspace, AO overlap and physical fragment-target responses remain.
Canonical stochastic triples and alternative dressing/frame implementations
have been removed. `smooth=False` is a deterministic validation reference only.

See the [current smooth triples guide](../../doc/source/user_guide/smooth_lis_triples.md)
for the equations, complete Python/CLI option tables and defaults, variance
allocation, diagnostics, and benchmark commands. The
[branch README](../../README_SMOOTH_CCSD_T.md) gives 32-thread C5H6 and C6H14
force-comparison examples. Library triples defaults are deterministic mode,
10 fitted points, and (when stochastic mode is requested) 10000 pilot draws per
point and trace exponent 5.0. The MP2 trace model uses exponent 1.4. The production
examples request total triples SE `5e-5` Eh (0.05 mEh). These controls affect energies and
their gradients; production energy and derivative estimates share samples.
CC selections use schema 10; rebuild saved selections/checkpoints after changing
from earlier implementations. Native energy/VJP entry points are
`triples_asymmetric_energy` and `triples_asymmetric_energy_vjp`.

## Optional STC CCSD amplitudes

`ccsd.kernel` and `ccsd.value_and_grad` accept `cc_solver="stc"` and
`stc_cc_controls`. Their default remains `cc_solver="deterministic"`; importing
or running that default does not require the external STC CCSD packages.
Install the separate `stc_cc` package from its source repository and
`stc-cc-jax[cc]` into the same environment as PySCFAD. For editable checkouts:

```bash
python -m pip install -e /path/to/stc_qc_paper
python -m pip install -e '/path/to/stc-cc-jax[cc]'
```

The latter builds its optional CPU native sampler during installation; the
external package also supports its JAX fallback. See the external packages'
installation instructions for their compiler/runtime requirements. Nothing is
compiled or installed when a DLNO calculation imports the adapter.

With an existing CC selection `cc_static` and converged density-fitted `mf`:

```python
from pyscfad.dlno_stc import ccsd

energy, components = ccsd.kernel(
    mf, cc_static, cc_solver="stc", return_components=True,
    baseline_controls={"mode": "deterministic"},
    matrix_controls={"mode": "deterministic"},
    stc_cc_controls={
        "energy_tolerance_mEh": 0.1,  # final molecular sampling standard error
        "target_error_mEh": 0.4,      # optional local update-noise setting
        "max_cycle": 50,
        "average": 10,
        "seed": 7,
        "localize": True,
        "grad_max_cycle": 50,
        "grad_average": 10,
    },
)
```

The same solver arguments apply to `ccsd.value_and_grad`. Reuse the exact
baseline/matrix controls used for the saved selection, including any fitted
Laplace grid. With the molecular precision controller, these stages and any
triples contractions must be deterministic; unsupported stochastic combinations
are rejected. Without it, the existing independent stage controls remain
available.

The primary precision input is **`energy_tolerance_mEh`**, a requested sampling
standard error for the final sum of projected corrections. For example, `0.1`
means 0.1 mEh, not 0.1 Eh. `components.stc_standard_error` and
`components.stc_target_standard_error` report Eh; multiply by 1000 for mEh.
`components.stc_target_met` states whether the observed combined SE meets its
target. This is an empirical SE, not an upper bound on the energy error.

| Input | Units | Meaning |
|---|---|---|
| `energy_tolerance_mEh` | mEh | Final molecular sampling SE, with allocation across selected LIS |
| `target_error_mEh` | mEh | Advanced local per-update noise target |
| `error_tol_mEh` | mEh | Optional upstream local mean-energy stopping SE |
| `grad_target_error_mEh` | mEh | Adjoint sampling allocation proxy; not a force tolerance |

Legacy `target_error`, `error_tol`, and `grad_target_error` retain **Eh** units;
explicit `*_mEh` inputs are converted once. Supplying both forms of a setting
is an error. Normalized controls store Eh internally, including
`energy_tolerance`. The default fixed-budget, single-solve STC path is preserved
when no molecular tolerance is supplied.

For N selected physical LIS fragments, the controller assigns each a variance
budget `tau_f**2 = tau_total**2 / N`. It evaluates independent complete-solve
pilot replicas of the actual projected correction `E_CCSD + E_T - E_MP2_LIS`.
The pilot sample variance determines a fixed independent production count
`ceil(pilot_variance / tau_f**2)`, subject to the configured minimum and maximum.
Pilots are excluded from the returned estimate. The production correction is
the **mean of replica energies**, each evaluated at that replica's averaged
CC amplitudes; it is not the energy of amplitudes averaged across replicas.
The production sample variance divided by its count estimates each LIS mean
variance. Independent fragment streams permit summing these variances, then
taking the square root for the molecular SE. When only a subset is selected,
the target covers that selected correction sum.

Controls `precision_pilot_replicas=4`, `precision_min_replicas=4`, and
`precision_max_replicas=32` bound the work. This initial empirical controller
therefore costs at least eight solves per LIS. It consumes production VJPs
sequentially and averages cotangents, preserving the existing physical
projector/domain/SCF pullback. Discrete pilot allocation is held fixed in
response. Replica precision uses ordinary local DF responses even when
`PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE` is enabled, so integral derivatives are
averaged with the other cotangents. Pilot and production streams are independent; MPI ownership does
not change their fragment seeds. A preflight checks every eligible replica's
actual 32-bit forward and adjoint seed within each stream family. If it finds a
collision, choose a different root seed or reduce the replica cap as indicated
by the error; colliding streams are not silently included in the variance sum.

Per-fragment `ccsd_diagnostics["precision"]` retains pilot and production
energies, seeds, budgets, counts, the full component covariance, observed SE,
and cap/target status. CCSD/triples covariance is included because deterministic
triples still depend on stochastic amplitudes. A cap or missed empirical
target issues a warning; no successful accuracy claim is made silently.
Finite pilot uncertainty and finite-iteration/nonlinear/domain/quadrature bias
are outside this sampling SE. A molecular energy target does not control force
variance.

The adapter uses the already prepared retained LIS factors and Fock matrix.
`localize=True` chooses a private PySCF Boys frame within its occupied and
virtual spans; `False` keeps their supplied frame. This private gauge is fixed
in reverse mode. Tensor transformations, physical domain/LIS preparation,
projector response and SCF response remain differentiable. Semicanonical
Fock denominators are constructed in the private frame. The external adjoint
acts on differentiable **averaged amplitudes**, which by default stay in the
shared private Boys frame for projected fragment energy and smooth triples. The
external total-energy-only orbital cotangent filter is not applied. Stable
physical fragment IDs receive distinct reproducible sampling streams;
changing MPI ownership does not change the intended fragment seed.

Selecting `cc_solver="stc"` defaults to `localize=True` and
`shared_local_frame=True`; no extra frame flags are needed. The shared frame
retains the averaged amplitudes in the private Boys frame and evaluates both
the projected CCSD energy and smooth (T) in that same frame, using the rotated
DF factors, full Fock matrix and physical projector `prjlo @ Q_occ`. There is
no final amplitude backrotation, including for diagnostics. The original LIS
MP2 subtraction is unchanged. With `localize=False` the shared frame is the
incoming LIS frame, without an additional Boys rotation.

The occupied/virtual rotations are computed once per fragment solve and held
fixed through its CCSD, triples and saved adjoint evaluation. The rotated
physical projector remains differentiable; fixing the private rotation does
not freeze domain/LIS, integral or SCF response. No numerical frame is cached
across fragments or geometries. This is the same fixed-private-gauge response
convention as the original STC adapter, not an additional Boys response solve.

Shared-frame triples requires `smooth=True` and supports deterministic and
stochastic triples, with either `integral_transform="df"` or `"rotate"`.
The cached-integral path constructs its local `ovvv` block lazily from the
already rotated DF factors. Full-Fock Laplace dressing is still required at
each node. Conventional diagonal-denominator triples rejects a shared frame.
To restore the previous backrotation route, set
`stc_cc_controls={"shared_local_frame": False}` alongside any sampling controls.
This explicit override is also required for STC+(T) with `smooth=False`.
The frame option changes the representation, not the retained LIS, CCSD
equations or precision budget. The overall solver default remains deterministic.
Finite stochastic (T) samples need not be identical in different frames even
though the deterministic functional is rotation invariant.

Supported local controls include `target_error` (or `target_error_mEh`), optional `error_tol` (or `error_tol_mEh`), `max_cycle`,
`average`, `seed`, `localize`, `shared_local_frame`; `diis_space` (`None` disables it, otherwise at
least two), `diis_shift`, `init_damp`; `gamma`, `batch_size`, `tail_batch_size`,
`nrun`, `init_nsamples`, `std_tol`, `max_nsamples_var`, `min_nsamples`;
`sampled_terms`, `share_tables`, `iid_root`, `unroll`, `flat`, `sampler`,
`block_size`, `block_min_entries`, `block_store_P`; and `grad_max_cycle`,
`grad_average`, `grad_target_error`, `grad_seed`, `vjp_nsamples`. Unknown or
invalid controls fail before solving. Variance pilots require `nrun >= 2` and
`init_nsamples >= 2`. `average=0` selects the last iterate.

`target_error`, `error_tol` and `grad_target_error` use Hartree. With
`error_tol=None`, the solver completes a fixed iteration budget; its upstream
`converged=True` means that budget completed and **does not certify CC residual
convergence**. With `error_tol`, the upstream flag means its total-local
mean-energy standard-error stopping rule was satisfied. The returned
`termination_mode` and `convergence_note` make this distinction explicit.
Neither setting guarantees the accuracy of a projected fragment or a force.

In the single-solve mode, each fragment's `ccsd_diagnostics` records controls,
effective seeds and projected iterate energies. Molecular precision mode
records its independent-replica estimator and nested precision diagnostics. `nsamples_forward_production` counts only forward
production draws; it excludes variance pilots and adjoint draws and is not a
total-cost metric. `projected_mean_iterate_energy` is the
mean of observed projected energies in the external solver's averaging
window; `projected_energy_at_mean_amplitudes` is the physical projected energy
at the returned mean amplitudes. Their difference reflects nonlinear
averaging. `projected_mean_iterate_stderr_iid` is only an observed IID-iterate
estimate: it ignores autocorrelation and burn-in bias. The external
`local_mean_iterate_stderr` refers to the total active-space functional,
not the projected fragment. Use independent seeds and deterministic
references to assess finite-iteration/sampling error; these diagnostics are
not force-error bounds. The upstream external adjoint currently fails for an
exactly zero amplitude cotangent (its residual normalization divides by zero);
objectives whose amplitude response is identically zero should avoid calling
that external backward path until it is fixed upstream.

Serial checkpoints include the backend and normalized controls. STC
checkpoints additionally fingerprint installed external package versions,
Python sources and native binaries, including editable source changes and
the native extension selected at runtime. Changed implementation identity
rejects resume. Checkpoints must use the current CC selection schema;
serial progress checkpoints remain unavailable for multi-rank execution.
