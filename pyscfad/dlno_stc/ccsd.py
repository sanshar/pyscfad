"""Physical Boys-fragment CCSD(T) corrections with shared W energy.

Discrete domains and LIS ranks are fixed. Native D/W chooses each LIS in its
STC frame. The solver uses normalized domain projections of the original
physical Boys targets.
"""
from dataclasses import dataclass, replace
import gc
import os

import jax
import numpy
from pyscfad.tools import resource_profile
from pyscfad import numpy as np
from pyscfad.lno.ccsd import _impurity_solve_core

from . import backend, driver, _stc_precision
from ._profile import response as profile_response
from ._memory import memory_calculation
from ._selection import (DomainSelections, _physical_groups, _strong_preparation_key,
                         build_domain_groups)
from ._workflow import allocate_domain_controls
from ._df_basis import (basis_kwargs, basis_identity, selection_auxbasis,
                        validate_selection_basis)
from .adjoint import apply_input_pullback
from .controls import (normalize_matrix_controls, normalize_triples_controls,
                       DEFAULT_TRIPLES_TRACE_EXPONENT)
from .domain import build_boys_ed_frame
from .protocol import validate_full_result, validate_input_cotangents


CC_SELECTIONS_SCHEMA_VERSION = 10


def _clear_jax_caches(stage, *live_results, **metadata):
    """Optionally trade recompilation for lower residency between eager stages.

    Saved VJP residuals remain live: only JAX's compilation caches are cleared.
    This does not impose a hard memory limit or change the selected spaces.
    """
    if os.environ.get('PYSCFAD_STC_CLEAR_JAX_CACHES', '0').lower() not in (
            '1', 'true', 'yes', 'on'):
        return
    with resource_profile.section('stc.cc.clear_jax_caches', stage=stage, **metadata):
        jax.block_until_ready(live_results)
        jax.clear_caches()
        gc.collect()


@dataclass(frozen=True)
class CCSDSelections:
    """Fixed LIS choices and the scientific settings used to select them."""
    lis_static: object
    controls_identity: tuple
    schema_version: int = CC_SELECTIONS_SCHEMA_VERSION
    mp2_fragment_energies: tuple | None = None
    deferred_selection: dict | None = None
    correlation_basis_identity: str | None = None

    @property
    def mp2_static(self):
        return self.lis_static.mp2_static

    @property
    def matrix_kind(self):
        return self.lis_static.matrix_kind

    @property
    def fragments(self):
        return self.lis_static.fragments


@dataclass(frozen=True)
class FragmentComponents:
    fragment_index: int
    e_mp2_lis: float
    e_ccsd: float
    e_triples: float
    matrix_energy_diagnostic: float
    nocc_lis: int | None = None
    nvir_lis: int | None = None
    direct_host_bar_norm: float | None = None
    native_host_bar_norm: float | None = None
    matrix_diagnostics: dict | None = None
    triples_method: str = 'none'
    triples_standard_error: float = 0.
    triples_samples: int = 0
    triples_timings: dict | None = None
    triples_diagnostics: dict | None = None
    ccsd_method: str = 'deterministic'
    ccsd_diagnostics: dict | None = None
    auxiliary_compression: dict | None = None

    @property
    def e_ccsd_t(self):
        return self.e_triples


@dataclass(frozen=True)
class CCSDComponents:
    """Correlation accounting; W supplies its own strong-domain MP2 energy."""
    e_mp2_base: float
    e_mp2_weak: float
    fragments: tuple
    matrix_kind: str
    baseline_standard_error: float

    @property
    def e_ccsd(self):
        return sum(f.e_ccsd for f in self.fragments)

    @property
    def e_mp2_lis(self):
        return sum(f.e_mp2_lis for f in self.fragments)

    @property
    def e_triples(self):
        return sum(f.e_triples for f in self.fragments)

    @property
    def e_ccsd_t(self):
        return self.e_triples

    @property
    def e_corr(self):
        return self.e_mp2_base + self.e_ccsd + self.e_triples - self.e_mp2_lis

    @property
    def triples_standard_error(self):
        """Energy SE for independently sampled LIS triples, excluding D error."""
        return sum(f.triples_standard_error**2 for f in self.fragments)**.5


    @property
    def stc_standard_error(self):
        """Empirical SE of independent final projected replica means, or None."""
        statistics = [(f.ccsd_diagnostics or {}).get('precision') for f in self.fragments]
        if not statistics or any(item is None for item in statistics):
            return None
        return sum(item['standard_error']**2 for item in statistics)**.5

    @property
    def stc_target_standard_error(self):
        """Requested SE over selected physical fragments, in Hartree, or None."""
        statistics = [(f.ccsd_diagnostics or {}).get('precision') for f in self.fragments]
        if not statistics or any(item is None for item in statistics):
            return None
        return sum(item['fragment_target_standard_error']**2 for item in statistics)**.5

    @property
    def stc_target_met(self):
        """Whether the empirical global SE meets its target; no bias guarantee."""
        error, target = self.stc_standard_error, self.stc_target_standard_error
        return None if error is None else error <= target


def _cc_solver_options(cc_solver, stc_cc_controls, lis_aux_svd_tol=None):
    """Validate the optional backend without importing external STC packages."""
    from ._aux_compress import normalize_tolerance
    tolerance = normalize_tolerance(lis_aux_svd_tol)
    options = {} if tolerance is None else {'lis_aux_svd_tol': tolerance}
    if cc_solver not in ('deterministic', 'stc'):
        raise ValueError("cc_solver must be 'deterministic' or 'stc'")
    if cc_solver == 'deterministic':
        if stc_cc_controls is not None:
            raise ValueError("stc_cc_controls requires cc_solver='stc'")
        return options
    from .stc_ccsd import normalize_controls
    return dict(options, cc_solver='stc', stc_cc_controls=normalize_controls(stc_cc_controls))


def _validate_shared_frame_triples(static, ccsd_t, stc_cc_controls):
    """A local amplitude frame needs full-Fock rather than diagonal triples."""
    if not ccsd_t or not (stc_cc_controls or {}).get('shared_local_frame', False):
        return
    deferred = getattr(static, 'deferred_selection', None)
    smooth = (deferred['smooth'] if deferred is not None else
              static.lis_static.smooth_frames is not None)
    if not smooth:
        raise ValueError('shared_local_frame with (T) requires smooth=True full-Fock triples; '
                         'set shared_local_frame=False to backrotate to the incoming LIS frame')


def _normalize_controls(baseline_controls, matrix_controls, matrix_kind):
    baseline = driver._driver_controls(baseline_controls, scope='domain')
    baseline.setdefault('mode', 'deterministic')
    baseline.setdefault('global_seed', 0)
    matrix = normalize_matrix_controls(matrix_controls, matrix_kind)
    if matrix_kind == 'W' and any(not numpy.array_equal(baseline[key], matrix[key])
                                 for key in ('laplace_roots', 'laplace_weights')):
        raise ValueError('W supplies the MP2 energy; baseline and matrix Laplace grids must match')
    return baseline, matrix


def _scientific_identity(baseline, matrix):
    # Memory/thread knobs do not alter the scientific selection. Sampling
    # allocation, seeds, batching and the Laplace grid do and remain recorded.
    operational = {'threads', 'memory_limit_mb', 'virtual_block_size'}
    def freeze(value):
        if isinstance(value, dict):
            return tuple((key, freeze(item)) for key, item in sorted(value.items())
                         if key not in operational)
        if isinstance(value, (list, tuple, numpy.ndarray)):
            return tuple(freeze(item) for item in value)
        return value.item() if isinstance(value, numpy.generic) else value
    return freeze(baseline), freeze(matrix)


def _validate_run(static, baseline_controls, matrix_controls, matrix_kind):
    if not isinstance(static, CCSDSelections):
        raise TypeError('static must be CCSDSelections from build_static_selections')
    if vars(static).get('schema_version') != CC_SELECTIONS_SCHEMA_VERSION:
        raise ValueError('incompatible CC selection schema; rebuild domain-space LIS selections')
    from .lis import _validate_lis_space
    _validate_lis_space(static.lis_static.lis_space)
    for fragment in static.fragments:
        if fragment is not None:
            _validate_lis_space(fragment.lis_space)
    kind = static.matrix_kind if matrix_kind is None else matrix_kind
    if kind != static.matrix_kind:
        raise ValueError('matrix_kind differs from the saved LIS selection')
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, kind)
    if _scientific_identity(baseline, matrix) != static.controls_identity:
        raise ValueError('scientific controls differ from the saved LIS selection')
    return baseline, matrix


def _fragment_indices(static, indices, *, selecting=False):
    fragments = (_physical_groups(static) if selecting and isinstance(static, DomainSelections)
                 else static.fragments)
    deferred = getattr(static, 'deferred_selection', None)
    available = (tuple(deferred['indices']) if deferred is not None else
                 tuple(i for i, fragment in enumerate(fragments)
                       if selecting or fragment is not None))
    if indices is None:
        return available
    indices = tuple(indices)
    if any(isinstance(i, (bool, numpy.bool_)) or not isinstance(i, (int, numpy.integer))
           for i in indices):
        raise TypeError('fragment_indices must contain integers')
    if len(set(indices)) != len(indices) or any(i not in available for i in indices):
        raise ValueError('fragment_indices must be unique available LIS selections')
    return tuple(int(i) for i in indices)


def _group_cc_fragments(static, group):
    """Map physical IDs to target slots in their first member's preparation job.

    Preparation merging affects ownership only. Every physical member must
    have exactly the representative's ordered strong frame, even when the
    preparation jobs themselves are left unmerged.
    """
    domains = static.mp2_static if hasattr(static, 'mp2_static') else static
    if not isinstance(domains, DomainSelections):
        raise TypeError('physical CC grouping requires STC Boys DomainSelections')
    physical = _physical_groups(domains)
    owned = [(index, members) for index, members in enumerate(physical)
             if members[0] in group.fragment_indices]
    if not owned:
        return ()
    labels = tuple(int(columns[0]) for columns in domains.frag_lolist)
    representative = group.representative_index
    frame_key, _, _ = _strong_preparation_key(
        domains, domains.fragments[representative], representative, labels)
    partners = tuple(int(i) for i in domains.fragments[representative].strong_fragments)
    result = []
    for index, members in owned:
        targets = []
        for member in members:
            key, _, _ = _strong_preparation_key(domains, domains.fragments[member], member, labels)
            if key != frame_key or partners.count(member) != 1:
                raise ValueError(f'physical CC fragment {index} has incompatible preparation frames; rebuild domains')
            targets.append(partners.index(member))
        if len(set(targets)) != len(members):
            raise ValueError(f'physical CC fragment {index} must map each target exactly once')
        result.append((index, tuple(targets)))
    return tuple(result)


def _fragment_dimensions(static, index):
    if not isinstance(static, CCSDSelections):
        return None, None
    item = static.fragments[index]
    if item is None:
        return None, None
    parent = item.parent_dimensions
    occ = (parent[0] if item.full_occupied_space else
           len(item.internal_occ_keep) + len(item.occupied_lno_keep))
    vir = (parent[1] if item.full_virtual_space else
           len(item.internal_vir_keep) + len(item.virtual_lno_keep))
    return occ, vir


def _bar_norm(*trees):
    squares = [np.sum(np.abs(leaf)**2) for leaf in jax.tree_util.tree_leaves(trees)
               if hasattr(leaf, 'dtype') and leaf.dtype != jax.dtypes.float0]
    return float(np.sqrt(sum(squares)))


def build_static_selections(mf, mp2_static, *, matrix_kind, thresh_occ, thresh_vir,
                            baseline_controls, matrix_controls,
                            internal_rank_threshold=1e-6, merge_identical=True, fragment_indices=None,
                            smooth=True, smooth_laplace_points=10, single_pass=True,
                            lis_space='domain', correlation_auxbasis=None):
    """Select physical-fragment LIS ranks from explicit D or W cutoffs and controls.

    Use the run's ``correlation_auxbasis`` for D/W factors. ``None`` uses the
    SCF fit, or inherits the fit inside a driver-owned selection callback.
    ``mp2_static`` is a concrete Boys ``DomainSelections``. D and W use
    separately supplied absolute-eigenvalue cutoffs, with no conversion.
    ``fragment_indices`` selects only those LIS ranks; unselected entries
    remain absent while all domain metadata is retained.

    Smooth anchored frames with full occupied and virtual Fock matrices are
    the default. Set ``smooth=False`` only for a deterministic semicanonical reference.
    D/W subspace and AO overlap response are retained; there are no internal
    Fock cluster projectors. Ranks, anchors and quadratures are held fixed
    during differentiation.
    ``smooth_laplace_points`` defaults to ten fitted points per LIS.
    The two-pass path saves MP2 fragment proxies for optional triples variance
    allocation. ``variance_allocation='occupied_count'`` needs no proxies.
    By default (``single_pass=True``), return a plan without preparing domains or W.
    Each production fragment selects its discrete LIS labels from its
    own W result, then uses that same result in the complete pullback. This
    path uses occupied-count triples budgets and retains only one domain's
    preparation tape at a time on each group owner, including MPI workers.
    Set ``single_pass=False`` for D matrices or concrete LIS selections,
    anchors and grids before production.
    Occupied and virtual LNO eigenproblems always
    use compact extended-domain coordinates, preserving the projected target
    occupied span and all domain-frame/target-projector responses.

    LIS orbitals are selected directly in the orthonormal ED, retaining
    normalized domain projections of the physical Boys targets. Domain space
    is the only supported definition. Zero cutoffs retain the full ED.
    Direct integral regeneration is the default for energies and gradients; set
    PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE=half for resident H_ED reuse.
    """
    correlation_auxbasis = selection_auxbasis(mf, correlation_auxbasis)
    validate_selection_basis(mf, mp2_static, correlation_auxbasis)
    identity = (None if correlation_auxbasis is None else
                basis_identity(mf, correlation_auxbasis))
    from .lis import LISSelections, _validate_thresholds, _validate_lis_space
    _validate_lis_space(lis_space)
    if not isinstance(smooth, (bool, numpy.bool_)):
        raise TypeError('smooth must be a boolean')
    if not isinstance(single_pass, (bool, numpy.bool_)):
        raise TypeError('single_pass must be a boolean')
    _validate_thresholds(thresh_occ, thresh_vir, internal_rank_threshold)
    if single_pass and matrix_kind != 'W':
        raise ValueError('single-pass LIS selection requires W')
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, matrix_kind)
    indices = _fragment_indices(mp2_static, fragment_indices, selecting=True)
    physical = _physical_groups(mp2_static)
    selections = [None] * len(physical)
    frames, grids = [None] * len(physical), [None] * len(physical)
    mp2_energies = [0.] * len(physical)
    if single_pass:
        lis = LISSelections(mp2_static, matrix_kind, thresh_occ, thresh_vir,
            internal_rank_threshold, tuple(selections),
            tuple(frames) if smooth else None, tuple(grids) if smooth else None,
            lis_space=lis_space)
        return CCSDSelections(lis, _scientific_identity(baseline, matrix),
            deferred_selection=dict(indices=tuple(indices), smooth=bool(smooth),
                smooth_laplace_points=smooth_laplace_points),
            correlation_basis_identity=identity)
    state = driver._initialize(mf, mp2_static, baseline, with_grad=False,
                               **basis_kwargs(correlation_auxbasis))
    groups = build_domain_groups(mp2_static, merge_identical=merge_identical)
    driver.lib.logger.info(mf, 'STC CC physical fragments %d; Boys grouping %s; preparation groups %d',
                           len(physical), mp2_static.boys_grouping, len(groups))
    for index, members in enumerate(physical):
        targets = tuple(int(mp2_static.frag_lolist[member][0]) for member in members)
        driver.lib.logger.info(mf, 'STC CC fragment %d; orbital records %s; targets %s; n_F=%d',
                               index, members, targets, len(members))
    for group in groups:
        corrections = _group_cc_fragments(mp2_static, group)
        if not any(index in indices for index, _ in corrections):
            continue
        inputs, host, _, frame = driver._prepare(
            state, mp2_static, group, baseline, False, return_frame=True)
        for index, targets in corrections:
            if index not in indices:
                continue
            forward = backend.solve_matrices(host, {'fragment_id': index, 'target_indices': targets},
                                             matrix, matrix_kind=matrix_kind)
            mp2_energies[index] = float(forward['energy'])
            selections[index], frames[index], grids[index] = _select_fragment_lis(
                mf, state['common'], mp2_static, index, group, frame, forward,
                matrix_kind=matrix_kind, thresh_occ=thresh_occ, thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold, smooth=smooth,
                smooth_laplace_points=smooth_laplace_points, lis_space=lis_space)
            del forward
        del inputs, host, frame
        gc.collect()
    lis = LISSelections(mp2_static=mp2_static, matrix_kind=matrix_kind,
                        thresh_occ=thresh_occ, thresh_vir=thresh_vir,
                        internal_rank_threshold=internal_rank_threshold,
                        fragments=tuple(selections),
                        smooth_frames=tuple(frames) if smooth else None,
                        smooth_grid=tuple(grids) if smooth else None, lis_space=lis_space)
    static = CCSDSelections(lis, _scientific_identity(baseline, matrix),
                            mp2_fragment_energies=tuple(mp2_energies),
                            correlation_basis_identity=identity)
    for index in indices:
        driver.lib.logger.info(mf, 'STC CC fragment %d; LIS occupied %d; LIS virtual %d',
                               index, *_fragment_dimensions(static, index))
    return static


def _select_fragment_lis(mf, common, domains, index, group, frame, forward, *,
                         matrix_kind, thresh_occ, thresh_vir, internal_rank_threshold,
                         smooth, smooth_laplace_points, lis_space='domain'):
    """Choose concrete labels/anchors from existing inputs; never build W here."""
    from .lis import LISSelections, build_fragment_lis_selection, build_fragment_lis
    matrices = {name: np.asarray(forward[name]) for name in ('occupied', 'virtual')}
    with resource_profile.section('stc.lis.select_ranks', frag=index+1):
        selection = build_fragment_lis_selection(
            mf, domains, index, matrices=matrices, matrix_kind=matrix_kind,
            common=common, frame=frame, frame_fragment_index=group.representative_index,
            thresh_occ=thresh_occ, thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold, lis_space=lis_space)
    frames = grid = None
    if smooth:
        from .smooth import select_frame
        from .smooth_triples import make_grid
        selections = [None] * len(_physical_groups(domains))
        selections[index] = selection
        partial_lis = LISSelections(domains, matrix_kind, thresh_occ, thresh_vir,
                                   internal_rank_threshold, tuple(selections), lis_space=lis_space)
        with resource_profile.section('stc.lis.select_frame', frag=index+1):
            selected = build_fragment_lis(mf, common, partial_lis, index,
                matrices=matrices, frame=frame, frame_fragment_index=group.representative_index)
            coefficients = (selected.active_occupied_coeff, selected.active_virtual_coeff)
            jax.block_until_ready(coefficients)
            if all(coeff.shape[1] for coeff in coefficients):
                frames = tuple(select_frame(coeff, common.fock) for coeff in coefficients)
                spectra = [numpy.linalg.eigvalsh(item.anchor.T @
                    numpy.asarray(common.fock) @ item.anchor) for item in frames]
        if frames is not None:
            with resource_profile.section('stc.lis.fit_grid', frag=index+1):
                grid = make_grid(*spectra, points=smooth_laplace_points)
    return selection, frames, grid


def _select_current_fragment(state, static, index, group, frame, forward):
    """Create this fragment's immutable selection while its input VJP is live."""
    options = static.deferred_selection
    with resource_profile.section('stc.cc.lis_selection', frag=index+1):
        selection, frames, grid = _select_fragment_lis(
            state['mf'], state['common'], static.mp2_static, index, group, frame, forward,
            matrix_kind=static.matrix_kind, thresh_occ=static.lis_static.thresh_occ,
            thresh_vir=static.lis_static.thresh_vir,
            internal_rank_threshold=static.lis_static.internal_rank_threshold,
            lis_space=static.lis_static.lis_space,
            **{key: options[key] for key in
               ('smooth', 'smooth_laplace_points')})
    fragments = list(static.fragments)
    fragments[index] = selection
    lis = replace(static.lis_static, fragments=tuple(fragments))
    if options['smooth']:
        all_frames, all_grids = list(lis.smooth_frames), list(lis.smooth_grid)
        all_frames[index], all_grids[index] = frames, grid
        lis = replace(lis, smooth_frames=tuple(all_frames), smooth_grid=tuple(all_grids))
    return replace(static, lis_static=lis, deferred_selection=None)


def _allocate_triples_controls(controls, nfragments, *, static=None, indices=None):
    """Allocate a total triples SE budget over globally requested physical fragments."""
    smooth = getattr(getattr(static, 'lis_static', None), 'smooth_frames', None) is not None
    if controls is not None and controls.get('integral_transform') == 'rotate' and not smooth:
        raise ValueError('integral_transform=rotate requires smooth triples selections')
    if controls is not None and controls.get('mode') == 'stochastic':
        if not smooth:
            raise ValueError('stochastic triples requires full-LIS smooth frames; '
                             'canonical triples sampling has been removed')
        controls = dict(controls)
        controls.setdefault('trace_exponent', DEFAULT_TRIPLES_TRACE_EXPONENT)
    if controls is None or 'energy_tolerance' not in controls:
        return controls
    deferred = getattr(static, 'deferred_selection', None)
    allocation = controls.get('variance_allocation', 'occupied_count' if deferred else 'mp2_energy')
    if deferred and allocation != 'occupied_count':
        raise ValueError('single-pass W requires occupied_count variance allocation')
    if allocation == 'occupied_count':
        if static is None or 'fragment_energies' in controls:
            raise ValueError('occupied_count allocation requires domains and no energy proxies')
        domains = static.mp2_static
        groups = _physical_groups(domains)
        counts = [sum(len(domains.frag_lolist[i]) for i in members) for members in groups]
        total = len(domains.active_occ_indices)
        if total <= 0 or min(counts) <= 0 or sum(counts) != total:
            raise ValueError('physical occupied targets must partition the active occupied space')
        indices = tuple(range(len(groups))) if indices is None else tuple(indices)
        fractions = {i: counts[i] / total for i in indices}
        return dict(controls, variance_allocation='occupied_count',
            fragment_variance_fractions=fractions,
            fragment_standard_errors={i: controls['energy_tolerance'] * value**.5
                                      for i, value in fractions.items()},
            fragment_occupied_counts={i: counts[i] for i in indices}, total_active_occupied=total)
    energies = controls.get('fragment_energies', getattr(static, 'mp2_fragment_energies', None))
    if energies is not None:
        if static is None or len(energies) != len(static.fragments):
            raise ValueError('fragment_energies must contain one energy per physical S')
        indices = tuple(range(len(energies))) if indices is None else tuple(indices)
        if not indices:
            return dict(controls, fragment_standard_errors={}, fragment_variance_fractions={})
        selected = numpy.asarray(energies, dtype=float)[list(indices)]
        if not numpy.isfinite(selected).all():
            raise ValueError('fragment energy proxies must be finite')
        weights = numpy.abs(selected)
        largest = weights.max()
        # A tiny floor keeps support if an MP2 contribution happens to vanish.
        weights = numpy.maximum(weights / largest, 1e-12) if largest else numpy.ones_like(weights)
        fractions = weights / weights.sum()
        targets = controls['energy_tolerance'] * numpy.sqrt(fractions)
        if numpy.any(targets == 0.):
            raise ValueError('energy_tolerance is too small for a nonzero per-fragment target')
        return dict(controls,
                    fragment_standard_errors=dict(zip(indices, map(float, targets))),
                    fragment_variance_fractions=dict(zip(indices, map(float, fractions))),
                    fragment_reference_energies=dict(zip(indices, map(float, selected))))
    target = controls['energy_tolerance'] / numpy.sqrt(nfragments) if nfragments else 0.
    if nfragments and target == 0.:
        raise ValueError('energy_tolerance is too small for a nonzero per-fragment target')
    return dict(controls, target_standard_error=float(target))


def _fragment_triples_controls(controls, index):
    if controls is None or 'fragment_standard_errors' not in controls:
        return controls
    local = dict(controls, target_standard_error=controls['fragment_standard_errors'][index],
                 fragment_variance_fraction=controls['fragment_variance_fractions'][index])
    if 'fragment_reference_energies' in controls:
        local['fragment_reference_energy'] = controls['fragment_reference_energies'][index]
    if 'fragment_occupied_counts' in controls:
        local['fragment_occupied_count'] = controls['fragment_occupied_counts'][index]
    return local


def _triples_fragment_seed(global_seed, fragment_index):
    """Derive a triples stream from fragment identity, independent of ownership."""
    sequence = numpy.random.SeedSequence([global_seed, 0x54524950, fragment_index])
    return int(sequence.generate_state(1, dtype=numpy.uint64)[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _SolverLIS:
    """Compact occupied/virtual LIS inputs with frozen labels kept static."""
    mo_coeff: object
    fragment_iao_coeff: object
    active_occupied_coeff: object
    active_virtual_coeff: object
    frozen: tuple

    def tree_flatten(self):
        return ((self.mo_coeff, self.fragment_iao_coeff,
                 self.active_occupied_coeff, self.active_virtual_coeff), self.frozen)

    @classmethod
    def tree_unflatten(cls, frozen, arrays):
        return cls(*arrays, frozen)


def _build_solver_lis(mf, common, static, fragment_index, matrices, frame, representative):
    from .lis import build_fragment_lis
    if not resource_profile.enabled():
        return build_fragment_lis(mf, common, static, fragment_index,
            matrices=matrices, frame=frame, frame_fragment_index=representative)
    # Differentiate only numerical outputs used by the solver. Frozen labels
    # come from the existing constructor and remain static pytree metadata.
    frame_metadata = frame._replace(occupied_coeff=None, virtual_coeff=None)
    def build(current, data, blocks, occupied, virtual):
        current_frame = frame_metadata._replace(occupied_coeff=occupied, virtual_coeff=virtual)
        lis = build_fragment_lis(current, data, static, fragment_index,
            matrices=blocks, frame=current_frame, frame_fragment_index=representative)
        return _SolverLIS(lis.mo_coeff, lis.fragment_iao_coeff,
            lis.active_occupied_coeff, lis.active_virtual_coeff,
            tuple(int(index) for index in lis.frozen))
    with resource_profile.section('stc.lis.build', frag=fragment_index+1):
        lis = profile_response('stc.lis.pullback', build, mf, common, matrices,
            frame.occupied_coeff, frame.virtual_coeff, frag=fragment_index+1)
        jax.block_until_ready(lis)
    return lis


def _solve_fragment(mf, common, static, fragment_index, matrices, group, *,
                    verbose_imp, ccsd_t, triples_controls=None, with_grad=False,
                    vvvv_direct=False, cc_solver='deterministic', stc_cc_controls=None,
                    frame=None, correlation_auxbasis=None, df_compressor=None):
    """Return MP2-LIS, CCSD and triples; stochastic calls also return diagnostics."""
    _validate_shared_frame_triples(static, ccsd_t, stc_cc_controls)
    representative = group.representative_index
    if frame is None:
        frame = build_boys_ed_frame(common, static.mp2_static, representative)
    lis = _build_solver_lis(mf, common, static.lis_static, fragment_index,
                            matrices, frame, representative)
    stochastic = triples_controls is not None and triples_controls['mode'] == 'stochastic'
    if not lis.active_occupied_coeff.shape[1] or not lis.active_virtual_coeff.shape[1]:
        zero = np.zeros((), dtype=common.s1e.dtype)
        values = (zero, zero, zero)
        return ((values, {}) if df_compressor is not None or cc_solver == 'stc' or stochastic or
                (ccsd_t and static.lis_static.smooth_frames is not None) else values)
    metadata = {'fragment_index': int(fragment_index),
                'extended_primary_domain': numpy.asarray(
                    static.mp2_static.fragments[representative].extended_atoms, dtype=numpy.int32)}
    options = basis_kwargs(correlation_auxbasis)
    options['active_nocc'] = lis.active_occupied_coeff.shape[1]
    if df_compressor is not None:
        options['df_compressor'] = df_compressor
    if cc_solver == 'stc':
        from .stc_ccsd import make_solver
        options['amplitude_solver'] = make_solver(stc_cc_controls, fragment_index=fragment_index)
    if static.lis_static.smooth_frames is not None:
        from .smooth import noncanonical_mp2
        from .smooth_triples import make_solver
        options['mp2_solver'] = noncanonical_mp2
        if resource_profile.enabled():
            options['mp2_solver'] = lambda eris, projection: profile_response(
                'solver.mp2_reference_pullback', noncanonical_mp2, eris, projection,
                frag=fragment_index+1)
        if ccsd_t:
            local_controls = dict(triples_controls)
            local_controls['global_seed'] = _triples_fragment_seed(
                local_controls.get('global_seed', 71), fragment_index)
            options['triples_solver'] = make_solver(local_controls,
                static.lis_static.smooth_grid[fragment_index],
                with_grad=with_grad)
    elif stochastic:
        raise ValueError('canonical triples sampling has been removed; rebuild smooth selections')
    result = _impurity_solve_core(
        mf, lis.mo_coeff, lis.fragment_iao_coeff, common.fock, common.s1e,
        frozen=lis.frozen, frag_prescreen=metadata, verbose_imp=verbose_imp,
        ccsd_t=ccsd_t, dcsd=False, profile_info=None,
        vvvv_direct=vvvv_direct, **options)
    if df_compressor is None:
        return result
    if 'amplitude_solver' in options or 'triples_solver' in options:
        values, diagnostics = result
    else:
        values, diagnostics = result, {}
    return values, diagnostics


def _run_group(state, static, group, baseline_controls, matrix_controls, *,
               with_grad, verbose_imp, ccsd_t, triples_controls=None,
               include_baseline=True, fragment_indices=None, vvvv_direct=False,
               cc_solver='deterministic', stc_cc_controls=None, lis_aux_svd_tol=None):
    """Optionally join all local DF producer responses for this domain."""
    from pyscfad.lno import _df_shared
    options = dict(with_grad=with_grad, verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                   triples_controls=triples_controls, include_baseline=include_baseline,
                   fragment_indices=fragment_indices, vvvv_direct=vvvv_direct,
                   cc_solver=cc_solver, stc_cc_controls=stc_cc_controls)
    if lis_aux_svd_tol:
        options['lis_aux_svd_tol'] = lis_aux_svd_tol
    # Replica averaging scales returned VJP trees. Deferred integral bars in an
    # outer shared scope would escape that scaling, so use ordinary responses.
    replica_precision = cc_solver == 'stc' and _stc_precision.enabled(stc_cc_controls)
    if not hasattr(state['mf'], 'mol') or with_grad and replica_precision:
        return _run_group_responses(state, static, group, baseline_controls,
                                    matrix_controls, **options)
    atoms = static.mp2_static.fragments[group.representative_index].extended_atoms
    with _df_shared.reverse_scope(
            state['mf'], atoms, **basis_kwargs(state.get('correlation_auxbasis'))) as shared:
        if state.get('common') is not None:
            from ._df_inputs import configure_cache
            configure_cache(shared, static)
        records = _run_group_responses(state, static, group, baseline_controls,
                                       matrix_controls, **options)
        with resource_profile.section('stc.cc.shared_df_integral_pullback',
                                      domain_representative=group.representative_index+1):
            coords = shared.finish()
            if with_grad:
                state['mf_bar'].mol.coords = state['mf_bar'].mol.coords + np.asarray(coords)
                jax.block_until_ready(state['mf_bar'].mol.coords)
        driver.lib.logger.info(state['mf'], 'Shared local DF reverse: %s', shared.receipt)
    return records


def _run_group_responses(state, static, group, baseline_controls, matrix_controls, *,
                        with_grad, verbose_imp, ccsd_t, triples_controls=None,
                        include_baseline=True, fragment_indices=None, vvvv_direct=False,
                        cc_solver='deterministic', stc_cc_controls=None, lis_aux_svd_tol=None):
    """Complete one owned group's preparation, baseline and physical-fragment responses.

    All process-local VJPs and matrix replay records are consumed here. Only
    scalar component records and the accumulated numerical host bars survive.
    One preparation pullback consumes the sum of its input cotangents and
    direct LIS frame cotangents. The domain frame and its reverse are shared
    by domain MP2/W and every physical-fragment correction in this group.
    """
    from pyscfad.lno._df_shared import release_consumed_inputs
    with resource_profile.section('stc.cc.prepare'):
        inputs, host, prepare_pullback, frame = driver._prepare(
            state, static.mp2_static, group, baseline_controls, with_grad,
            return_frame=True)
    frame_coeff = (frame.occupied_coeff, frame.virtual_coeff)
    frame_bars = tuple(np.zeros_like(coeff) for coeff in frame_coeff) if with_grad else None
    metadata = {'fragment_id': group.representative_index,
                'target_indices': tuple(group.target_indices)}
    input_bars = None
    baseline_from_w = include_baseline and static.matrix_kind == 'W'
    if include_baseline and not baseline_from_w:
        with resource_profile.section('stc.cc.baseline'):
            baseline = backend.solve(host, metadata, baseline_controls, with_grad=with_grad)
        state['energy'] = state['energy'] + baseline['energy']
        state['baseline_energy'] = state.get('baseline_energy', 0.) + float(baseline['energy'])
        state['variance'] += float(baseline['energy_standard_error']) ** 2
        if with_grad:
            validate_full_result(baseline, inputs)
            input_bars = {name: numpy.array(value, dtype=numpy.float64, copy=True)
                          for name, value in baseline['cotangents'].items()}
        del baseline
    records = []
    for index, targets in _group_cc_fragments(static, group):
        correct_fragment = fragment_indices is None or index in fragment_indices
        if not correct_fragment and not baseline_from_w:
            continue
        with resource_profile.section('stc.cc.matrix_forward', frag=index+1):
            forward = backend.solve_matrices(
                host, {'fragment_id': index, 'target_indices': targets},
                matrix_controls, matrix_kind=static.matrix_kind)
        if baseline_from_w:
            # W already supplies the deterministic or sampled MP2 energy.
            # Seed its derivative only after downstream LIS bars are known.
            state['energy'] = state['energy'] + forward['energy']
            state['baseline_energy'] = state.get('baseline_energy', 0.) + float(forward['energy'])
            state['variance'] += float(forward['diagnostics']['energy_standard_error']) ** 2
        if not correct_fragment:
            # Restricting CC corrections must still retain the complete MP2
            # baseline, including physical fragments without a selected LIS.
            if with_grad:
                with resource_profile.section('stc.cc.matrix_pullback', frag=index+1):
                    bars = backend.pullback_matrices(
                        host, forward, energy_bar=1.,
                        occupied_bar=numpy.zeros_like(host['foo']),
                        virtual_bar=numpy.zeros_like(host['fvv']))
                validate_input_cotangents(bars, inputs)
                if input_bars is None:
                    input_bars = {name: numpy.array(value, dtype=numpy.float64, copy=True)
                                  for name, value in bars.items()}
                else:
                    for name in inputs:
                        input_bars[name] += bars[name]
                del bars
            del forward
            continue
        matrices = {name: np.asarray(forward[name]) for name in ('occupied', 'virtual')}
        fragment_static = (_select_current_fragment(state, static, index, group, frame, forward)
                           if getattr(static, 'deferred_selection', None) is not None else static)
        _clear_jax_caches('before_correction', matrices, frame_coeff, frag=index+1)
        local_triples = _fragment_triples_controls(triples_controls, index)
        pilot_mf = None
        def evaluate(configuration, need_grad):
            nonlocal pilot_mf
            solver_options = (dict(cc_solver=cc_solver, stc_cc_controls=configuration)
                              if cc_solver == 'stc' else {})
            compressor = None
            if lis_aux_svd_tol:
                from ._aux_compress import make_compressor
                compressor = make_compressor(lis_aux_svd_tol)
                solver_options['df_compressor'] = compressor
            def correction(mf, common, blocks, coeff):
                current_frame = frame._replace(occupied_coeff=coeff[0], virtual_coeff=coeff[1])
                values = _solve_fragment(mf, common, fragment_static, index, blocks, group,
                                         verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                                         triples_controls=local_triples, with_grad=need_grad,
                                         vvvv_direct=vvvv_direct, frame=current_frame, **solver_options,
                                         **basis_kwargs(state.get('correlation_auxbasis')))
                if lis_aux_svd_tol or cc_solver == 'stc' or (ccsd_t and (
                        getattr(getattr(fragment_static, 'lis_static', None), 'smooth_frames', None) is not None or
                        triples_controls is not None and triples_controls['mode'] == 'stochastic')):
                    values, diagnostics = values
                else:
                    diagnostics = {}
                emp2, ecc, et = values
                return ecc + et - emp2, (values, diagnostics)
            if not need_grad:
                reference = state['mf']
                if with_grad:
                    # Plain forward pilots populate local-DF caches. Keep those
                    # mutations off the live MF whose response trees are saved.
                    if pilot_mf is None:
                        pilot_mf = jax.tree.map(lambda value: value, reference)
                        df = getattr(pilot_mf, 'with_df', None)
                        cache = getattr(df, '_lno_local_df_cache', None)
                        if cache is not None:
                            df._lno_local_df_cache = dict(cache)
                    reference = pilot_mf
                energy, auxiliary = correction(reference, state['common'], matrices, frame_coeff)
                if compressor is not None:
                    auxiliary[1]['auxiliary_compression'] = dict(compressor.diagnostics)
                return energy, auxiliary, None
            with resource_profile.section('stc.cc.correction_forward', frag=index+1):
                energy, pullback, auxiliary = jax.vjp(
                    correction, state['mf'], state['common'], matrices, frame_coeff, has_aux=True)
                jax.block_until_ready((energy, auxiliary))
            _clear_jax_caches('before_correction_pullback', energy, auxiliary, frag=index+1)
            with resource_profile.section('stc.cc.correction_pullback', frag=index+1):
                bars = pullback(np.ones_like(energy))
                jax.block_until_ready(bars)
            if compressor is not None:
                auxiliary[1]['auxiliary_compression'] = dict(compressor.diagnostics)
            return energy, auxiliary, bars
        if cc_solver == 'stc' and _stc_precision.enabled(stc_cc_controls):
            nfragments = len(_fragment_indices(static, fragment_indices))
            energy, auxiliary, correction_bars = _stc_precision.run_replicas(
                evaluate, stc_cc_controls, index, nfragments, with_grad=with_grad)
        else:
            energy, auxiliary, correction_bars = evaluate(stc_cc_controls, with_grad)
        if with_grad:
            mf_bar, common_bar, matrix_bars, coeff_bars = correction_bars
            frame_bars = tuple(left + right for left, right in zip(frame_bars, coeff_bars))
            jax.block_until_ready((mf_bar, common_bar, matrix_bars, frame_bars))
            # The direct host response holds the shared frame fixed; apply its
            # accumulated frame contribution once through preparation below.
            direct_norm = _bar_norm(mf_bar, common_bar)
            driver._add_bars(state, mf_bar, common_bar)
            # These bars have been accumulated into state/frame_bars. Release
            # their separate trees and completed CC response executables before
            # the native W reverse allocates domain-sized factor buffers.
            del correction_bars, mf_bar, common_bar, coeff_bars
            _clear_jax_caches('before_matrix_pullback', state['mf_bar'],
                              state['common_bar'], frame_bars, matrix_bars,
                              frag=index+1)
            with resource_profile.section('stc.cc.matrix_pullback', frag=index+1):
                bars = backend.pullback_matrices(
                    host, forward, energy_bar=float(baseline_from_w), occupied_bar=matrix_bars['occupied'],
                    virtual_bar=matrix_bars['virtual'])
            validate_input_cotangents(bars, inputs)
            if input_bars is None:
                # Native outputs are owned and writable, including the empty
                # fast path. Adopt them without duplicating the domain-sized
                # B bar; diagnostic wrappers may return read-only snapshots.
                input_bars = {name: value if value.flags.writeable else value.copy()
                              for name, value in bars.items()}
            else:
                for name in inputs:
                    input_bars[name] += bars[name]
            del matrix_bars, bars
        else:
            direct_norm = None
        values, diagnostics = auxiliary
        def diagnostic_value(value):
            value = numpy.asarray(value)
            return value.item() if value.ndim == 0 else value.tolist()
        diagnostics = jax.tree_util.tree_map(diagnostic_value, diagnostics)
        cc_diagnostics = diagnostics.pop('ccsd', None)
        compression_diagnostics = diagnostics.pop('auxiliary_compression', None)
        if local_triples is not None:
            diagnostics.update({key: local_triples[key] for key in (
                'target_standard_error', 'fragment_variance_fraction',
                'fragment_reference_energy', 'fragment_occupied_count',
                'total_active_occupied', 'variance_allocation') if key in local_triples})
        state['energy'] = state['energy'] + energy
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
        records.append(FragmentComponents(index, *(float(value) for value in values),
                                           float(forward['energy']),
                                           *_fragment_dimensions(fragment_static, index),
                                           direct_norm, None,
                                           {key: value for key, value in forward['diagnostics'].items()
                                            if key not in ('occupied_standard_error',
                                                           'virtual_standard_error')},
                                           triples_method=(triples_controls['mode']
                                               if ccsd_t and triples_controls is not None else
                                               'deterministic' if ccsd_t else 'none'),
                                           triples_standard_error=float(diagnostics.get('standard_error', 0.)),
                                           triples_samples=int(diagnostics.get('samples', 0)),
                                           triples_timings={key: value for key, value in diagnostics.items()
                                                            if key.endswith('_seconds')} or None,
                                           triples_diagnostics=diagnostics or None,
                                           ccsd_method=cc_solver,
                                           ccsd_diagnostics=cc_diagnostics,
                                           auxiliary_compression=compression_diagnostics))
        del forward, matrices, evaluate, pilot_mf, energy, values, auxiliary, diagnostics, fragment_static
        release_consumed_inputs(forward_only=not with_grad)
        gc.collect()
    if input_bars is not None:
        with resource_profile.section('stc.cc.input_pullback'):
            # AD adds the coefficient bars from DF/Fock construction to the
            # explicit LIS bars before traversing the frame producer once.
            combined_pullback = lambda bars: prepare_pullback((bars, frame_bars))
            driver._add_bars(state, *apply_input_pullback(combined_pullback, inputs, input_bars))
            del input_bars
            jax.block_until_ready((state['mf_bar'], state['common_bar']))
        release_consumed_inputs()
    return records


def _run_weak(state, static, *, with_grad, pairs=None):
    """Complete separately owned baseline weak terms once."""
    pairs = state['weak_pairs'] if pairs is None else pairs
    if not pairs:
        return
    if with_grad:
        energy, bars = driver._weak_energy_and_grad(state, static.mp2_static, pairs)
        driver._add_bars(state, *bars)
    else:
        energy = driver.weak_pair_energy_sum(
            state['mf'], state['common'], static.mp2_static, pairs=pairs)
    state['energy'] = state['energy'] + energy
    state['baseline_energy'] = state.get('baseline_energy', 0.) + float(energy)
    state['weak_energy'] = state.get('weak_energy', 0.) + float(energy)


def _run_serial(state, static, baseline, matrix, *, with_grad, include_hf,
                merge_identical, verbose_imp, ccsd_t, return_components,
                triples_controls=None, fragment_indices=None, include_baseline=True,
                checkpoint_dir=None, resume=False, vvvv_direct=False,
                cc_solver='deterministic', stc_cc_controls=None, lis_aux_svd_tol=None):
    from ._ccsd_checkpoint import open_checkpoint, restore_progress, save_progress
    groups = build_domain_groups(static.mp2_static, merge_identical=merge_identical)
    allocated = allocate_domain_controls(baseline, len(groups))
    options = dict(with_grad=with_grad, include_hf=include_hf, ccsd_t=ccsd_t,
                   triples_controls=triples_controls, vvvv_direct=vvvv_direct,
                   include_baseline=include_baseline, fragment_indices=fragment_indices,
                   merge_identical=merge_identical)
    solver_options = (dict(cc_solver=cc_solver, stc_cc_controls=stc_cc_controls)
                      if cc_solver == 'stc' else {})
    if lis_aux_svd_tol:
        solver_options['lis_aux_svd_tol'] = lis_aux_svd_tol
    options.update(solver_options)
    checkpoint = open_checkpoint(state, static, groups, checkpoint_dir=checkpoint_dir,
                                  resume=resume, options=options)
    try:
        start, weak_done, records = restore_progress(checkpoint, state, with_grad=with_grad)
        if weak_done or not include_baseline:
            state.pop('_prepared_weak', None)
        if not 0 <= start <= len(groups):
            raise ValueError('CC checkpoint progress is outside the saved group worklist')
        for position in range(start, len(groups)):
            group = groups[position]
            corrections = _group_cc_fragments(static, group)
            if include_baseline or any(fragment_indices is None or index in fragment_indices
                                       for index, _ in corrections):
                records.extend(_run_group(
                    state, static, group, allocated, matrix, with_grad=with_grad,
                    verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                    triples_controls=triples_controls, vvvv_direct=vvvv_direct,
                    include_baseline=include_baseline, fragment_indices=fragment_indices,
                    **solver_options))
                _clear_jax_caches('after_group', state['energy'], state['mf_bar'],
                                  state['common_bar'], frag=group.representative_index+1)
            save_progress(checkpoint, state, records, position + 1, False, with_grad=with_grad)
            gc.collect()
        if not weak_done:
            if include_baseline:
                _run_weak(state, static, with_grad=with_grad)
            save_progress(checkpoint, state, records, len(groups), True, with_grad=with_grad)
        return _finish(state, static, records, with_grad=with_grad, include_hf=include_hf,
                       return_components=return_components)
    finally:
        if checkpoint is not None:
            checkpoint.close()


def _finish(state, static, records, *, with_grad, include_hf, return_components):
    components = CCSDComponents(state.get('baseline_energy', 0.), state.get('weak_energy', 0.),
                                tuple(sorted(records, key=lambda item: item.fragment_index)),
                                static.matrix_kind, state['variance'] ** .5)
    if components.stc_standard_error is not None:
        driver.lib.logger.info(state['mf'],
            'STC final selected-correction empirical SE %.6g mEh; target %.6g mEh; met %s',
            components.stc_standard_error * 1000., components.stc_target_standard_error * 1000.,
            components.stc_target_met)
    result = driver._finish(
        state, with_grad=with_grad, include_hf=include_hf,
        uncertainty_label='MP2 baseline standard error')
    if not with_grad and include_hf:
        result = result + state['mf'].e_tot
    if return_components:
        return (*result, components) if with_grad else (result, components)
    return result


def _run(initializer, worker_initializer, comm, *, checkpoint_dir, resume,
         parallel_scf_jk=True, scf_worker_initializer=None, **options):
    if comm is not None and comm.Get_size() > 1:
        from ._mpi_ccsd import _run_mpi
        def check_options():
            if checkpoint_dir is not None or resume:
                raise NotImplementedError('CC progress checkpoints currently require serial execution')
        driver._root_call(comm, comm.Get_rank(), check_options, 'CC checkpoint option validation')
        return _run_mpi(initializer, worker_initializer, comm,
                        parallel_scf_jk=parallel_scf_jk,
                        scf_worker_initializer=scf_worker_initializer, **options)
    if not isinstance(options.get('vvvv_direct', False), bool):
        raise TypeError('vvvv_direct must be a bool')
    state, static, baseline, matrix = initializer()
    _clear_jax_caches('initialization', state.get('mf'), state.get('common'))
    options['fragment_indices'] = state.pop('cc_fragment_indices')
    options['triples_controls'] = state.pop('cc_triples_controls')
    options.update(state.pop('cc_solver_options', {}))
    return _run_serial(state, static, baseline, matrix,
                       checkpoint_dir=checkpoint_dir, resume=resume, **options)


@memory_calculation
def kernel(mf, static, *, baseline_controls, matrix_controls, matrix_kind=None,
           ccsd_t=False, triples_controls=None, verbose_imp=0, include_hf=False,
           return_components=False,
           merge_identical=True, comm=None, fragment_indices=None, include_baseline=True,
           checkpoint_dir=None, resume=False, parallel_scf_jk=True, vvvv_direct=False,
           cc_solver='deterministic', stc_cc_controls=None, correlation_auxbasis=None,
           lis_aux_svd_tol=1e-5):
    """Return correlation energy, or total energy with ``include_hf=True``.

    ``correlation_auxbasis`` selects local MP2, D/W and CC fitting; HF/Fock
    retain the SCF fit. Use the same basis for selection and execution.
    ``lis_aux_svd_tol`` compresses only LIS auxiliary factors with the strict
    squared-singular-value cutoff ``s**2 > tol`` (default ``1e-5``).
    Pass ``None`` or zero to disable compression.
    Domain MP2, D/W and orbital selection remain unchanged. Compression
    diagnostics are included in each fragment's ``auxiliary_compression``.
    ``return_components=True`` returns ``(energy, CCSDComponents)``. With W,
    strong-domain MP2 energy is read directly from its traces, in deterministic
    and stochastic modes, including fragments without a CC correction. Matrix
    controls set its precision; there is no separate strong-domain MP2 solve.
    Baseline and matrix Laplace grids must match. D retains a separate MP2
    solver. The MP2 baseline includes weak completion in either case.
    With ``ccsd_t=True``, omitted ``triples_controls`` preserves deterministic
    triples; stochastic mode requires integer ``nsamples >= 2``. A positive
    ``energy_tolerance`` targets the total triples energy SE in Hartree and
    divides variance according to ``variance_allocation``. ``occupied_count``
    uses each physical fragment's occupied targets divided by all active
    occupied orbitals (including for subset calculations); it is required by
    single-pass selection. The legacy ``mp2_energy`` allocation uses saved
    absolute MP2 fragment energies; ``fragment_energies`` overrides those
    proxies. In the smooth path, ``pilot_samples`` defaults to 10000 per
    point and determines independent production counts. In the conventional
    path it overrides the pilot count otherwise supplied by ``nsamples``.
    Omitted/zero tolerance uses fixed production counts.
    ``fragment_indices`` restricts corrections (default: all saved LIS ranks).
    The complete baseline remains included unless ``include_baseline=False``;
    that option returns only selected CC-minus-MP2-LIS corrections and forbids HF.
    Deterministic fragment RCCSD uses the in-memory PySCF/NumPy CPU path by
    default. Set ``pyscfad_dfccsd_native=False`` for the JAX reference path;
    ``vvvv_direct=False`` (default) caches the virtual ladder when it fits
    within 60% of remaining total memory, for both backends. ``True`` skips
    that cache and contracts resident DF factors in blocks. Native caches
    use PySCF's packed layout and live only for each host solve.
    Multi-rank calls distribute complete correlation groups and their
    single-pass LIS selections. Shared preparation uses distributed DF J/K by default.
    Each rank must supply a matching MF skeleton; preceding SCF is caller-owned.
    ``parallel_scf_jk=True`` also distributes common-preparation DF J/K work. Serial and
    single-rank calls keep their existing behavior.

    ``cc_solver='stc'`` selects the experimental external ``stc_cc_jax``
    amplitude solver; ``stc_cc_controls`` contains its sampling/localization
    settings (unsuffixed energy fields: Hartree; explicit ``*_mEh`` aliases:
    millihartree). All other stages retain their independent controls.
    STC defaults to ``localize=True, shared_local_frame=True``: differentiable
    averaged amplitudes, projected CCSD energy and smooth triples use one fixed
    private Boys frame with consistently transformed integrals and projector.
    Triples then requires a smooth selection. ``shared_local_frame=False``
    restores amplitude backrotation to the original LIS basis. Fragment CC
    diagnostics are separate from triples and baseline uncertainty. The default
    deterministic solver does not import the optional dependency.
    ``stc_cc_controls['energy_tolerance_mEh']`` enables independent complete-solve
    replicas, allocating equal variance across the globally selected physical LIS.
    Independent pilots choose a fixed production count per LIS. Returned energies
    and response cotangents are production means; ``components.stc_standard_error``
    is their empirical combined SE. Baseline, matrices and triples contractions
    must be deterministic. This SE excludes iteration/approximation bias and does
    not bound forces; cap/unmet-target diagnostics remain explicit.
    """
    def initialize():
        solver_options = _cc_solver_options(cc_solver, stc_cc_controls, lis_aux_svd_tol)
        triples = normalize_triples_controls(triples_controls, ccsd_t=ccsd_t)
        if include_hf and not include_baseline:
            raise ValueError('include_hf requires include_baseline=True')
        baseline, matrix = _validate_run(static, baseline_controls, matrix_controls, matrix_kind)
        _validate_shared_frame_triples(static, ccsd_t, solver_options.get('stc_cc_controls'))
        indices = _fragment_indices(static, fragment_indices)
        _stc_precision.validate_stages(solver_options.get('stc_cc_controls'), baseline, matrix, triples)
        _stc_precision.validate_seed_independence(solver_options.get('stc_cc_controls'), indices)
        triples = _allocate_triples_controls(triples, len(indices), static=static, indices=indices)
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(mf)):
            raise TypeError('STC CC kernel is forward-only; use value_and_grad')
        validate_selection_basis(mf, static, correlation_auxbasis)
        state = driver._initialize(mf, static.mp2_static, baseline, with_grad=False,
                                   **basis_kwargs(correlation_auxbasis))
        state['cc_fragment_indices'] = indices
        state['cc_triples_controls'] = triples
        if solver_options:
            state['cc_solver_options'] = solver_options
        return state, static, baseline, matrix

    def initialize_worker(canonical):
        if mf is None:
            raise ValueError('CC MPI kernel requires a matching MF skeleton on every rank')
        for name, value in canonical.items():
            setattr(mf, name, np.asarray(value))
        mf.converged = True
        return mf

    return _run(initialize, initialize_worker, comm, with_grad=False, include_hf=include_hf,
                merge_identical=merge_identical, verbose_imp=verbose_imp,
                ccsd_t=ccsd_t, return_components=return_components,
                vvvv_direct=vvvv_direct,
                fragment_indices=fragment_indices, include_baseline=include_baseline,
                checkpoint_dir=checkpoint_dir, resume=resume,
                parallel_scf_jk=parallel_scf_jk, scf_worker_initializer=lambda: mf)


@memory_calculation
def value_and_grad(mol, build_mf, static=None, *, baseline_controls, matrix_controls,
                   matrix_kind=None, ccsd_t=False, triples_controls=None,
                   verbose_imp=0, include_hf=False,
                   return_components=False, merge_identical=True, comm=None, build_static=None,
                   fragment_indices=None, include_baseline=True, checkpoint_dir=None, resume=False,
                   parallel_scf_jk=True, vvvv_direct=False,
                   cc_solver='deterministic', stc_cc_controls=None, correlation_auxbasis=None,
                   lis_aux_svd_tol=1e-5):
    """Return ``(energy, mol_bar)``; ``mol_bar.coords`` is a nuclear gradient.

    ``correlation_auxbasis`` follows :func:`kernel`; ``build_static`` inherits
    it on its detached reference. The live SCF DF object remains unchanged.
    ``lis_aux_svd_tol`` defaults to ``1e-5`` and follows :func:`kernel`,
    including the response of the retained auxiliary subspace at its fixed rank.
    Pass ``None`` or zero to disable compression.
    Include HF once with ``include_hf=True``. Append ``CCSDComponents`` with
    ``return_components=True``. A force is the negated coordinate gradient.
    ``build_static(mf)`` may select ranks once on the detached SCF reference.
    Subset and ``include_baseline`` semantics match :func:`kernel`; even a
    correction-only subset closes common-data and implicit SCF response once.
    ``triples_controls`` has the same independent sampling semantics as in
    :func:`kernel`; the reported triples SE estimates energy uncertainty only.
    ``vvvv_direct`` has the memory/performance semantics of :func:`kernel`.
    Multi-rank calls distribute complete correlation groups, including deferred
    single-pass LIS selection. The default ``parallel_scf_jk=True``
    also distributes DF J/K work in SCF, common
    preparation and their coordinate pullbacks. Worker ``build_mf`` must accept
    ``mo_coeff_init``, ``mo_energy_init``, ``mo_occ_init`` and ``e_tot_init``,
    adopt these values and skip SCF. An attached out-of-core CDERI source
    must be available on every rank. ``parallel_scf_jk=False`` retains
    root-only SCF and shared response. All root-owned calls use the same
    DF geometry fusion, response caching, opt-in resident DF buffer and
    BLAS-controller scopes as the MP2 driver;
    the complete SCF pullback must reach its DF producer. See
    README_SERIAL_DF_SCF.md for activation and prebuilt-factor ownership.

    ``cc_solver`` and ``stc_cc_controls`` match :func:`kernel`. STC response
    uses its stochastic Lambda solve at the averaged amplitudes; energy-error
    controls do not bound nuclear-gradient errors. Physical domain/LIS response
    remains active, including the original fragment projector.
    """
    def initialize():
        solver_options = _cc_solver_options(cc_solver, stc_cc_controls, lis_aux_svd_tol)
        triples = normalize_triples_controls(triples_controls, ccsd_t=ccsd_t)
        driver._coordinate_only(mol)
        if include_hf and not include_baseline:
            raise ValueError('include_hf requires include_baseline=True')
        prechecked_indices = None
        if build_static is not None:
            if static is not None:
                raise ValueError('supply either static or build_static, not both')
            if not callable(build_static):
                raise TypeError('build_static must be callable')
        else:
            early_baseline, early_matrix = _validate_run(static, baseline_controls, matrix_controls, matrix_kind)
            _validate_shared_frame_triples(static, ccsd_t, solver_options.get('stc_cc_controls'))
            _stc_precision.validate_stages(solver_options.get('stc_cc_controls'), early_baseline, early_matrix, triples)
            if _stc_precision.enabled(solver_options.get('stc_cc_controls')):
                prechecked_indices = _fragment_indices(static, fragment_indices)
                _stc_precision.validate_seed_independence(solver_options['stc_cc_controls'], prechecked_indices)
        local_response = driver._root_owned_scf_response(comm, parallel_scf_jk)
        with resource_profile.section('stc.domain.scf'):
            mf, scf_pullback = driver._initialize_scf(
                mol, build_mf, combine_df=local_response, share_scf_data=True)
            if resource_profile.enabled():
                jax.block_until_ready(mf)
        if local_response:
            scf_pullback = driver._serial_scf_pullback(scf_pullback)
        selected = static
        scf_data = getattr(scf_pullback, 'scf_data', None)
        scf_data_pullback = prepared_domain = None
        if build_static is not None:
            selected, scf_data, scf_data_pullback, prepared_domain = driver._select_domain_reference(
                mf, build_static, reuse_weak=include_baseline and not resume,
                scf_data=scf_data,
                **basis_kwargs(correlation_auxbasis))
        validate_selection_basis(mf, selected, correlation_auxbasis)
        baseline, matrix = _validate_run(selected, baseline_controls, matrix_controls, matrix_kind)
        _validate_shared_frame_triples(selected, ccsd_t, solver_options.get('stc_cc_controls'))
        indices = (_fragment_indices(selected, fragment_indices)
                   if prechecked_indices is None else prechecked_indices)
        _stc_precision.validate_stages(solver_options.get('stc_cc_controls'), baseline, matrix, triples)
        if build_static is not None:
            _stc_precision.validate_seed_independence(solver_options.get('stc_cc_controls'), indices)
        triples = _allocate_triples_controls(triples, len(indices), static=selected, indices=indices)
        state = driver._initialize(mf, selected.mp2_static, baseline, with_grad=True,
                                   scf_pullback=scf_pullback, scf_data=scf_data,
                                   scf_data_pullback=scf_data_pullback,
                                   prepared_domain=prepared_domain,
                                   **basis_kwargs(correlation_auxbasis))
        state['cc_fragment_indices'] = indices
        state['cc_triples_controls'] = triples
        if solver_options:
            state['cc_solver_options'] = solver_options
        return state, selected, baseline, matrix

    def initialize_worker(canonical):
        driver._coordinate_only(mol)
        return build_mf(mol, **{name + '_init': value for name, value in canonical.items()})

    def initialize_scf_worker():
        nao = mol.nao
        occupation = numpy.zeros(nao)
        occupation[:mol.nelectron // 2] = 2.0
        return initialize_worker({
            'mo_coeff': numpy.eye(nao), 'mo_energy': numpy.zeros(nao),
            'mo_occ': occupation, 'e_tot': numpy.asarray(0.0),
        })

    return _run(initialize, initialize_worker, comm, with_grad=True, include_hf=include_hf,
                merge_identical=merge_identical, verbose_imp=verbose_imp,
                ccsd_t=ccsd_t, return_components=return_components,
                vvvv_direct=vvvv_direct,
                fragment_indices=fragment_indices, include_baseline=include_baseline,
                checkpoint_dir=checkpoint_dir, resume=resume,
                parallel_scf_jk=parallel_scf_jk,
                scf_worker_initializer=initialize_scf_worker)
