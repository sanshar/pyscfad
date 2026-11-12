"""Domain and whole-system STC-MP2 energies and molecular gradients.

    Discrete domains, ranks, partner sets and Laplace grid are fixed. Nuclear
    coordinate gradients require coordinate-only molecule leaves and a
    positive-definite local auxiliary fitting metric.
"""

from functools import partial
from contextlib import contextmanager, nullcontext, ExitStack
import gc
import math
import warnings
from time import perf_counter

import jax
import numpy
from pyscf import df as pyscf_df, lib
from pyscfad import config_update, numpy as np
from pyscfad.ops import stop_trace
from pyscfad.lib._compilation_cache import configure_compilation_cache
from pyscfad.tools import resource_profile
from ._selection import DomainSelections, _extract_active_indices, build_domain_groups
from .domain import build_scf_data, rebuild_domain_data
from ._selection import _validate_boys_target_groups as validate_boys_target_groups
from .adjoint import (
    add_cotangent as _add_cotangent,
    fix_restart_mo_phases as _fix_restart_mo_phases,
    zero_term_cotangents as _zero_term_cotangents,
    zero_mf_cotangent as _zero_mf_cotangent,
)
from ._pairs import weak_pair_energy_and_grad, weak_pair_energy_sum, weak_pair_indices


from . import backend
from ._memory import memory_calculation
from ._workflow import check_system_memory, check_domain_memory, allocate_domain_controls
from .adjoint import apply_full_pullback, apply_selected_pullback
from .controls import DEFAULT_ENERGY_TOLERANCE, default_laplace_grid
from .domain import (
    select_system_boys_reference, select_system_virtual_anchor_columns,
)
from .parallel import run_backend
from .prepare import prepare_system_inputs, prepare_boys_group_inputs
from .protocol import ARRAY_NAMES
from ._df_basis import basis_kwargs, can_reuse_scf_df, validate_selection_basis


def _root_call(comm, rank, function, phase):
    """Complete a root-only phase and announce failure before continuing."""
    if comm is None:
        return function()
    result = None
    if rank == 0:
        try:
            result = function()
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"STC {phase} failed on root: {error}")
    return result


def _validate_static(static):
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be DomainSelections")
    if static.frozen is None:
        raise ValueError("STC requires an explicit frozen-space setting")
    nocc = len(static.active_occ_indices)
    if (static.lo_type != "boys" or len(static.fragments) != nocc
            or static.frag_lolist is None):
        raise ValueError("Boys targets must form a complete singleton partition")
    groups = validate_boys_target_groups(static.frag_lolist, nocc)
    mask = numpy.asarray(static.strong_mask, dtype=bool)
    if (mask.shape != (nocc, nocc) or not numpy.array_equal(mask, mask.T)
            or not numpy.diag(mask).all()):
        raise ValueError("strong_mask must be symmetric with every target self-strong")
    for index, fragment in enumerate(static.fragments):
        if not numpy.array_equal(fragment.iao_indices, groups[index]):
            raise ValueError("fragment target does not match the singleton partition")
        partners = numpy.asarray(fragment.strong_fragments)
        if (partners.ndim != 1 or partners.dtype.kind not in "iu"
                or not numpy.array_equal(numpy.sort(partners), numpy.flatnonzero(mask[index]))):
            raise ValueError("strong partner selections disagree with strong_mask")


def _driver_controls(controls, *, scope=None):
    """Copy root-owned options and apply high-level defaults before preparation."""
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dictionary")
    controls = controls.copy()
    if "laplace_roots" not in controls and "laplace_weights" not in controls:
        controls["laplace_roots"], controls["laplace_weights"] = default_laplace_grid()
    if controls.get("mode", "deterministic") == "stochastic" and "production_samples" not in controls:
        controls.setdefault("energy_tolerance", DEFAULT_ENERGY_TOLERANCE)
    _validate_controls(controls)
    if controls.get("canonical_fock", False):
        raise ValueError(
            "high-level STC uses full local Fock blocks; canonical_fock=True "
            "is reserved for low-level canonical reference calculations"
        )
    return controls


def _validate_controls(controls):
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dictionary")
    if "canonical_fock" in controls and not isinstance(controls["canonical_fock"], bool):
        raise TypeError("canonical_fock must be a bool")
    for name in ("laplace_roots", "laplace_weights"):
        if name not in controls:
            raise ValueError(f"controls must supply {name}")
        values = numpy.asarray(controls[name], dtype=numpy.float64)
        if values.ndim != 1 or not values.size or not numpy.isfinite(values).all():
            raise ValueError(f"{name} must be a nonempty finite vector")
    roots = numpy.asarray(controls["laplace_roots"])
    weights = numpy.asarray(controls["laplace_weights"])
    if roots.shape != weights.shape or numpy.any(roots < 0):
        raise ValueError("Laplace roots/weights must match and roots must be nonnegative")
    if controls.get("mode", "deterministic") not in ("deterministic", "stochastic"):
        raise ValueError("mode must be deterministic or stochastic")
    if "energy_tolerance" in controls:
        try:
            tolerance = float(controls["energy_tolerance"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("energy_tolerance must be positive and finite") from exc
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("energy_tolerance must be positive and finite")
    block = controls.get("virtual_block_size", 16)
    if (not isinstance(block, (int, numpy.integer))
            or isinstance(block, (bool, numpy.bool_)) or block <= 0):
        raise ValueError("virtual_block_size must be a positive integer")


def _validate_reference(mf, static, *, correlation_auxbasis=None):
    validate_selection_basis(mf, static, correlation_auxbasis)
    _system_active_indices(mf, static.frozen)
    for name, expected in zip(
        ("active_occ_indices", "active_vir_indices", "pao_projected_out_indices"),
        _extract_active_indices(mf, static.frozen),
    ):
        if not numpy.array_equal(getattr(static, name), expected):
            raise ValueError(f"saved {name} does not match this SCF/frozen setting")
    # The portable path must never silently fall back to cached AO-pair CDERI.
    try:
        from pyscf.mp.dfmp2 import _init_mp_df_eris_direct
    except ImportError as exc:
        raise RuntimeError("STC requires PySCF's integral-direct local Lov helper") from exc
    if not callable(_init_mp_df_eris_direct):
        raise RuntimeError("STC requires PySCF's integral-direct local Lov helper")


def _coordinate_only(mol):
    if any(getattr(mol, name, None) is not None for name in ("exp", "ctr_coeff", "r0")):
        raise NotImplementedError(
            "STC molecular gradients differentiate nuclear coordinates only; "
            "build mol with trace_exp=False and trace_ctr_coeff=False, and no r0 leaf"
        )


def _select_domain_reference(mf, build_static, *, reuse_weak=True,
                             correlation_auxbasis=None,
                             scf_data=None, scf_data_pullback=None):
    """Share live S/F with a callback through its detached numerical reference."""
    if scf_data is None:
        # Standalone/prebuilt references have no jointly owned SCF output tape.
        with resource_profile.section('stc.domain.scf_data_prepare'):
            scf_data, scf_data_pullback = jax.vjp(build_scf_data, mf)
            jax.block_until_ready(scf_data)
    host_data = tuple(numpy.asarray(jax.device_get(value)) for value in scf_data)

    from ._selection_response import SelectionResponse
    prepared = SelectionResponse(mf, scf_data, scf_data_pullback) if reuse_weak else None

    def select(reference):
        # Only the detached copy carries this selection-time cache. In
        # particular, it must never become metadata on the live SCF pytree.
        reference._stc_scf_data = host_data
        reference._stc_selection_response = prepared
        reference._stc_correlation_auxbasis = correlation_auxbasis
        try:
            return build_static(reference)
        finally:
            reference.__dict__.pop('_stc_scf_data', None)
            reference.__dict__.pop('_stc_selection_response', None)
            reference.__dict__.pop('_stc_correlation_auxbasis', None)

    with resource_profile.section('stc.domain.selection'):
        selected = stop_trace(select)(mf)
    return selected, scf_data, scf_data_pullback, prepared


def _build_common_with_pullback(mf, static, scf_data, scf_data_pullback):
    common, construction_pullback = jax.vjp(
        lambda current, data: rebuild_domain_data(current, static, scf_data=data),
        mf, scf_data)
    if scf_data_pullback is None:
        # Independent output seeds rejoin the single SCF tape in _finish.
        return common, construction_pullback

    def common_pullback(common_bar):
        direct, data_bar = construction_pullback(common_bar)
        via_scf_data, = scf_data_pullback(data_bar)
        return (jax.tree_util.tree_map(_add_cotangent, direct, via_scf_data),)

    return common, common_pullback


def _weak_energy_and_grad(state, static, pairs):
    """Consume a selection-time contribution, or evaluate saved-static inputs."""
    pairs = tuple(pairs)
    pending = state.pop('_prepared_weak', None)
    if pending is not None and pairs == pending[0]:
        return pending[1]
    return weak_pair_energy_and_grad(state['mf'], state['common'], static, pairs=pairs)


def _initialize(mf, static, controls, *, with_grad, scf_pullback=None,
                scf_data=None, scf_data_pullback=None, prepared_domain=None,
                correlation_auxbasis=None):
    _validate_static(static)
    _validate_controls(controls)
    _validate_reference(mf, static, **basis_kwargs(correlation_auxbasis))
    configure_compilation_cache()
    pending = None
    with resource_profile.section('stc.domain.common_prepare'):
        if with_grad:
            if scf_data is None:
                scf_data = getattr(scf_pullback, 'scf_data', None)
            if scf_data is None:
                scf_data, scf_data_pullback = jax.vjp(build_scf_data, mf)
            if (scf_data_pullback is None
                    and getattr(scf_pullback, 'scf_data', None) is None):
                raise ValueError('live S/F data require their pullback for gradients')
            adopted = None if prepared_domain is None else prepared_domain.take(static)
            if adopted is None:
                common, common_pullback = _build_common_with_pullback(
                    mf, static, scf_data, scf_data_pullback)
            else:
                common, common_pullback, pending = adopted

            mf_bar, common_bar = _zero_term_cotangents(mf, common)
        else:
            if scf_data is None:
                scf_data = getattr(mf, '_stc_scf_data', None)
            common = rebuild_domain_data(mf, static, scf_data=scf_data)
            common_pullback = mf_bar = common_bar = None
        jax.block_until_ready(common)
    return {"mf": mf, "correlation_auxbasis": correlation_auxbasis, "static": static,
            "common": common, "common_pullback": common_pullback,
            "scf_pullback": scf_pullback, "mf_bar": mf_bar, "common_bar": common_bar,
            "energy": np.zeros((), dtype=common.s1e.dtype), "variance": 0.0,
            "weak_pairs": tuple(weak_pair_indices(static)), "_prepared_weak": pending}


def _prepare(state, static, group, controls, with_grad, *, return_frame=False):
    """Prepare a domain, optionally exposing its frame and combined VJP.

    The optional fourth return value is the frame. Its occupied/virtual
    coefficients are also outputs of the saved VJP, whose seed then has shape
    ``(input_bars, (occupied_bar, virtual_bar))``. Static frame metadata never
    enters the differentiable output tree.
    """
    memory_controls = ({"mode": "deterministic",
                        "target_batch_size": controls.get("target_batch_size", 1),
                        "threads": controls.get("threads")}
                       if controls.get("mode", "deterministic") == "deterministic" else {})
    check_domain_memory(state["mf"], static, group.representative_index,
                        with_grad=with_grad,
                        virtual_block_size=controls.get("virtual_block_size", 16),
                        **memory_controls, **basis_kwargs(state.get("correlation_auxbasis")))
    if return_frame:
        prepare = lambda mf_, common_: prepare_boys_group_inputs(
            mf_, common_, static, group, return_frame=True,
            **basis_kwargs(state.get("correlation_auxbasis")))
    else:
        prepare = lambda mf_, common_: prepare_boys_group_inputs(
            mf_, common_, static, group, **basis_kwargs(state.get("correlation_auxbasis")))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="integral-direct local Lov is unavailable.*",
                                category=RuntimeWarning)
        if return_frame:
            if with_grad:
                (inputs, _), pullback, frame = jax.vjp(
                    prepare, state["mf"], state["common"], has_aux=True)
            else:
                (inputs, _), frame = prepare(state["mf"], state["common"])
                pullback = None
        elif with_grad:
            inputs, pullback = jax.vjp(prepare, state["mf"], state["common"])
        else:
            inputs, pullback = prepare(state["mf"], state["common"]), None
    jax.block_until_ready(inputs)
    if return_frame:
        return inputs, backend.host_inputs(inputs), pullback, frame
    return inputs, backend.host_inputs(inputs), pullback


def _add_bars(state, mf_bar, common_bar):
    state["mf_bar"] = jax.tree_util.tree_map(_add_cotangent, state["mf_bar"], mf_bar)
    state["common_bar"] = jax.tree_util.tree_map(_add_cotangent,
                                              state["common_bar"], common_bar)


def _finish(state, *, with_grad, include_hf,
            uncertainty_label="stochastic standard error"):
    energy = state["energy"]
    lib.logger.info(state["mf"], "STC correlation energy %.12g; %s %.6g",
                    float(energy), uncertainty_label, math.sqrt(state["variance"]))
    if not with_grad:
        jax.block_until_ready(energy)
        return energy
    with _global_df_response(state):
        mf_bar = state.pop("mf_bar")
        common_pullback = state.pop("common_pullback", None)
        common_bar = state.pop("common_bar", None)
        if common_pullback is not None:
            with resource_profile.section('stc.domain.common_pullback'):
                common_bars = common_pullback(common_bar)
                common_mf_bar = common_bars[0]
                if len(common_bars) == 2:
                    state['_scf_data_bar'] = common_bars[1]
                mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, common_mf_bar)
                jax.block_until_ready((mf_bar, common_bars[1:]))
                del common_mf_bar, common_bars
        # VJP callables retain their forward residuals even after being called.
        # Consume the common frame before allocating the SCF-response workspace.
        del common_pullback, common_bar
        state.pop("common", None)
        gc.collect()
        return _finish_response(state, energy, mf_bar, include_hf=include_hf)


@contextmanager
def _global_df_response(state):
    """Share root-owned common-Fock/SCF geometry work and response headroom."""
    from pyscfad.df import _ooc_cotangent
    from pyscfad.lib import _scf_response_cache
    from pyscfad.lib._threading import fixed_threads
    from pyscfad.sgx import _shared as sgx_shared
    obj = getattr(state['mf'], 'with_df', None)
    roots, shared = [], None
    with ExitStack() as stack:
        stack.enter_context(fixed_threads())
        sgx = stack.enter_context(sgx_shared.producer_scope(state['mf']))
        if obj is not None and getattr(obj, '_ooc_geometry_producer', None) is not None:
            roots = stack.enter_context(_scf_response_cache.record_response_caches())
            stack.enter_context(_scf_response_cache.memory_scope(obj))
            shared = stack.enter_context(_ooc_cotangent.producer_scope(obj))
        yield
    if sgx is not None:
        lib.logger.info(state['mf'], 'Shared SGX SCF reverse: %s', sgx.stats)
    if shared is not None:
        lib.logger.info(state['mf'], 'Shared global SCF DF reverse: %s', shared.stats)
    for receipt in roots:
        if receipt['occupied_transform_calls'] or receipt['preconditioner_cache_bytes']:
            lib.logger.info(state['mf'], 'SCF response factor cache: %s', receipt)


def _finish_response(state, energy, mf_bar, *, include_hf):
    """Add HF once, then apply the single implicit SCF response."""
    if include_hf:
        hf_energy = state["mf"].e_tot
        mf_bar.e_tot = _add_cotangent(mf_bar.e_tot, np.ones_like(hf_energy))
        energy = energy + hf_energy
    scope = 'system' if 'fock_from_scf' in state else 'domain'
    with resource_profile.section(f'stc.{scope}.scf_response'):
        scf_pullback = state.pop("scf_pullback")
        data_bar = state.pop('_scf_data_bar', None)
        mol_bar, = (scf_pullback(mf_bar) if data_bar is None
                    else scf_pullback(mf_bar, data_bar))
        jax.block_until_ready((energy, mol_bar))
        del scf_pullback
        gc.collect()
    return energy, mol_bar


@contextmanager
def _mp2_df_scope(state, atoms=None):
    """Retain the metric/tape and, for domains, the configured H cache."""
    from pyscfad.lno import _df_shared
    mf = state['mf']
    if not hasattr(mf, 'mol'):
        yield None
        return
    basis = state.get('correlation_auxbasis')
    domain_scope = atoms is not None
    if atoms is None:
        if can_reuse_scf_df(mf, basis):
            yield None
            return
        atoms = numpy.arange(mf.mol.natm, dtype=numpy.int32)
    with _df_shared.reverse_scope(mf, atoms, **basis_kwargs(basis)) as shared:
        if domain_scope:
            from ._df_inputs import configure_cache
            configure_cache(shared)
        yield shared


def _finish_mp2_df(shared, mf_bar):
    """Consume deferred integral response once, before the final SCF response."""
    if shared is not None:
        # The coefficient VJP has consumed H. Geometry uses the saved MO
        # adjoints and coefficient snapshots, so release H before its scratch.
        if shared.half is not None:
            shared.half.release_consumed()
        with resource_profile.section('stc.mp2.shared_df_integral_pullback'):
            coords = shared.finish()
            if mf_bar is not None:
                mf_bar.mol.coords += np.asarray(coords)
                jax.block_until_ready(mf_bar.mol.coords)
    return mf_bar


def _run_domain(initializer, static, controls, comm, *, with_grad, include_hf=False,
                merge_identical=True):
    rank = 0 if comm is None else comm.Get_rank()
    with resource_profile.section('stc.domain.initialize'):
        state = _root_call(comm, rank, initializer, "initial preparation")
    if rank == 0 and static is None:
        static = state["static"]
    groups = _root_call(comm, rank,
                        lambda: build_domain_groups(static, merge_identical=merge_identical),
                        "group selection")
    pairs = state["weak_pairs"] if rank == 0 else None
    if comm is not None:
        groups, pairs = comm.bcast((groups, pairs), root=0)
    settings = allocate_domain_controls(controls, len(groups)) if rank == 0 else None
    report = rank == 0 and getattr(state["mf"], "verbose", 0) >= lib.logger.INFO
    if report:
        lib.logger.info(state["mf"], "Domain MP2: domains=%d, weak pairs=%d", len(groups), len(pairs))
    for number, group in enumerate(groups, 1):
        started = perf_counter()
        fragment_id = group.representative_index
        with ExitStack() as resources:
            shared = _root_call(comm, rank, lambda: resources.enter_context(
                _mp2_df_scope(state, static.fragments[fragment_id].extended_atoms)),
                f"group {fragment_id} DF setup")
            if report:
                fragment = static.fragments[fragment_id]
                lib.logger.info(state["mf"],
                    "Domain %d/%d: nao=%d nocc=%d nvirt=%d naux=%d targets=%d",
                    number, len(groups), shared.mol.nao,
                    len(fragment.strong_occ_metric_keep), len(fragment.virtual_anchor_columns),
                    shared.auxmol.nao, len(group.target_indices))
            with resource_profile.section('stc.domain.prepare'):
                prepared = _root_call(comm, rank,
                    lambda: _prepare(state, static, group, controls, with_grad),
                    f"group {fragment_id} preparation")
            prepared_at = perf_counter()
            metadata = ({"fragment_id": fragment_id, "target_indices": group.target_indices,
                         "fragment_indices": group.fragment_indices,
                         "global_target_orbitals": tuple(int(static.frag_lolist[i][0])
                                                         for i in group.fragment_indices),
                         "budget": {"total_tolerance": controls.get("energy_tolerance"),
                                    "effective_tolerance": settings.get("energy_tolerance"),
                                    "original_target_count": len(static.fragments),
                                    "group_count": len(groups)}} if rank == 0 else None)
            with resource_profile.section('stc.domain.native'):
                result = run_backend(prepared[1] if rank == 0 else None, metadata, settings,
                                     partial(backend.solve, with_grad=with_grad), comm=comm)
            solved_at = perf_counter()

            def apply_group():
                if with_grad:
                    _add_bars(state, *apply_selected_pullback(prepared[2], prepared[0], result))
                state["mf_bar"] = _finish_mp2_df(shared, state["mf_bar"])
                state["energy"] = state["energy"] + result["energy"]
                state["variance"] += float(result["energy_standard_error"]) ** 2
                jax.block_until_ready((state["energy"], state["mf_bar"], state["common_bar"]))

            with resource_profile.section('stc.domain.input_pullback'):
                _root_call(comm, rank, apply_group, f"group {fragment_id} pullback")
            response_at = perf_counter()
            domain_energy = float(result["energy"]) if report else None
        del prepared, result, apply_group
        gc.collect()
        if report:
            response = ("; local response=%.2f s" % (response_at-solved_at)
                        if with_grad else "")
            lib.logger.info(state["mf"],
                "Domain %d/%d: E_corr(strong)=%.12f Ha; prepare=%.2f s; %s=%.2f s%s; total=%.2f s",
                number, len(groups), domain_energy, prepared_at-started,
                "solve+adjoint" if with_grad else "solve", solved_at-prepared_at,
                response, perf_counter()-started)
    if pairs:
        weak_started = perf_counter()
        if report:
            lib.logger.info(state["mf"], "Evaluating %d weak pairs...", len(pairs))
        def apply_weak():
            if with_grad:
                energy, bars = _weak_energy_and_grad(state, static, pairs)
                _add_bars(state, *bars)
            else:
                energy = weak_pair_energy_sum(
                    state["mf"], state["common"], static, pairs=pairs)
            state["energy"] = state["energy"] + energy
            jax.block_until_ready((state["energy"], state["mf_bar"], state["common_bar"]))
            if report:
                lib.logger.info(state["mf"], "Weak-pair E_corr=%.12f Ha; final weak stage=%.2f s",
                                float(energy), perf_counter()-weak_started)
        with resource_profile.section('stc.domain.weak'):
            _root_call(comm, rank, apply_weak, "weak endpoint response")
        del apply_weak
        if not with_grad:
            # The gradient helper already collects after releasing its tapes.
            gc.collect()
    final_started = perf_counter()
    if report and with_grad:
        lib.logger.info(state["mf"], "Final common/orbital/SCF gradient response...")
    with resource_profile.section('stc.domain.response'):
        result = _root_call(comm, rank, lambda: _finish(state, with_grad=with_grad,
                           include_hf=include_hf), "final response")
    if report and with_grad:
        lib.logger.info(state["mf"], "Final gradient response completed in %.2f s",
                        perf_counter()-final_started)
    return result


def _select_scope(static, scope, frozen, lo_kwargs=None):
    """Validate root-owned call options before any scientific scope branch."""
    if scope not in ("domain", "system"):
        raise ValueError("scope must be domain or system")
    if scope == "domain" and frozen is not None:
        raise ValueError("domain frozen setting must come from static; frozen must be None")
    if scope == "domain" and lo_kwargs is not None:
        raise ValueError("domain localization options must come from static.lo_kwargs")
    if scope == "system" and static is not None:
        raise ValueError("system scope requires static=None")
    return scope


def _shared_scope(static, scope, frozen, comm, lo_kwargs=None):
    rank = 0 if comm is None else comm.Get_rank()
    selected = _root_call(comm, rank, lambda: _select_scope(static, scope, frozen, lo_kwargs),
                          "scope selection")
    return selected if comm is None else comm.bcast(selected, root=0)


def _system_active_indices(mf, frozen):
    """Validate restricted DF and frozen spaces before inherited selection."""
    if getattr(mf, "with_df", None) is None:
        raise ValueError("system STC requires a density-fitted SCF object")
    if not getattr(mf, "converged", False):
        raise ValueError("system STC requires a converged SCF reference")
    coeff, occ = numpy.asarray(mf.mo_coeff), numpy.asarray(mf.mo_occ)
    if numpy.iscomplexobj(coeff) or numpy.iscomplexobj(occ):
        raise NotImplementedError("system STC supports real orbitals only")
    if (coeff.ndim != 2 or occ.ndim != 1 or occ.size != coeff.shape[1]
            or not numpy.all((numpy.abs(occ) < 1e-12) | (numpy.abs(occ - 2) < 1e-12))):
        raise NotImplementedError("system STC supports restricted closed-shell references only")
    if not numpy.isfinite(coeff).all() or not numpy.isfinite(occ).all():
        raise ValueError("system SCF orbitals and occupations must be finite")
    noccupied, nmo = int(numpy.count_nonzero(occ > 1)), occ.size
    if frozen is None:
        frozen = 0
    if isinstance(frozen, (int, numpy.integer)) and not isinstance(frozen, (bool, numpy.bool_)):
        if not 0 <= frozen <= noccupied:
            raise ValueError("frozen integer must lie between zero and occupied count")
        frozen = int(frozen)
    elif isinstance(frozen, (list, tuple, numpy.ndarray)):
        values = list(frozen)
        if any(not isinstance(i, (int, numpy.integer)) or isinstance(i, (bool, numpy.bool_))
               for i in values):
            raise ValueError("frozen list must contain integer MO indices")
        if any(i < 0 or i >= nmo for i in values) or len(set(values)) != len(values):
            raise ValueError("frozen MO indices must be unique and within the MO range")
        frozen = [int(i) for i in values]
    else:
        raise TypeError("frozen must be None, an integer, or an integer MO index list")
    occupied, virtual, _ = _extract_active_indices(mf, frozen)
    return occupied, virtual


class _JointSCFPullback:
    """One immutable SCF tape accepting orbital/energy and S/F output seeds."""

    def __init__(self, pullback, scf_data):
        self.pullback = pullback
        self.scf_data = scf_data

    def __call__(self, mf_bar, data_bar=None):
        if data_bar is None:
            data_bar = jax.tree_util.tree_map(np.zeros_like, self.scf_data)
        return self.pullback((mf_bar, data_bar))


def _initialize_scf(mol, build_mf, *, fix_phases=True, combine_df=False,
                    share_scf_data=False):
    """Build one implicit SCF VJP, optionally exposing its final live S/F.

    Domain drivers request joint outputs so the common-frame cotangents can
    join the original SCF tape. Builders that adopt a prebuilt reference or
    use a different kernel retain the ordinary separate S/F response path.
    """
    _coordinate_only(mol)
    if not callable(build_mf):
        raise TypeError("build_mf must be callable")
    from pyscfad.df._ooc_cotangent import capture_producers
    from pyscfad.sgx import _shared as sgx_shared
    with (capture_producers(enabled=combine_df),
          sgx_shared.capture_producers(enabled=combine_df),
          config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        build = (lambda mol_: _fix_restart_mo_phases(build_mf(mol_))) if fix_phases else build_mf
        if not share_scf_data:
            mf, pullback = jax.vjp(build, mol)
            return mf, _sgx_scf_pullback(mf, pullback)
        from pyscfad.scf._response_data import capture

        def build_with_data(current):
            with capture() as records:
                mf = build(current)
                saved = records.get(id(mf))
                data = saved[1] if saved is not None and saved[0] is mf else None
            return mf, data

        (mf, data), pullback = jax.vjp(build_with_data, mol)
        return mf, _sgx_scf_pullback(mf, _JointSCFPullback(pullback, data))


def _sgx_scf_pullback(mf, pullback):
    """Own SGX accumulation when no wider common-Fock reverse owns it."""
    if getattr(mf, '_sgx_backend', None) is None:
        return pullback
    from pyscfad.sgx import _shared
    def apply(cotangent, data_bar=None):
        with _shared.producer_scope(mf):
            return (pullback(cotangent) if data_bar is None
                    else pullback(cotangent, data_bar))
    apply.scf_data = getattr(pullback, 'scf_data', None)
    return apply


def _root_owned_scf_response(comm, parallel_scf_jk, *, scope='domain'):
    """Whether the complete shared reverse runs without a distributed J/K service."""
    return (scope == 'system' or comm is None or comm.Get_size() <= 1
            or not parallel_scf_jk)


def _serial_scf_pullback(pullback):
    """Use existing DF buffers/thread scopes for one complete root-owned reverse.

    Producer registration remains opt-in before tracing. An enclosing caller
    scope owns its finalization; otherwise this pullback must include the DF
    producer. Standalone/partial VJPs and distributed J/K keep their existing paths.
    """
    from pyscfad.lib import _cotangent_buffer
    from pyscfad.lib._threading import dense_blas_scope, fixed_threads

    def apply(cotangent, data_bar=None):
        shared = (nullcontext() if _cotangent_buffer.is_active()
                  else _cotangent_buffer.reverse_scope())
        with shared, dense_blas_scope(), fixed_threads():
            result = (pullback(cotangent) if data_bar is None
                      else pullback(cotangent, data_bar))
            jax.block_until_ready(result)
        return result

    apply.scf_data = getattr(pullback, 'scf_data', None)
    return apply


def _initialize_system(mf, frozen, controls, *, with_grad, scf_pullback=None,
                       lo_kwargs=None, fock_from_scf=False, correlation_auxbasis=None):
    _validate_controls(controls)
    occupied, virtual = _system_active_indices(mf, frozen)
    requested_fock_from_scf = fock_from_scf
    if fock_from_scf and len(occupied) and len(virtual):
        energies = numpy.asarray(mf.mo_energy)
        if (energies.shape != (mf.mo_coeff.shape[1],) or numpy.iscomplexobj(energies)
                or not numpy.isfinite(energies).all()):
            raise ValueError("SCF Fock projection requires real finite orbital energies "
                             "matching the canonical coefficient columns")
        # SCF's generalized-eigh response masks rotations at gaps <= 1e-9.
        # Reconstructing F from those outputs would lose off-diagonal Fock
        # response inside a degenerate active block. Select the general path
        # outside the AD tape, using the same cutoff as the SCF eigensolver.
        if any(numpy.any(numpy.diff(numpy.sort(energies[indices])) <= 1e-9)
               for indices in (occupied, virtual)):
            fock_from_scf = False
    resource_profile.checkpoint('stc.system.fock_state',
        requested_fock_from_scf=requested_fock_from_scf, fock_from_scf=fock_from_scf,
        degeneracy_fallback=requested_fock_from_scf and not fock_from_scf)
    dtype = mf.mo_coeff.dtype
    return {"mf": mf, "correlation_auxbasis": correlation_auxbasis,
            "occupied": occupied, "virtual": virtual,
            "lo_kwargs": lo_kwargs, "fock_from_scf": fock_from_scf,
            "scf_pullback": scf_pullback, "energy": np.zeros((), dtype=dtype),
            "variance": 0.0,
            "mf_bar": (_zero_mf_cotangent(mf, dtype)
                       if with_grad and not (len(occupied) and len(virtual)) else None)}


def _validate_canonical_system(mf):
    """Validate the converged canonical eigenproblem outside the AD tape."""
    coeff = numpy.asarray(mf.mo_coeff)
    energies = numpy.asarray(mf.mo_energy)
    if energies.shape != (coeff.shape[1],) or not numpy.isfinite(energies).all():
        raise ValueError("canonical_fock requires finite canonical orbital energies")
    fock = numpy.asarray(mf.get_fock())
    residual = float(numpy.max(numpy.abs(coeff.T @ fock @ coeff - numpy.diag(energies))))
    grad_tolerance = mf.conv_tol_grad
    if grad_tolerance is None:
        energy_tolerance = float(mf.conv_tol)
        if not math.isfinite(energy_tolerance) or energy_tolerance <= 0:
            raise ValueError("canonical_fock requires a finite positive SCF energy tolerance")
        grad_tolerance = math.sqrt(energy_tolerance)
    grad_tolerance = float(grad_tolerance)
    if not math.isfinite(grad_tolerance) or grad_tolerance <= 0:
        raise ValueError("canonical_fock requires a finite positive SCF gradient tolerance")
    residual_limit = max(1e-7, grad_tolerance)
    if not numpy.isfinite(residual) or residual > residual_limit:
        raise ValueError(
            "canonical_fock requires canonical SCF orbitals and matching energies; "
            f"max |C.T F C - diag(eps)| = {residual:.3g} Eh "
            f"exceeds {residual_limit:.3g} Eh")


def _prepare_system(state, controls, with_grad):
    memory_controls = ({"mode": "deterministic",
                        "target_batch_size": controls.get("target_batch_size", 1),
                        "threads": controls.get("threads")}
                       if controls.get("mode", "deterministic") == "deterministic" else {})
    check_system_memory(state["mf"], len(state["occupied"]), len(state["virtual"]),
                        with_grad=with_grad,
                        virtual_block_size=controls.get("virtual_block_size", 16),
                        auxiliary_group_size=controls.get("auxiliary_group_size", 32),
                        **memory_controls, **basis_kwargs(state.get("correlation_auxbasis")))
    fitting = state["mf"].with_df
    basis = state.get("correlation_auxbasis")
    if can_reuse_scf_df(state["mf"], basis):
        # Eigenvalue fallback drops auxiliary rows. Check an existing source's
        # rank without rebuilding the metric solely for validation.
        auxmol = fitting.auxmol
        if auxmol is None:
            auxmol = pyscf_df.addons.make_auxmol(state["mf"].mol.to_pyscf(), fitting.auxbasis)
        if fitting.get_naoaux() != auxmol.nao:
            raise NotImplementedError(
                "system STC requires a full-rank cached factor for a positive-definite "
                "auxiliary fitting metric; cached and auxiliary dimensions differ")
        # Resolve source compatibility on the concrete reference. Separate
        # geometry leaves become distinct tracers inside the preparation VJP.
        basis = None
    # Save discrete labels before building the VJP; rebuild all continuous
    # coefficients inside it. System scope needs no domain selections.
    state["boys_reference"] = select_system_boys_reference(
        state["mf"], state["occupied"], lo_kwargs=state["lo_kwargs"],
    )
    state["virtual_anchor_columns"] = select_system_virtual_anchor_columns(
        state["mf"], state["virtual"],
    )
    prepare = lambda mf_: prepare_system_inputs(
        mf_, state["occupied"], state["virtual"],
        boys_reference=state["boys_reference"],
        virtual_anchor_columns=state["virtual_anchor_columns"],
        fock_from_scf=state["fock_from_scf"], **basis_kwargs(basis),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="integral-direct local Lov is unavailable.*",
                                category=RuntimeWarning)
        if with_grad:
            inputs, pullback = jax.vjp(prepare, state["mf"])
        else:
            inputs, pullback = prepare(state["mf"]), None
    return inputs, backend.host_inputs(inputs, scope="system"), pullback


def _run_system(initializer, controls, comm, *, with_grad, include_hf=False):
    rank = 0 if comm is None else comm.Get_rank()
    with resource_profile.section('stc.system.scf'):
        state = _root_call(comm, rank, initializer, "initial system preparation")
        if rank == 0 and resource_profile.enabled():
            jax.block_until_ready((state["mf"], state["scf_pullback"]))
    has_work = bool(len(state["occupied"]) and len(state["virtual"])) if rank == 0 else None
    if comm is not None:
        has_work = comm.bcast(has_work, root=0)
    if has_work:
        with ExitStack() as resources:
            shared = _root_call(comm, rank, lambda: resources.enter_context(
                _mp2_df_scope(state)), "system DF setup")
            with resource_profile.section('stc.system.prepare'):
                prepared = _root_call(comm, rank, lambda: _prepare_system(state, controls, with_grad),
                                      "system preparation")
            with resource_profile.section('stc.system.native'):
                result = run_backend(
                    prepared[1] if rank == 0 else None, {} if rank == 0 else None,
                    controls if rank == 0 else None,
                    partial(backend.solve, with_grad=with_grad, scope="system"),
                    comm=comm, array_names=ARRAY_NAMES,
                )

            def apply_system():
                if with_grad:
                    # Validate the existing host mirror without compiling many
                    # bounded JAX slices solely for repeated finite-value checks.
                    state["mf_bar"], = apply_full_pullback(prepared[2], prepared[1], result)
                state["mf_bar"] = _finish_mp2_df(shared, state["mf_bar"])
                state["energy"] = state["energy"] + result["energy"]
                state["variance"] = float(result["energy_standard_error"]) ** 2
                jax.block_until_ready((state["energy"], state["mf_bar"]))

            with resource_profile.section('stc.system.input_pullback'):
                _root_call(comm, rank, apply_system, "system pullback")
        del prepared, result, apply_system
        gc.collect()
    with resource_profile.section('stc.system.response'):
        return _root_call(comm, rank, lambda: _finish(state, with_grad=with_grad,
                                                    include_hf=include_hf), "final response")


@memory_calculation
def kernel(mf, static=None, *, scope="domain", frozen=None, controls, comm=None,
           lo_kwargs=None, merge_identical=True, parallel_scf_jk=True,
           correlation_auxbasis=None):
    """Return correlation energy on root.

    ``correlation_auxbasis`` overrides local correlation fitting only. Its
    default ``None`` retains the existing SCF fitting configuration/workflow.
    Domain jobs share identical saved frames by default. ``merge_identical=False``
    runs each singleton separately for diagnostic comparisons.
    With a multi-rank domain communicator, each rank supplies a matching MF
    object; workers may use an unconverged skeleton. Root's canonical state
    is installed on workers without SCF. Only root needs selections/controls.
    The default ``parallel_scf_jk=True`` distributes common-preparation DF J/K
    work for multi-rank domain calls; ``False`` keeps this work on root.
    SCF preceding this energy-only API is caller-owned.
    System, serial and single-rank calls retain their existing behavior.
    """
    scope = _shared_scope(static, scope, frozen, comm, lo_kwargs)
    rank = 0 if comm is None else comm.Get_rank()
    controls = _root_call(comm, rank, lambda: _driver_controls(controls, scope=scope),
                          "control validation")

    def initialize():
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(mf)):
            raise TypeError("STC kernel is forward-only; use value_and_grad")
        if scope == "system":
            return _initialize_system(mf, frozen, controls, with_grad=False,
                                      lo_kwargs=lo_kwargs, **basis_kwargs(correlation_auxbasis))
        return _initialize(mf, static, controls, with_grad=False, **basis_kwargs(correlation_auxbasis))

    if scope == "system":
        return _run_system(initialize, controls, comm, with_grad=False)
    if comm is not None and comm.Get_size() > 1:
        from ._mpi import _run_domain_mpi

        def initialize_worker(canonical):
            if mf is None:
                raise ValueError("domain MPI kernel requires an MF skeleton on every rank")
            for name, value in canonical.items():
                setattr(mf, name, np.asarray(value))
            mf.converged = True
            return mf

        return _run_domain_mpi(initialize, initialize_worker, static, controls, comm,
                               with_grad=False, merge_identical=merge_identical,
                               parallel_scf_jk=parallel_scf_jk,
                               scf_worker_initializer=lambda: mf)
    return _run_domain(initialize, static, controls, comm, with_grad=False,
                       merge_identical=merge_identical)


@memory_calculation
def value_and_grad(mol, build_mf, static=None, *, scope="domain", frozen=None,
                   controls, comm=None, include_hf=False, lo_kwargs=None, merge_identical=True,
                   build_static=None, parallel_scf_jk=True, correlation_auxbasis=None):
    """Return ``(energy, mol_bar)`` on root; worker ranks return ``None``.

    ``correlation_auxbasis`` selects local correlation fitting independently
    of HF/Fock/SCF response. ``None`` preserves the existing fitting workflow.
    A selection callback inherits the explicit correlation basis.
    Discrete selections and the Laplace grid are fixed. System scope fixes
    Boys reference labels and complete PAO virtual anchors, while coordinate
    response includes localization, projections, Cholesky orthonormalization,
    full Fock blocks, fitted factors and SCF.
    For system scope, ``build_mf`` must return converged canonical orbitals
    and matching orbital energies without subsequent orbital rotations.
    Local Fock blocks use these differentiable energies and rotations to
    avoid a separate AO Fock construction and its preparation pullback.
    Degenerate active occupied or virtual spectra use AO Fock preparation
    because the SCF eigensolver masks their internal rotation response.
    System ``lo_kwargs`` are passed to the shared Boys localization; domain
    calls take their saved localization options from ``static``.
    Domain calls may instead supply ``build_static(mf)``. It runs once on
    a detached, phase-fixed reference after the single differentiated SCF,
    and returns fixed ``DomainSelections``. AO overlap/Fock data are built
    once and shared by selection and differentiable common preparation.
    Selection-only mutations cannot
    change the live SCF tree retained by its pullback. ``static`` and
    ``build_static`` are mutually exclusive; system scope uses neither.
    Identical domain frames share one selected solve/VJP unless the diagnostic
    option ``merge_identical=False`` is supplied. Public targets stay singleton.
    Stochastic energy uncertainty does not bound gradient uncertainty.
    Multi-rank domain calls distribute whole groups and weak endpoints.
    Worker ``build_mf`` must accept ``mo_coeff_init``, ``mo_energy_init``,
    ``mo_occ_init`` and ``e_tot_init``, adopt their supplied values, and skip
    SCF. Workers may pass ``None`` for selections and controls; ``build_static``
    runs only on root. Results are returned only on communicator rank zero.
    The default ``parallel_scf_jk=True`` distributes DF J/K contractions in SCF,
    common-data preparation and their coordinate pullbacks.
    With this option, workers first build an unrun reference with placeholder
    canonical arrays to prepare compatible global DF factors, then adopt
    root's converged state.
    An attached out-of-core CDERI source must be available on every rank.
    ``parallel_scf_jk=False`` retains root-only SCF and common response.
    System, serial and single-rank calls never activate this executor.
    All root-owned calls, including MPI fragment execution, scope the complete
    SCF pullback for DF geometry fusion, response caching, resident DF buffer
    sharing and dense-BLAS controller reuse. Enable
    PYSCFAD_DF_SHARED_COTANGENT before tracing build_mf to use shared buffers;
    build_mf must include their DF producer. A caller-owned complete reverse
    scope is reused. See README_SERIAL_DF_SCF.md for staged/prebuilt DF cases.
    """
    scope = _shared_scope(static, scope, frozen, comm, lo_kwargs)
    rank = 0 if comm is None else comm.Get_rank()
    controls = _root_call(comm, rank, lambda: _driver_controls(controls, scope=scope),
                          "control validation")

    def initialize():
        _coordinate_only(mol)
        if build_static is not None:
            if scope != "domain":
                raise ValueError("build_static is only supported for domain scope")
            if static is not None:
                raise ValueError("supply either static or build_static, not both")
            if not callable(build_static):
                raise TypeError("build_static must be callable")
        elif scope == "domain":
            _validate_static(static)
        local_response = _root_owned_scf_response(comm, parallel_scf_jk, scope=scope)
        with (resource_profile.section('stc.domain.scf') if scope == 'domain' else nullcontext()):
            mf, scf_pullback = _initialize_scf(
                mol, build_mf, combine_df=local_response, share_scf_data=scope == 'domain')
        if local_response:
            scf_pullback = _serial_scf_pullback(scf_pullback)
        if scope == "system":
            return _initialize_system(mf, frozen, controls, with_grad=True,
                                      scf_pullback=scf_pullback, lo_kwargs=lo_kwargs,
                                      fock_from_scf=True, **basis_kwargs(correlation_auxbasis))
        selected = static
        scf_data = getattr(scf_pullback, 'scf_data', None)
        scf_data_pullback = prepared_domain = None
        if build_static is not None:
            # Distributed MP2 already owns endpoint work across ranks. Preserve
            # that executor; CCSD's root-owned weak completion can reuse selection.
            selected, scf_data, scf_data_pullback, prepared_domain = _select_domain_reference(
                mf, build_static, reuse_weak=comm is None or comm.Get_size() == 1,
                scf_data=scf_data,
                **basis_kwargs(correlation_auxbasis))
        return _initialize(mf, selected, controls, with_grad=True,
                           scf_pullback=scf_pullback, scf_data=scf_data,
                           scf_data_pullback=scf_data_pullback,
                           prepared_domain=prepared_domain, **basis_kwargs(correlation_auxbasis))

    if scope == "system":
        return _run_system(initialize, controls, comm, with_grad=True, include_hf=include_hf)
    if comm is not None and comm.Get_size() > 1:
        from ._mpi import _run_domain_mpi

        def initialize_worker(canonical):
            _coordinate_only(mol)
            try:
                return build_mf(mol, **{name + "_init": value
                                        for name, value in canonical.items()})
            except TypeError as exc:
                raise TypeError(
                    "domain MPI worker build_mf must accept mo_coeff_init, "
                    "mo_energy_init, mo_occ_init, and e_tot_init without running SCF"
                ) from exc

        def initialize_scf_worker():
            nao = mol.nao
            occupation = numpy.zeros(nao)
            occupation[:mol.nelectron // 2] = 2.0
            return initialize_worker({
                "mo_coeff": numpy.eye(nao), "mo_energy": numpy.zeros(nao),
                "mo_occ": occupation, "e_tot": numpy.asarray(0.0),
            })

        return _run_domain_mpi(initialize, initialize_worker, static, controls, comm,
                               with_grad=True, include_hf=include_hf,
                               merge_identical=merge_identical,
                               parallel_scf_jk=parallel_scf_jk,
                               scf_worker_initializer=initialize_scf_worker)
    return _run_domain(initialize, static, controls, comm, with_grad=True, include_hf=include_hf,
                       merge_identical=merge_identical)
