# Copyright 2023-2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from mpi4py import MPI
import numpy
import jax
import jax.numpy as jnp

from pyscfad import numpy as np
from pyscfad.ops import stop_trace
from pyscfad.lno import lno_base_mpi as lno_base_mpi_mod
from pyscfad.lno import lno_base
from pyscfad.lno.ccsd import LNOCCSD, _impurity_solve_core
from pyscfad.dlno import ccsd as dlno_ccsd
from pyscfad.dlno.ccsd import (
    DLNOCCSD as _DLNOCCSDSingle,
    _add_cotangent,
    _build_static_dlno_topology,
    _cleanup_after_fragment,
    _make_stub_eris,
    _VERBOSE_PROGRESS,
)


def _path_key(path):
    """Stable, picklable string key for a jax tree-path."""
    return jax.tree_util.keystr(path)


def _to_numpy_leaf(leaf):
    if leaf is None:
        return None
    if hasattr(leaf, 'dtype') and leaf.dtype == jax.dtypes.float0:
        return leaf
    try:
        return numpy.array(numpy.asarray(leaf), copy=True, order='C')
    except (TypeError, ValueError):
        return leaf


def _tree_sum_to_root(comm, tree, root=0):
    """Sum a JAX pytree across MPI ranks into ``root`` rank.

    Each rank flattens with ``tree_flatten_with_path``; leaves are
    keyed by ``keystr`` so cross-rank alignment doesn't depend on
    object identity of registered pytree nodes (Mole, mf, with_df,
    ...).  Returns the summed pytree on ``root`` (rebuilt with its own
    treedef) and ``None`` on non-root.  Non-numeric leaves on root
    pass through; on non-root they're discarded.
    """
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    if nproc == 1:
        return tree
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: x is None,
    )
    local_paths = [_path_key(p) for p, _ in leaves_with_path]
    local_arrays = [_to_numpy_leaf(l) for _, l in leaves_with_path]
    if len(set(local_paths)) != len(local_paths):
        from collections import Counter
        c = Counter(local_paths)
        dups = {k: n for k, n in c.items() if n > 1}
        raise RuntimeError(
            f'tree_sum_to_root: rank {rank} has duplicate path keys: '
            f'{dict(list(dups.items())[:5])}'
        )
    local_dict = dict(zip(local_paths, local_arrays))
    gathered = comm.gather(local_dict, root=root)
    if rank == root:
        summed_dict = dict(gathered[0])
        for other in gathered[1:]:
            for k, v in other.items():
                if k in summed_dict:
                    summed_dict[k] = _add_cotangent(summed_dict[k], v)
                else:
                    summed_dict[k] = v
        summed = [summed_dict.get(k) for k in local_paths]
        return jax.tree_util.tree_unflatten(treedef, summed)
    return None


def _bcast_canonical_setup(comm, mol, mf, lo_coeff, frag_lolist_static,
                            prescreen_data, root=0):
    """Broadcast rank-0's SCF / LO / fragment topology to all ranks.

    Returns a dict of canonical state suitable for non-root ranks to
    construct skeleton ``mf`` objects via the user's ``build_mf`` and
    to initialize ``lo_coeff``, ``frag_lolist_static``, and
    ``prescreen_data`` without re-running any of the rank-0 setup.
    """
    rank = comm.Get_rank()
    if rank == root:
        canonical = {
            'mo_coeff':  numpy.asarray(mf.mo_coeff),
            'mo_energy': numpy.asarray(mf.mo_energy),
            'mo_occ':    numpy.asarray(mf.mo_occ),
            'e_tot':     float(mf.e_tot),
            'lo_coeff':  numpy.asarray(lo_coeff),
            'frag_lolist': tuple(numpy.asarray(f) for f in frag_lolist_static),
            'prescreen_data': prescreen_data,
        }
    else:
        canonical = None
    canonical = comm.bcast(canonical, root=root)
    return canonical


class DLNOCCSD(lno_base_mpi_mod.LNO, _DLNOCCSDSingle):
    """MPI variant of :class:`pyscfad.dlno.ccsd.DLNOCCSD`.

    Both :meth:`kernel` (inherited from ``lno_base_mpi.LNO`` for the
    plain-energy MPI flow used by example 12) and :meth:`value_and_grad`
    (defined here) are exposed.
    """

    @classmethod
    def value_and_grad(cls, mol, *, build_mf, frag_lolist=None,
                       include_mp2_correction=True,
                       mp2_correction_method='sos',
                       mp2_correction_scope='domain',
                       sos_c_os=None,
                       # CCSD solver config
                       frozen=0,
                       thresh_occ=1e-4,
                       thresh_vir=1e-5,
                       lo_type='iao',
                       no_type='ie',
                       ccsd_t=False,
                       dcsd=False,
                       verbose_imp=0,
                       # Prescreen build config
                       lmo_bp_domain_thr=None,
                       pao_bp_domain_thr=None,
                       domain_pao_thr=None,
                       pair_energy_thr=None,
                       multipole_order=None):
        """MPI-parallel value_and_grad: rank-0 owns SCF + LO + prescreen +
        ``lo_vjp`` + ``scf_vjp``; non-root ranks only do fragment
        forward/backward; cotangents are gathered to rank 0 for the
        LO and SCF close-out.

        Same call signature as
        :meth:`pyscfad.dlno.ccsd.DLNOCCSD.value_and_grad`, plus one
        extra requirement on the user-supplied ``build_mf``: when
        called with ``mo_coeff_init`` / ``mo_energy_init`` /
        ``mo_occ_init`` / ``e_tot_init`` kwargs, it must construct a
        *skeleton* mf (with_df attached, scf-object set up) that
        adopts those canonical values directly **without calling
        ``mf.kernel()``**.  Non-root ranks don't need a registered
        SCF custom_vjp -- only the forward fragment computations
        consume the mf -- so skipping the kernel call is correct and
        avoids all the SCF-gauge / convergence-noise / first-order-
        custom-replay subtleties that creep in when each rank also
        tries to register its own ``scf_vjp``.

        Strategy:

            1. **Rank 0 alone** runs the canonical setup:

               - ``mf, scf_vjp = jax.vjp(build_mf, mol)`` (full SCF)
               - ``lo_coeff, lo_vjp = jax.vjp(build_lo, mf)``
               - eager DLNO topology / prescreen build

            2. **Broadcast** the canonical
               ``mo_coeff``/``mo_energy``/``mo_occ``/``e_tot``, the
               ``lo_coeff``, the ``frag_lolist``, and the
               ``prescreen_data`` from rank 0 to every rank.

            3. **Each rank** constructs a skeleton mf from the
               broadcast canonical state (via the user's
               ``build_mf`` with canonical kwargs).  ``lo_coeff`` is
               loaded as a JAX-array leaf.  No ``scf_vjp`` or
               ``lo_vjp`` is captured on non-root.

            4. **Each rank** processes its round-robin subset of
               fragments: ``e_frag, vjp_fn = jax.vjp(per_frag_fn, mf,
               lo_coeff); vjp_fn(1.0)`` -> local cotangents on
               ``(mf, lo_coeff)``.

            5. **Gather to rank 0**: the per-rank ``grad_mf`` and
               ``grad_lo`` pytrees are summed onto rank 0 via the
               path-keyed ``_tree_sum_to_root``.

            6. **Rank 0** applies ``lo_vjp`` then ``scf_vjp`` to the
               gathered cotangents and broadcasts the resulting
               ``grad_mol`` to all ranks.

        Why this is correct (and robust to all the gauge issues):

            All per-rank fragments operate on **exactly the same**
            ``mo_coeff`` / ``mo_energy`` / ``lo_coeff`` /
            ``frag_lolist`` / ``prescreen_data`` -- everything came
            from rank 0's single setup.  No rank has any chance of
            picking a different orbital sign, internal rotation,
            fragmentation order, or LO localization order.  The only
            ``scf_vjp`` / ``lo_vjp`` that ever runs is rank 0's, so
            there is no per-rank-CPHF inconsistency either.  Summing
            cotangents across ranks before the close-out is exact
            because the cotangents are all on the same canonical
            objects.

        The HF energy contribution is seeded into ``grad_mf`` on
        rank 0 only, so it appears exactly once.
        """
        # Resolve prescreen defaults from the class
        if lmo_bp_domain_thr is None: lmo_bp_domain_thr = cls.lmo_bp_domain_thr
        if pao_bp_domain_thr is None: pao_bp_domain_thr = cls.pao_bp_domain_thr
        if domain_pao_thr    is None: domain_pao_thr    = cls.domain_pao_thr
        if pair_energy_thr   is None: pair_energy_thr   = cls.pair_energy_thr
        if multipole_order   is None: multipole_order   = cls.multipole_order
        if sos_c_os is None:
            sos_c_os = lno_base.DOMAIN_SOS_MP2_C_OS
        mp2_correction_scope = str(mp2_correction_scope).lower().replace('_', '-')
        if mp2_correction_scope not in ('domain', 'full'):
            raise ValueError(
                "mp2_correction_scope must be either 'domain' or 'full', "
                f"got {mp2_correction_scope!r}"
            )

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        rank = comm.Get_rank()

        verbose = int(getattr(mol, 'verbose', 0))
        log = ((lambda msg: print(msg, flush=True))
               if (rank == 0 and verbose >= _VERBOSE_PROGRESS)
               else (lambda msg: None))
        t_overall = time.perf_counter()
        log(f'DLNOCCSD.value_and_grad (MPI, nproc={nproc}): start')

        # ---- Rank 0: full canonical setup (SCF + LO + prescreen) ----
        # Everything below produces canonical state that we'll
        # broadcast to all ranks.  Only rank 0 holds the JAX vjp
        # closures (scf_vjp, lo_vjp) -- those are used at the
        # close-out.
        if rank == 0:
            t0 = time.perf_counter()
            mf, scf_vjp = jax.vjp(build_mf, mol)
            log(f'  SCF (build_mf + jax.vjp):   {time.perf_counter() - t0:8.2f} s')

            def _build_lo(mf_):
                cc_local = LNOCCSD(mf_, frozen=frozen)
                cc_local.lo_type = lo_type
                return cc_local.get_lo(lo_type=cc_local.lo_type)
            t0 = time.perf_counter()
            lo_coeff, lo_vjp = jax.vjp(_build_lo, mf)
            log(f'  LO transform + jax.vjp:     {time.perf_counter() - t0:8.2f} s')

            def _topo_builder(mf_, frag_lolist_):
                return _build_static_dlno_topology(
                    mf_, frag_lolist_, frozen, lo_type,
                    lmo_bp_domain_thr, pao_bp_domain_thr,
                    domain_pao_thr, pair_energy_thr, multipole_order,
                )
            t0 = time.perf_counter()
            frag_lolist_static, prescreen_data = stop_trace(_topo_builder)(mf, frag_lolist)
            log(f'  DLNO prescreen build:       {time.perf_counter() - t0:8.2f} s')
        else:
            mf = scf_vjp = lo_vjp = lo_coeff = None
            frag_lolist_static = prescreen_data = None

        # ---- Broadcast canonical state to all ranks ----
        t0 = time.perf_counter()
        canonical = _bcast_canonical_setup(
            comm, mol, mf, lo_coeff, frag_lolist_static, prescreen_data,
        )
        log(f'  broadcast canonical setup:  {time.perf_counter() - t0:8.2f} s')

        # ---- Non-root ranks: build skeleton mf + load lo_coeff ----
        if rank != 0:
            mf = build_mf(
                mol,
                mo_coeff_init=canonical['mo_coeff'],
                mo_energy_init=canonical['mo_energy'],
                mo_occ_init=canonical['mo_occ'],
                e_tot_init=canonical['e_tot'],
            )
        # All ranks now use the canonical lo_coeff / fragment topology
        lo_coeff = jnp.asarray(canonical['lo_coeff'])
        frag_lolist_static = canonical['frag_lolist']
        prescreen_data = canonical['prescreen_data']

        nfrag = len(frag_lolist_static)
        sign_pt2 = -1.0 if include_mp2_correction else 0.0

        # Round-robin partition: rank r processes fragments i where i % nproc == r
        my_fragment_indices = [i for i in range(nfrag) if i % nproc == rank]

        if verbose >= _VERBOSE_PROGRESS:
            atom_counts = [
                int(numpy.asarray(prescreen_data['fragment_data'][i]
                                  .get('extended_primary_domain')).size)
                for i in range(nfrag)
            ]
            atom_range = (f'{min(atom_counts)}-{max(atom_counts)}'
                          if min(atom_counts) != max(atom_counts)
                          else f'{atom_counts[0]}')
            log(f'  Fragments: {nfrag} (domain atoms/frag: {atom_range})')
            if include_mp2_correction:
                correction_method = str(mp2_correction_method).lower().replace('_', '-')
                scope_label = 'full-system' if mp2_correction_scope == 'full' else 'per-domain'
                if correction_method in ('sos', 'sos-mp2', 'lt-sos', 'lt-sos-mp2'):
                    correction_label = f'{scope_label} SOS-MP2 (c_os={sos_c_os:g})'
                elif correction_method in ('mp2', 'full-mp2', 'conventional'):
                    correction_label = f'{scope_label} full MP2'
                else:
                    correction_label = f'{scope_label} {mp2_correction_method}'
            else:
                correction_label = 'off'
            log(f'  MP2 correction: {correction_label}'
                f',  CCSD(T): {ccsd_t}')
            log(f'  MPI partition: nproc={nproc}')

        # ---- HF-energy seed only on rank 0 ----
        # We only run scf_vjp on rank 0, so the HF cotangent only
        # needs to live there.
        if rank == 0:
            _, hf_vjp = jax.vjp(lambda m: m.e_tot, mf)
            grad_mf = hf_vjp(1.0)[0]
        else:
            # On non-root grad_mf starts at None/zero; we just
            # accumulate fragment cotangents into it.  We initialize
            # it to None per-leaf via the tree-map below; the
            # _add_cotangent reduction handles the None entries.
            grad_mf = None
        grad_lo = jnp.zeros_like(lo_coeff)

        # ---- Per-fragment loop (rank-local subset) ----
        e_corr_local = jnp.float64(0.0)
        for ifrag in my_fragment_indices:
            fraglo_idx = frag_lolist_static[ifrag]
            frag_prescreen = prescreen_data['fragment_data'][ifrag]
            weight = 1.0

            def per_frag_fn(mf_, lo_coeff_,
                            _ifrag=ifrag,
                            _fraglo=fraglo_idx,
                            _frag_prescreen=frag_prescreen,
                            _weight=weight):
                cc_local = LNOCCSD(mf_, frozen=frozen)
                cc_local.thresh_occ = thresh_occ
                cc_local.thresh_vir = thresh_vir
                cc_local.lo_type = lo_type
                cc_local.no_type = no_type
                cc_local.ccsd_t = ccsd_t
                cc_local.dcsd = dcsd
                cc_local.verbose_imp = verbose_imp
                cc_local.compute_domain_pt2 = (
                    include_mp2_correction
                    and mp2_correction_scope == 'domain'
                )
                cc_local.domain_mp2_method = mp2_correction_method
                cc_local.domain_sos_mp2_c_os = sos_c_os
                
                cc_local.use_dlno_prescreen = True
                cc_local.dlno_prescreen_data = _frag_prescreen

                orbfragloc = lo_coeff_[:, _fraglo]
                stub_eris = _make_stub_eris(mf_)
                eris_fpno = lno_base.make_fragment_eris(
                    cc_local, stub_eris, _frag_prescreen,
                )
                t_fpno1 = time.perf_counter()
                frzfrag, orbfrag, domain_pt2 = lno_base.make_fpno1(
                    cc_local, eris_fpno, orbfragloc, no_type,
                    lno_base.THRESH_INTERNAL,
                    (cc_local.thresh_occ, cc_local.thresh_vir),
                    frag_prescreen=_frag_prescreen,
                    frozen_mask=cc_local.get_frozen_mask(),
                )
                if verbose >= _VERBOSE_PROGRESS:
                    print(f'  [rank {rank}] [frag {_ifrag+1}/{nfrag}] '
                          f'make_fpno1:          '
                          f'{time.perf_counter() - t_fpno1:8.2f} s',
                          flush=True)

                if orbfrag is None:
                    contribution = (
                        domain_pt2
                        if include_mp2_correction and mp2_correction_scope == 'domain'
                        else jnp.float64(0.0)
                    )
                    aux = {
                        'pt2':        jnp.float64(0.0),
                        'cc':         jnp.float64(0.0),
                        'cc_t':       jnp.float64(0.0),
                        'domain_pt2': jnp.asarray(
                            domain_pt2
                            if include_mp2_correction and mp2_correction_scope == 'domain'
                            else 0.0
                        ),
                    }
                    return _weight * contribution, aux

                res = _impurity_solve_core(
                    mf_, orbfrag, orbfragloc,
                    eris_fpno.fock, eris_fpno.s1e,
                    frozen=frzfrag, frag_prescreen=_frag_prescreen,
                    verbose_imp=cc_local.verbose_imp,
                    ccsd_t=cc_local.ccsd_t,
                    dcsd=cc_local.dcsd,
                    profile_info=None,
                    profile_pass=getattr(cc_local, 'profile_pass', None),
                    pt2_fragment_method=(
                        mp2_correction_method if include_mp2_correction else 'mp2'
                    ),
                    sos_c_os=sos_c_os,
                )
                e_pt2_frag, e_cc_frag, e_cc_t_frag = res

                contribution = e_cc_frag + e_cc_t_frag + sign_pt2 * e_pt2_frag
                if include_mp2_correction and mp2_correction_scope == 'domain':
                    contribution = contribution + domain_pt2
                aux = {
                    'pt2':        jnp.asarray(e_pt2_frag),
                    'cc':         jnp.asarray(e_cc_frag),
                    'cc_t':       jnp.asarray(e_cc_t_frag),
                    'domain_pt2': jnp.asarray(
                        domain_pt2
                        if include_mp2_correction and mp2_correction_scope == 'domain'
                        else 0.0
                    ),
                }
                return _weight * contribution, aux

            t_frag = time.perf_counter()
            if verbose >= _VERBOSE_PROGRESS:
                n_atoms = int(numpy.asarray(
                    frag_prescreen.get('extended_primary_domain')).size)
                print(f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] '
                      f'domain={n_atoms} atoms, lo_size={len(fraglo_idx)}: '
                      f'starting fwd+bwd...', flush=True)

            e_frag, vjp_fn, aux = jax.vjp(
                per_frag_fn, mf, lo_coeff, has_aux=True,
            )
            e_corr_local = e_corr_local + e_frag
            if verbose >= _VERBOSE_PROGRESS:
                print(f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] '
                      'reverse VJP: start', flush=True)
            progress_prefix = (
                f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] reverse VJP:'
                if verbose >= _VERBOSE_PROGRESS else None
            )
            with lno_base.vjp_progress(progress_prefix):
                g_mf_i, g_lo_i = vjp_fn(jnp.float64(1.0))
            if verbose >= _VERBOSE_PROGRESS:
                print(f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] '
                      'reverse VJP: done', flush=True)
            if grad_mf is None:
                grad_mf = g_mf_i
            else:
                grad_mf = jax.tree_util.tree_map(_add_cotangent, grad_mf, g_mf_i)
            grad_lo = jax.tree_util.tree_map(_add_cotangent, grad_lo, g_lo_i)
            jax.block_until_ready((e_corr_local, grad_mf, grad_lo))

            if verbose >= _VERBOSE_PROGRESS:
                print(f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] done in '
                      f'{time.perf_counter() - t_frag:.2f} s, '
                      f'contribution = {float(e_frag):+.8f}  '
                      f'(pt2={float(aux["pt2"]):+.8f}, '
                      f'cc={float(aux["cc"]):+.8f}, '
                      f'(T)={float(aux["cc_t"]):+.8f}'
                      + (f', domain_pt2={float(aux["domain_pt2"]):+.8f}'
                         if include_mp2_correction and mp2_correction_scope == 'domain' else '')
                      + ')',
                      flush=True)
            del e_frag, vjp_fn, aux, g_mf_i, g_lo_i, per_frag_fn
            del fraglo_idx, frag_prescreen
            _cleanup_after_fragment(
                log if verbose >= _VERBOSE_PROGRESS else None,
                label=f'rank {rank} fragment {ifrag+1}/{nfrag}',
            )

        if rank == 0 and include_mp2_correction and mp2_correction_scope == 'full':
            t0 = time.perf_counter()
            full_mp2_correction, full_mp2_vjp = jax.vjp(
                lambda m: lno_base.full_system_mp2_correction(
                    m, method=mp2_correction_method, c_os=sos_c_os
                ),
                mf,
            )
            grad_mf_full, = full_mp2_vjp(jnp.float64(1.0))
            grad_mf = jax.tree_util.tree_map(
                _add_cotangent, grad_mf, grad_mf_full
            )
            e_corr_local = e_corr_local + full_mp2_correction
            jax.block_until_ready((e_corr_local, grad_mf))
            log(f'  Full-system MP2 correction: {time.perf_counter() - t0:.2f} s, '
                f'e = {float(full_mp2_correction):+.10f}')
            del full_mp2_correction, full_mp2_vjp, grad_mf_full

        # Defensive: ranks with no fragments have grad_mf=None still.
        # Seed grad_mf as a zeros-of-canonical shape so the gather
        # finds matching paths.  (Currently the HF seed on rank 0
        # plus at least one fragment on each rank in our typical
        # round-robin means this code path is rarely hit, but it's
        # cheap to be safe.)
        if grad_mf is None and rank != 0:
            grad_mf = jax.tree_util.tree_map(
                lambda x: (jnp.zeros_like(x)
                           if hasattr(x, 'dtype')
                              and x.dtype != jax.dtypes.float0
                           else None),
                mf,
            )

        # ---- Gather cotangents to rank 0 ----
        t0 = time.perf_counter()
        grad_mf_root = _tree_sum_to_root(comm, grad_mf, root=0)
        grad_lo_root = _tree_sum_to_root(comm, grad_lo, root=0)
        e_corr_total = comm.allreduce(float(e_corr_local), op=MPI.SUM)
        log(f'  MPI gather (energy + cotangents): {time.perf_counter() - t0:.2f} s')

        # ---- Rank 0: LO + SCF close-out ----
        if rank == 0:
            t0 = time.perf_counter()
            grad_mf_via_lo, = lo_vjp(grad_lo_root)
            grad_mf_root = jax.tree_util.tree_map(
                _add_cotangent, grad_mf_root, grad_mf_via_lo,
            )
            log(f'  LO vjp close-out:           {time.perf_counter() - t0:8.2f} s')

            t0 = time.perf_counter()
            grad_mol, = scf_vjp(grad_mf_root)
            log(f'  SCF (CPHF) vjp close-out:   {time.perf_counter() - t0:8.2f} s')
        else:
            grad_mol = None

        # ``grad_mol`` is left on rank 0 only.  A ``comm.bcast`` of the
        # Mole-shaped pytree drops the ``coords`` attribute through the
        # pickle round-trip (pyscfad's dynamic-attr machinery isn't
        # re-applied on unpickle), so we don't ship it.  This matches
        # the example-12 pattern: rank 0 owns the reduced gradient,
        # non-root ranks get ``None``.

        e_hf = canonical['e_tot']
        e_total = e_hf + e_corr_total
        log(f'DLNOCCSD.value_and_grad (MPI, nproc={nproc}): done in '
            f'{time.perf_counter() - t_overall:.2f} s total, '
            f'e_total = {float(e_total):.10f}')
        return e_total, grad_mol
