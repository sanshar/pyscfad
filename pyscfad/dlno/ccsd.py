# Copyright 2021-2025 Xing Zhang
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

import gc
import os
import time

import jax
import jax.numpy as jnp

from pyscfad import numpy as np
from pyscfad.ops import stop_trace
from pyscfad.lno import lno_base
from pyscfad.lno.ccsd import LNOCCSD, _impurity_solve_core
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.dlno.prescreen import (
    build_dlno_prescreen_data,
    rebuild_dlno_prescreen_data,
)


# Progress-print verbosity threshold (matches pyscf logger convention:
# ``log.info`` = level 4).  Reads ``mol.verbose`` from the input mol.
_VERBOSE_PROGRESS = 4


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _rss_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0**2
    except OSError:
        pass
    return None


def _cleanup_after_fragment(log=None, label=None):
    if _env_flag('PYSCFAD_DLNO_FRAGMENT_GC', True):
        gc.collect()
    if _env_flag('PYSCFAD_DLNO_CLEAR_JAX_CACHES_PER_FRAGMENT', False):
        jax.clear_caches()
    if log is not None and _env_flag('PYSCFAD_DLNO_LOG_FRAGMENT_RSS', True):
        rss = _rss_gb()
        if rss is not None:
            suffix = f' after {label}' if label else ''
            log(f'  RSS{suffix}: {rss:.2f} GB')


class DLNOCCSD(LNOCCSD):
    """CCSD with domain-restricted LNO prescreening.

    A thin subclass of :class:`pyscfad.lno.ccsd.LNOCCSD` whose constructor
    accepts ``dlno_prescreen_data`` directly and enables
    ``use_dlno_prescreen`` by default.  All solver logic (``impurity_solve``,
    ``kernel``, energy accessors, ``e_corr_pt2corrected``, ...) is inherited
    unchanged.  Use this class whenever a calculation needs the DLNO
    prescreening; use the parent :class:`LNOCCSD` for vanilla LNO.

    Default prescreen-build parameters live on the class
    (``lmo_bp_domain_thr`` etc.) and are consumed by :meth:`value_and_grad`
    when it builds the prescreen internally from a fresh ``mf``.
    """

    # Default prescreen-build parameters consumed by `value_and_grad`.
    # Override per-instance for non-default DLNO calculations.
    lmo_bp_domain_thr = 0.9
    pao_bp_domain_thr = 0.9
    domain_pao_thr = 1e-4
    pair_energy_thr = 1e-4
    multipole_order = 4

    def __init__(self, mf, thresh=1e-4, frozen=None, fock=None, s1e=None,
                 dlno_prescreen_data=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, fock=fock,
                         s1e=s1e, **kwargs)
        self.use_dlno_prescreen = True
        self.dlno_prescreen_data = dlno_prescreen_data

    # ------------------------------------------------------------------
    # Progressive per-fragment value_and_grad.
    #
    # Computes (total energy, d total energy / d mol) in one pass:
    # SCF is run once under jax.vjp; the LO transform is built once under
    # jax.vjp; the DLNO prescreen is built once eagerly (via stop_trace,
    # no backprop through the prescreen itself); then each fragment's
    # fwd + bwd run together inside the for-loop, so per-fragment t1, t2,
    # PNO orbitals, and the fragment-local Lov are freed at the end of
    # each iteration before the next fragment starts.
    #
    # MP2 correction: replaces the per-fragment LNO PT2 contribution with
    # the per-domain correction that ``make_fpno1`` returns.  The default
    # correction in value_and_grad is SOS-MP2.
    # ------------------------------------------------------------------

    @classmethod
    def value_and_grad(cls, mol, *, build_mf, frag_lolist=None,
                       include_mp2_correction=True,
                       mp2_correction_method='sos',
                       mp2_correction_scope='domain',
                       sos_c_os=None,
                       # CCSD solver config (mirrors LNOCCSD attributes):
                       frozen=0,
                       thresh_occ=1e-4,
                       thresh_vir=1e-5,
                       lo_type='iao',
                       no_type='ie',
                       ccsd_t=False,
                       dcsd=False,
                       verbose_imp=0,
                       # Prescreen build config (mirrors class attributes):
                       lmo_bp_domain_thr=None,
                       pao_bp_domain_thr=None,
                       domain_pao_thr=None,
                       pair_energy_thr=None,
                       multipole_order=None):
        """Compute (total energy, mol-gradient) via per-fragment progressive AD.

        This is a classmethod, *not* an instance method — there is no
        DLNOCCSD instance involved.  The user passes the molecule, an
        SCF builder, the fragment partitioning, and configuration kwargs
        directly.  Internally, SCF runs once under ``jax.vjp`` (via
        ``build_mf``), the LO transform is built once under ``jax.vjp``,
        the DLNO prescreen is built eagerly via ``stop_trace`` (no
        backprop through the topology), and each fragment's fwd + bwd
        run together inside the for-loop, so per-fragment t1, t2, PNO
        orbitals, and the fragment-local Lov are freed at the end of
        each iteration before the next fragment starts.

        MP2 correction: replaces the per-fragment LNO PT2 contribution
        with either the per-domain correction that ``make_fpno1`` returns
        (``mp2_correction_scope='domain'``) or one full-system correction
        evaluated once outside the fragment loop
        (``mp2_correction_scope='full'``).
        The default correction method is scaled opposite-spin MP2
        (``mp2_correction_method='sos'``), which uses a density-fitted
        Laplace/SOS contraction and avoids the conventional MP2
        virtual-pair tensors.  Set ``mp2_correction_method='mp2'`` to
        recover the previous full same-spin plus opposite-spin MP2
        correction within the selected correction scope.

        Args:
            mol : :class:`pyscfad.gto.Mole`
                Molecular geometry to differentiate with respect to.
            build_mf : callable
                ``mol -> mf``.  Should construct the SCF object (including
                any outcore-CDERI attachment) and call ``mf.kernel()``.
                Wrapped in ``jax.vjp`` here.
            frag_lolist : list of arrays, optional
                Per-fragment LO indices.  If ``None``, built eagerly via
                ``autofrag + map_lo_to_frag`` on the concrete reference
                geometry inside ``stop_trace``.
            include_mp2_correction : bool, default True
                If ``True``, each fragment's contribution is
                ``cc + cc_t - lno_pt2 + domain_mp2`` (per-domain MP2/SOS-MP2
                replaces LNO PT2; no full canonical MP2).  If ``False``,
                the contribution is just ``cc + cc_t``.
            mp2_correction_method : {'sos', 'mp2'}, default 'sos'
                Domain correction used when ``include_mp2_correction`` is true.
                ``'sos'`` uses scaled opposite-spin DF-MP2 and is the default
                for differentiated DLNO-CCSD/(T). ``'mp2'`` uses the previous
                full MP2 correction.
            mp2_correction_scope : {'domain', 'full'}, default 'domain'
                ``'domain'`` preserves the historical per-fragment domain MP2
                correction. ``'full'`` computes one full-system MP2/SOS-MP2
                correction and adds its VJP once, avoiding repeated
                near-full-domain MP2 work.
            sos_c_os : float, optional
                Opposite-spin scale factor for ``mp2_correction_method='sos'``.
                Defaults to ``lno_base.DOMAIN_SOS_MP2_C_OS`` (1.3 unless
                overridden through PySCF config).
            frozen, thresh_occ, thresh_vir, lo_type, no_type, ccsd_t,
            dcsd, verbose_imp:
                CCSD-solver settings, mirror the like-named attributes on
                ``LNOCCSD``.  Defaults match ``LNOCCSD``'s.
            lmo_bp_domain_thr, pao_bp_domain_thr, domain_pao_thr,
            pair_energy_thr, multipole_order:
                Prescreen-build settings.  If ``None``, the class default
                from ``cls`` is used (so e.g. ``DLNOCCSD.value_and_grad``
                uses ``DLNOCCSD.domain_pao_thr = 1e-4``).

        Returns:
            (e_total, mol_grad) where ``e_total`` includes ``mf.e_tot``
            and ``mol_grad`` is shaped like ``mol`` (typically just the
            ``coords`` leaf is nonzero for nuclear-coordinate derivatives).
        """
        # Resolve prescreen-build defaults from the class.
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

        # Progress reporting (gated on ``mol.verbose >= 4``, matching the
        # pyscf logger convention).
        verbose = int(getattr(mol, 'verbose', 0))
        log = (lambda msg: print(msg, flush=True)) if verbose >= _VERBOSE_PROGRESS else (lambda msg: None)
        t_overall = time.perf_counter()
        log('DLNOCCSD.value_and_grad: start')

        # ---------------- Outside the fragment loop ----------------

        # SCF: 1× iteration, save CPHF backward for close-out at the end.
        t0 = time.perf_counter()
        mf, scf_vjp = jax.vjp(build_mf, mol)
        log(f'  SCF (build_mf + jax.vjp):   {time.perf_counter() - t0:8.2f} s')

        # Seed the mf cotangent with the HF-energy contribution: the
        # final answer's e_total includes mf.e_tot, so its cotangent
        # (= 1.0 under jax.value_and_grad equivalence) flows back through
        # mf -> mol via scf_vjp at the end.
        e_hf, hf_vjp = jax.vjp(lambda m: m.e_tot, mf)
        grad_mf = hf_vjp(1.0)[0]

        # LO transform: 1× build, save its bwd for close-out.  Uses the
        # same ``cc.get_lo`` path as the existing kernel.
        def _build_lo(mf_):
            cc_local = LNOCCSD(mf_, frozen=frozen)
            cc_local.lo_type = lo_type
            return cc_local.get_lo(lo_type=cc_local.lo_type)

        t0 = time.perf_counter()
        lo_coeff, lo_vjp = jax.vjp(_build_lo, mf)
        grad_lo = jax.tree_util.tree_map(jnp.zeros_like, lo_coeff)
        log(f'  LO transform + jax.vjp:     {time.perf_counter() - t0:8.2f} s')

        # Fragment LO assignment + prescreen are built eagerly (concrete
        # numpy/JAX arrays).  Their construction reads mf state but the
        # *gradient* through them is intentionally dropped — they're
        # treated as fixed metadata that pins the fragmentation topology
        # to the reference geometry.  This is the same approximation as
        # the "fixed LNO" mode in example 13.
        # stop_trace only accepts JAX-array / pytree inputs as positional
        # args; the string/scalar config is captured via closure.
        def _topo_builder(mf_, frag_lolist_):
            return _build_static_dlno_topology(
                mf_, frag_lolist_, frozen, lo_type,
                lmo_bp_domain_thr, pao_bp_domain_thr,
                domain_pao_thr, pair_energy_thr, multipole_order,
            )
        t0 = time.perf_counter()
        frag_lolist_static, prescreen_data = stop_trace(
            _topo_builder
        )(mf, frag_lolist)
        log(f'  DLNO prescreen build:       {time.perf_counter() - t0:8.2f} s')

        nfrag = len(frag_lolist_static)
        frag_wghtlist = [1.0] * nfrag
        sign_pt2 = -1.0 if include_mp2_correction else 0.0

        if verbose >= _VERBOSE_PROGRESS:
            atom_counts = [
                int(np.asarray(prescreen_data['fragment_data'][i]
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

        # ---------------- Per-fragment loop ----------------

        e_corr = jnp.float64(0.0)
        for ifrag in range(nfrag):
            fraglo_idx = frag_lolist_static[ifrag]
            frag_prescreen = prescreen_data['fragment_data'][ifrag]
            weight = frag_wghtlist[ifrag]

            def per_frag_fn(mf_, lo_coeff_,
                            _ifrag=ifrag,
                            _fraglo=fraglo_idx,
                            _frag_prescreen=frag_prescreen,
                            _weight=weight):
                # Construct a transient plain LNOCCSD for this fragment.
                # The DLNO behavior at the per-fragment level flows
                # entirely through ``_frag_prescreen``; no DLNOCCSD-
                # specific attributes are needed inside the body.
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

                #jo changes
                cc_local.use_dlno_prescreen = True
                cc_local.dlno_prescreen_data = frag_prescreen
                cc_local._current_ifrag = _ifrag

                orbfragloc = lo_coeff_[:, _fraglo]

                # Build the fragment eris in place, streaming the
                # domain-local CDERI from the outcore file via
                # ``get_Lov(..., atmlst=...)``.  No global Lov is
                # ever materialized.
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
                    log(f'  [frag {_ifrag+1}/{nfrag}] make_fpno1:          '
                        f'{time.perf_counter() - t_fpno1:8.2f} s')

                if orbfrag is None:
                    # No PNOs survived the threshold; only the
                    # per-domain MP2 contributes (if MP2 correction on).
                    if include_mp2_correction and mp2_correction_scope == 'domain':
                        return _weight * domain_pt2
                    return jnp.float64(0.0)

                # Call _impurity_solve_core directly, bypassing the public
                # ``impurity_solve`` dispatcher and its ``_impurity_solve_jax``
                # custom_vjp wrap.  Inside the per-fragment progressive AD
                # loop the wrap is counterproductive: its bwd would re-run
                # ``_impurity_solve_core`` inside ``jax.vjp``, causing each
                # fragment's CCSD(T) to run twice (once in the outer
                # ``jax.vjp(per_frag_fn, ...)`` fwd that we're already
                # tracing, and again in the wrap's bwd re-trace).  Going
                # straight to ``_impurity_solve_core`` lets the outer
                # ``jax.vjp`` save the inner custom_vjps' residuals once
                # (``_dfccsd_kernel_custom`` saves t1/t2; the per-fragment
                # eris custom_vjps save their own residuals) and reuse them
                # in the pullback below — exactly one CCSD(T) per fragment.
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
                n_atoms = int(np.asarray(
                    frag_prescreen.get('extended_primary_domain')).size)
                log(f'  [frag {ifrag+1}/{nfrag}] domain={n_atoms} atoms, '
                    f'lo_size={len(fraglo_idx)}: starting fwd+bwd...')

            e_frag, vjp_fn, aux = jax.vjp(
                per_frag_fn, mf, lo_coeff, has_aux=True,
            )
            e_corr = e_corr + e_frag
            if verbose >= _VERBOSE_PROGRESS:
                log(f'  [frag {ifrag+1}/{nfrag}] reverse VJP: start')
            with lno_base.vjp_progress(
                f'  [frag {ifrag+1}/{nfrag}] reverse VJP:'
                if verbose >= _VERBOSE_PROGRESS else None
            ):
                g_mf_i, g_lo_i = vjp_fn(jnp.float64(1.0))
            if verbose >= _VERBOSE_PROGRESS:
                log(f'  [frag {ifrag+1}/{nfrag}] reverse VJP: done')
            grad_mf = jax.tree_util.tree_map(_add_cotangent, grad_mf, g_mf_i)
            grad_lo = jax.tree_util.tree_map(_add_cotangent, grad_lo, g_lo_i)
            jax.block_until_ready((e_corr, grad_mf, grad_lo))
            # per-fragment t1, t2, PNOs, eris_fpno, transient cc all
            # freed at the end of this iteration.
            log(f'  [frag {ifrag+1}/{nfrag}] done in '
                f'{time.perf_counter() - t_frag:8.2f} s, '
                f'contribution = {float(e_frag):+.8f}  '
                f'(pt2={float(aux["pt2"]):+.8f}, '
                f'cc={float(aux["cc"]):+.8f}, '
                f'(T)={float(aux["cc_t"]):+.8f}'
                + (f', domain_pt2={float(aux["domain_pt2"]):+.8f}'
                   if include_mp2_correction and mp2_correction_scope == 'domain' else '')
                + ')')
            del e_frag, vjp_fn, aux, g_mf_i, g_lo_i, per_frag_fn
            del fraglo_idx, frag_prescreen
            _cleanup_after_fragment(
                log if verbose >= _VERBOSE_PROGRESS else None,
                label=f'fragment {ifrag+1}/{nfrag}',
            )

        if include_mp2_correction and mp2_correction_scope == 'full':
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
            e_corr = e_corr + full_mp2_correction
            jax.block_until_ready((e_corr, grad_mf))
            log(f'  Full-system MP2 correction: {time.perf_counter() - t0:8.2f} s, '
                f'e = {float(full_mp2_correction):+.10f}')
            del full_mp2_correction, full_mp2_vjp, grad_mf_full

        # ---------------- Close out outer-loop vjps ----------------

        # LO contribution -> mf cotangent
        t0 = time.perf_counter()
        grad_mf_via_lo, = lo_vjp(grad_lo)
        grad_mf = jax.tree_util.tree_map(_add_cotangent, grad_mf, grad_mf_via_lo)
        log(f'  LO vjp close-out:           {time.perf_counter() - t0:8.2f} s')

        # SCF (CPHF) close-out: one solve, mol cotangent.
        t0 = time.perf_counter()
        grad_mol, = scf_vjp(grad_mf)
        log(f'  SCF (CPHF) vjp close-out:   {time.perf_counter() - t0:8.2f} s')

        e_total = e_hf + e_corr
        log(f'DLNOCCSD.value_and_grad: done in '
            f'{time.perf_counter() - t_overall:.2f} s total,  '
            f'e_total = {float(e_total):.10f}')
        return e_total, grad_mol



def _build_static_dlno_topology(mf, frag_lolist, frozen, lo_type,
                                lmo_bp_domain_thr, pao_bp_domain_thr,
                                domain_pao_thr, pair_energy_thr,
                                multipole_order):
    """Eager DLNO prescreen build.  Called under ``stop_trace`` from
    ``DLNOCCSD.value_and_grad`` so the gradient path treats the
    fragmentation topology as fixed metadata (no backprop through it).
    """
    verbose = int(getattr(mf.mol, 'verbose', 0))
    log = (lambda msg: print(msg, flush=True)) if verbose >= _VERBOSE_PROGRESS else (lambda msg: None)

    t0 = time.perf_counter()
    cc_local = LNOCCSD(mf, frozen=frozen)
    cc_local.lo_type = lo_type
    lo_coeff = cc_local.get_lo(lo_type=cc_local.lo_type)
    log(f'    [topology] get_lo:                  {time.perf_counter() - t0:8.2f} s')

    if frag_lolist is None:
        t0 = time.perf_counter()
        frag_atmlist = autofrag(mf.mol)
        frag_lolist = map_lo_to_frag(
            mf.mol, lo_coeff, frag_atmlist, verbose=0,
        )
        log(f'    [topology] autofrag+map_lo_to_frag: '
            f'{time.perf_counter() - t0:8.2f} s ({len(frag_lolist)} fragments)')

    t0 = time.perf_counter()
    topology = build_dlno_prescreen_data(
        mf, lo_coeff, frag_lolist, frozen=frozen,
        lmo_bp_domain_thr=lmo_bp_domain_thr,
        pao_bp_domain_thr=pao_bp_domain_thr,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
        multipole_order=multipole_order,
    )
    log(f'    [topology] build_dlno_prescreen_data:'
        f'{time.perf_counter() - t0:8.2f} s')

    t0 = time.perf_counter()
    prescreen_data = rebuild_dlno_prescreen_data(
        mf, lo_coeff, topology, frozen=frozen,
    )
    log(f'    [topology] rebuild_dlno_prescreen_data:'
        f' {time.perf_counter() - t0:6.2f} s')
    return tuple(frag_lolist), prescreen_data


def _make_stub_eris(mf):
    """Build a minimal ``_LNOERIS`` with fock + s1e populated, no Lov.

    Used as the "reference eris" handed to ``make_fragment_eris`` so the
    domain-local Lov is built directly from the outcore CDERI without
    ever materializing a global Lov.
    """
    s1e = mf.get_ovlp()
    h1e = mf.get_hcore()
    vhf = mf.get_veff()
    fock = mf.get_fock(h1e=h1e, s1e=s1e, vhf=vhf)
    stub = lno_base._LNOERIS(fock=fock, s1e=s1e)
    return stub


def _add_cotangent(a, b):
    """Tree-aware cotangent accumulation that tolerates None / float0 leaves."""
    if a is None:
        return b
    if b is None:
        return a
    if hasattr(a, 'dtype') and a.dtype == jax.dtypes.float0:
        return b
    if hasattr(b, 'dtype') and b.dtype == jax.dtypes.float0:
        return a
    return a + b
