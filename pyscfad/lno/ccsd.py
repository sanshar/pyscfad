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

'''Impurity solver for LNO CCSD/CCSD(T).
'''

import os
import time
import numpy
from functools import reduce, partial

import jax
from jax import custom_vjp
from jax.interpreters import ad as jax_ad

from pyscf.lib import logger
from pyscf.mp.mp2 import _mo_without_core

from pyscfad import numpy as np
from pyscfad import lib
from pyscfad.cc import dfccsd, dfdcsd
from pyscfad.df import addons as df_addons
from pyscfad.ops import is_array
from pyscfad.lno import lno_base
from pyscfad.lno import ccsd_t as ccsd_t_mod


# Toggle for the analytical (custom_vjp) mp2_fragment_energy path.
_USE_CUSTOM_VJP_MP2_FRAG = True

_SOS_PT2_FRAGMENT_METHODS = (
    'sos', 'sos-mp2', 'df-sos', 'df-sos-mp2', 'lt-sos', 'lt-sos-mp2',
)

# Toggle for wrapping the eris-tensor construction (post-transform_df_to_mo).
_USE_CUSTOM_VJP_AO2MO_DF_ERIS = True

# Toggle for wrapping the full impurity_solve body as a single custom_vjp
# boundary.  When ``True`` AND the SCF object is in the outcore-CDERI
# regime, JAX sees one opaque op per fragment; the bwd re-runs
# ``_impurity_solve_core`` inside ``jax.vjp`` so the inner custom_vjps'
# bwds are composed automatically without retaining per-fragment
# residuals across the entire fragment loop's pullback.
#
# The wrap requires outcore CDERI (``mf.with_df.attach_outcore_cderi``):
# the incore ``_ao2mo.nr_e2`` calls ``numpy.asarray`` on the CDERI array
# and so cannot accept a JAX tracer, whereas the outcore variant
# (``_outcore_nr_e2``) is itself a custom_vjp that handles tracer inputs.
# When the SCF object is in incore mode, the dispatcher below detects
# this and silently routes through the plain (unwrapped) path so that
# tests/examples using in-core integrals continue to work.
#
# The wrap also relies on ``mo_coeff``, ``mo_energy`` and ``e_tot`` being
# declared as pytree-dynamic attributes on the SCF object (see
# ``pyscfad/df/df_jk.py:_DFHF._dynamic_attr``).  Without that, those
# attributes ride along as aux data through the pytree round-trip and
# leak stale outer-trace tracers into the re-traced bwd.
_USE_CUSTOM_VJP_IMPURITY_SOLVE = True


def _print_forward_t_energy(et, wall_time=None):
    jax.debug.print(
        '    forward (T) energy = {et:+.15g}',
        et=et,
        ordered=True,
    )
    if wall_time is not None:
        jax.debug.print(
            '    forward (T) time = {wall_time:.2f} s',
            wall_time=wall_time,
            ordered=True,
        )


def _should_print_forward_t_energy(verbose_imp, profile_pass):
    if profile_pass == 'backward replay':
        return False
    try:
        return int(verbose_imp) < logger.NOTE
    except (TypeError, ValueError):
        return True


def _should_print_t_timing(verbose_imp, mol_verbose):
    for value in (mol_verbose, verbose_imp):
        try:
            if int(value) >= logger.INFO:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _mf_supports_impurity_solve_wrap(mf):
    """True iff ``mf`` is in the outcore-CDERI regime the wrap requires.

    The wrap's bwd runs ``jax.vjp`` over the body, which lifts the CDERI
    tensor (a pytree leaf of ``mf.with_df``) to an abstract tracer in the
    re-traced scope.  The incore ao2mo path (``_ao2mo.nr_e2``) cannot
    consume an abstract tracer; the outcore path (``_outcore_nr_e2``)
    can.  We therefore only enable the wrap when the SCF is configured
    for outcore CDERI.
    """
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        return False
    placeholder_check = getattr(with_df, '_has_outcore_cderi_placeholder', None)
    if placeholder_check is None:
        return False
    return bool(placeholder_check())


def _is_zero_cot(x):
    return x is None or isinstance(x, jax_ad.Zero)


# ---------------------------------------------------------------------------
# Custom-vjp wrap of the per-fragment eris-tensor construction.
#
# After ``transform_df_to_mo`` returns the global Lpq tensor (shape
# ``(naux, nmo, nmo)``), this function builds the standard chemistry-eris
# attributes (Loo, Lov, Lvv, oooo, ovoo, ovov, oovv) via reshapes + matmuls.
# Those intermediates are otherwise transparent to the JAX trace and bloat
# the recorded jaxpr by ~500 MB at the LNO impurity fragment scale.
#
# Forward: same einsums as the original ``_make_df_eris_incore`` body.
# Backward: hand-coded adjoints
#   - oooo, ovoo, ovov, oovv → bar_Loo, bar_Lov, bar_Lvv (matmul transposes)
#   - bar_Lov/bar_Lvv/bar_Loo → bar_Lpq via slice scatter
# Saves residuals = (Lpq, Loo, Lov_flat, Lvv) plus nocc (static).
# ---------------------------------------------------------------------------
from functools import partial as _partial


def _adjoint_unpack_tril_last2(bar_dense, n):
    """Adjoint of unpack_tril along the last two dims.

    Forward: dense[..., r, c] = packed[..., K] for r >= c (lower);
             dense[..., c, r] = packed[..., K] (upper mirror).
    Adjoint: bar_packed[..., K] = bar_dense[..., r, c] + bar_dense[..., c, r]
             but with the diagonal counted only once.
    """
    rows, cols = numpy.tril_indices(n)
    diag_mask = numpy.asarray(rows == cols)
    sum_lower_upper = bar_dense[..., rows, cols] + bar_dense[..., cols, rows]
    # Diagonal got double-counted; subtract one back.
    diag_part = numpy.where(diag_mask, bar_dense[..., rows, cols], 0)
    return sum_lower_upper - diag_part


def _scatter_lower_tril_last2(bar_packed, n):
    """Inverse of pack_tril: scatter packed values into the lower triangle
    of an (n, n) matrix along the last two dims; upper triangle stays zero.
    """
    rows, cols = numpy.tril_indices(n)
    out_shape = bar_packed.shape[:-1] + (n, n)
    out = np.zeros(out_shape, dtype=bar_packed.dtype)
    out = out.at[..., rows, cols].set(bar_packed)
    return out


@_partial(custom_vjp, nondiff_argnums=(1,))
def _build_eris_tensors_from_Lpq(Lpq, nocc):
    naux, nmo, _ = Lpq.shape
    nvir = nmo - nocc
    Loo = Lpq[:, :nocc, :nocc].reshape(naux, nocc * nocc)
    Lov_flat = Lpq[:, :nocc, nocc:].reshape(naux, nocc * nvir)
    Lvv_full = Lpq[:, nocc:, nocc:]
    Lvv = lib.pack_tril(Lvv_full)
    Lov_resh = Lov_flat.reshape(naux, nocc, nvir)

    oooo = np.dot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)
    ovoo = np.dot(Lov_flat.T, Loo).reshape(nocc, nvir, nocc, nocc)
    ovov = np.dot(Lov_flat.T, Lov_flat).reshape(nocc, nvir, nocc, nvir)
    oovv_packed = np.dot(Loo.T, Lvv)
    oovv = lib.unpack_tril(oovv_packed).reshape(nocc, nocc, nvir, nvir)

    return Loo.reshape(naux, nocc, nocc), Lov_resh, Lvv, oooo, ovoo, ovov, oovv


def _build_eris_tensors_from_Lpq_fwd(Lpq, nocc):
    naux, nmo, _ = Lpq.shape
    nvir = nmo - nocc
    Loo = Lpq[:, :nocc, :nocc].reshape(naux, nocc * nocc)
    Lov_flat = Lpq[:, :nocc, nocc:].reshape(naux, nocc * nvir)
    Lvv_full = Lpq[:, nocc:, nocc:]
    Lvv = lib.pack_tril(Lvv_full)
    Lov_resh = Lov_flat.reshape(naux, nocc, nvir)

    oooo = np.dot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)
    ovoo = np.dot(Lov_flat.T, Loo).reshape(nocc, nvir, nocc, nocc)
    ovov = np.dot(Lov_flat.T, Lov_flat).reshape(nocc, nvir, nocc, nvir)
    oovv_packed = np.dot(Loo.T, Lvv)
    oovv = lib.unpack_tril(oovv_packed).reshape(nocc, nocc, nvir, nvir)

    return ((Loo.reshape(naux, nocc, nocc), Lov_resh, Lvv, oooo, ovoo, ovov, oovv),
            (Loo, Lov_flat, Lvv, naux, nvir))


def _build_eris_tensors_from_Lpq_bwd(nocc, res, cotangents):
    Loo, Lov_flat, Lvv, naux, nvir = res
    dtype = Loo.dtype
    bar_Loo_resh, bar_Lov_resh, bar_Lvv, bar_oooo, bar_ovoo, bar_ovov, bar_oovv = cotangents

    # Aggregate bar contributions to the L-form tensors.
    bar_Loo = np.zeros_like(Loo)
    bar_Lov_flat = np.zeros_like(Lov_flat)
    bar_Lvv_packed = np.zeros_like(Lvv)

    if not _is_zero_cot(bar_Loo_resh):
        bar_Loo = bar_Loo + bar_Loo_resh.reshape(naux, -1)
    if not _is_zero_cot(bar_Lov_resh):
        bar_Lov_flat = bar_Lov_flat + bar_Lov_resh.reshape(naux, -1)
    if not _is_zero_cot(bar_Lvv):
        bar_Lvv_packed = bar_Lvv_packed + bar_Lvv

    # oooo = Loo.T @ Loo  ->  bar_Loo += Loo @ (bar_oooo + bar_oooo.T)
    if not _is_zero_cot(bar_oooo):
        bf = bar_oooo.reshape(nocc * nocc, nocc * nocc)
        bar_Loo = bar_Loo + np.dot(Loo, bf + bf.T)

    # ovoo = Lov_flat.T @ Loo:
    #   bar_Lov_flat += Loo @ bar_ovoo.T
    #   bar_Loo      += Lov_flat @ bar_ovoo
    if not _is_zero_cot(bar_ovoo):
        bf = bar_ovoo.reshape(nocc * nvir, nocc * nocc)
        bar_Lov_flat = bar_Lov_flat + np.dot(Loo, bf.T)
        bar_Loo = bar_Loo + np.dot(Lov_flat, bf)

    # ovov = Lov_flat.T @ Lov_flat -> bar_Lov_flat += Lov_flat @ (bar + bar.T)
    if not _is_zero_cot(bar_ovov):
        bf = bar_ovov.reshape(nocc * nvir, nocc * nvir)
        bar_Lov_flat = bar_Lov_flat + np.dot(Lov_flat, bf + bf.T)

    # oovv = unpack_tril(Loo.T @ Lvv):
    #   bar_oovv_packed = adjoint_unpack_tril(bar_oovv_flat)   (size nocc^2 x nvir_pair)
    #   bar_Loo += Lvv @ bar_oovv_packed.T
    #   bar_Lvv += Loo @ bar_oovv_packed
    if not _is_zero_cot(bar_oovv):
        bf = bar_oovv.reshape(nocc * nocc, nvir, nvir)
        bar_oovv_packed = _adjoint_unpack_tril_last2(bf, nvir)
        bar_Loo = bar_Loo + np.dot(Lvv, bar_oovv_packed.T)
        bar_Lvv_packed = bar_Lvv_packed + np.dot(Loo, bar_oovv_packed)

    # bar_Lvv_packed -> bar_Lvv_full (scatter into lower triangle).
    bar_Lvv_full = _scatter_lower_tril_last2(bar_Lvv_packed, nvir)

    # Scatter all three back into bar_Lpq.
    nmo = nocc + nvir
    bar_Lpq = np.zeros((naux, nmo, nmo), dtype=dtype)
    bar_Lpq = bar_Lpq.at[:, :nocc, :nocc].add(
        bar_Loo.reshape(naux, nocc, nocc)
    )
    bar_Lpq = bar_Lpq.at[:, :nocc, nocc:].add(
        bar_Lov_flat.reshape(naux, nocc, nvir)
    )
    bar_Lpq = bar_Lpq.at[:, nocc:, nocc:].add(bar_Lvv_full)
    return (bar_Lpq,)


_build_eris_tensors_from_Lpq.defvjp(
    _build_eris_tensors_from_Lpq_fwd,
    _build_eris_tensors_from_Lpq_bwd,
)

class RCCSD(dfccsd.RCCSD):
    def ao2mo(self, mo_coeff=None, fockao=None):
        return _make_df_eris_incore(self, mo_coeff, fockao)

class RDCSD(dfdcsd.RDCSD):
    def ao2mo(self, mo_coeff=None, fockao=None):
        return _make_df_eris_incore(self, mo_coeff, fockao)

class _ChemistsERIs(dfccsd._ChemistsERIs):
    def _common_init_(self, mycc, mo_coeff=None, fockao=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        if fockao is None:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            vhf = mycc._scf.get_veff(mycc.mol, dm)
            fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        #self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        self.e_hf = mycc._scf.e_tot
        self.nocc = mycc.nocc
        self.mol = mycc.mol

        self.mo_energy = np.diagonal(self.fock).real
        return self

def _make_df_eris_incore(cc, mo_coeff=None, fockao=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff, fockao)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    atmlst = getattr(cc, '_domain_atmlst', None)
    Lpq = lno_base.transform_df_to_mo(
        cc._scf, mo, ijslice, aosym='s2', mosym='s1', atmlst=atmlst
    ).reshape(-1, nmo, nmo)

    if _USE_CUSTOM_VJP_AO2MO_DF_ERIS:
        # Single custom_vjp boundary hides Loo/Lov/Lvv + matmul outputs from
        # the outer LNO replay's recorded jaxpr.
        Loo, Lov_r, Lvv, oooo, ovoo, ovov, oovv = _build_eris_tensors_from_Lpq(
            Lpq, int(nocc),
        )
        eris.Loo = Loo
        eris.Lov = Lov_r
        eris.Lvv = Lvv
        eris.oooo = oooo
        eris.ovoo = ovoo
        eris.ovov = ovov
        eris.ovvo = ovov.transpose(0, 1, 3, 2)
        eris.oovv = oovv
    else:
        naux = Lpq.shape[0]
        Loo = Lpq[:, :nocc, :nocc].reshape(naux, -1)
        Lov = Lpq[:, :nocc, nocc:].reshape(naux, -1)
        eris.Loo = Loo.reshape(naux, nocc, nocc)
        eris.Lov = Lov.reshape(naux, nocc, nvir)
        eris.Lvv = Lvv = lib.pack_tril(Lpq[:, nocc:, nocc:])
        eris.oooo = np.dot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)
        eris.ovoo = np.dot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)
        ovov = np.dot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
        eris.ovov = ovov
        eris.ovvo = ovov.transpose(0, 1, 3, 2)
        oovv = np.dot(Loo.T, Lvv)
        eris.oovv = lib.unpack_tril(oovv).reshape(nocc, nocc, nvir, nvir)
    # eris.ovvv is built lazily via eris.get_ovvv_packed() on first access so
    # forward CCSD (which builds tiles from Lov/Lvv) doesn't pay the
    # persistent allocation.  The (T) and lambda paths trigger it on demand.
    Lpq = None
    return eris


def impurity_solve(mf, mo_coeff, lo_coeff, eris=None, frozen=None,
                   frag_prescreen=None,
                   verbose_imp=0, ccsd_t=False, dcsd=False,
                   profile_info=None, profile_pass=None,
                   pt2_fragment_method='mp2',
                   sos_c_os=lno_base.DOMAIN_SOS_MP2_C_OS):
    r'''Solve impurity problem and calculate local correlation energy.

    Args:
        mo_coeff : array
            MOs for which the impurity problem is solved.
        lo_coeff : array
            LOs on the current fragment.
        ccsd_t : bool
            If set to ``True``, CCSD(T) energy is calculated and returned as the third
            item (0 is returned otherwise). Default is ``False``.
        dcsd : bool
            If set to ``True``, the DCSD correlation energy is computed instead
            of CCSD. Default is ``False``.
        frozen : int or list, optional
            Same syntax as ``frozen`` in MP2, CCSD, etc.
        verbose_imp : int
            Verbosity for impurity solver printing. Default is 0.
        pt2_fragment_method : str
            PT2 reference used for the local correction. Default is
            conventional MP2.
        sos_c_os : float
            Opposite-spin scale for SOS-MP2 fragment references.

    Return:
        e_loc_corr_pt2, e_loc_corr_ccsd, e_loc_corr_ccsd_t:
            Local correlation energy at MP2, CCSD, and CCSD(T) levels. Note that
            the CCSD(T) energy is 0 unless ``ccsd_t`` is set to True.
    '''
    if _USE_CUSTOM_VJP_IMPURITY_SOLVE and _mf_supports_impurity_solve_wrap(mf):
        return _impurity_solve_jax(
            mf, mo_coeff, lo_coeff, eris.fock, eris.s1e, frag_prescreen,
            frozen, verbose_imp, ccsd_t, dcsd, profile_info, profile_pass,
            pt2_fragment_method, sos_c_os,
        )
    return _impurity_solve_core(
        mf, mo_coeff, lo_coeff, eris.fock, eris.s1e,
        frozen=frozen, frag_prescreen=frag_prescreen,
        verbose_imp=verbose_imp, ccsd_t=ccsd_t, dcsd=dcsd,
        profile_info=profile_info, profile_pass=profile_pass,
        pt2_fragment_method=pt2_fragment_method,
        sos_c_os=sos_c_os,
    )


# ---------------------------------------------------------------------------
# Whole-impurity_solve custom_vjp wrap.
#
# The forward and backward delegate to ``_impurity_solve_with_state``,
# which synchronizes ``mf`` with the frozen ``scf_state`` snapshot before
# running the numerical body.  Because that snapshot enumerates the
# stateful attributes ``mf.kernel()`` writes, the bwd's ``jax.vjp``
# re-trace sees current-trace tracers for those attributes instead of
# the stale outer-trace tracers that would otherwise ride along on
# ``mf``'s non-pytree fields.
#
# Only ``eris.fock`` and ``eris.s1e`` are used by the body (mcc.ao2mo
# rebuilds its own eris from cc._scf); they are passed as plain arrays so
# the unregistered ``_LNOERIS`` Python class never enters JAX tracing.
#
# nondiff_argnums (frozen, verbose_imp, ccsd_t, dcsd, profile_info,
# profile_pass, pt2_fragment_method, sos_c_os) are passed through unchanged;
# profile_info is intentionally
# set to None inside the bwd so the replay does not pollute the dict the
# forward already filled in.
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11, 12, 13))
def _impurity_solve_jax(mf, mo_coeff, lo_coeff, fock, s1e, frag_prescreen,
                        frozen, verbose_imp, ccsd_t, dcsd,
                        profile_info, profile_pass,
                        pt2_fragment_method, sos_c_os):
    return _impurity_solve_core(
        mf, mo_coeff, lo_coeff, fock, s1e,
        frozen=frozen, frag_prescreen=frag_prescreen,
        verbose_imp=verbose_imp, ccsd_t=ccsd_t, dcsd=dcsd,
        profile_info=profile_info, profile_pass=profile_pass,
        pt2_fragment_method=pt2_fragment_method,
        sos_c_os=sos_c_os,
    )


def _impurity_solve_jax_fwd(mf, mo_coeff, lo_coeff, fock, s1e, frag_prescreen,
                            frozen, verbose_imp, ccsd_t, dcsd,
                            profile_info, profile_pass,
                            pt2_fragment_method, sos_c_os):
    out = _impurity_solve_core(
        mf, mo_coeff, lo_coeff, fock, s1e,
        frozen=frozen, frag_prescreen=frag_prescreen,
        verbose_imp=verbose_imp, ccsd_t=ccsd_t, dcsd=dcsd,
        profile_info=profile_info, profile_pass=profile_pass,
        pt2_fragment_method=pt2_fragment_method,
        sos_c_os=sos_c_os,
    )
    res = (mf, mo_coeff, lo_coeff, fock, s1e, frag_prescreen)
    return out, res


def _impurity_solve_jax_bwd(frozen, verbose_imp, ccsd_t, dcsd,
                            profile_info, profile_pass,
                            pt2_fragment_method, sos_c_os, res, ybar):
    mf, mo_coeff, lo_coeff, fock, s1e, frag_prescreen = res

    def fn(mf_, mo_coeff_, lo_coeff_, fock_, s1e_, frag_prescreen_):
        return _impurity_solve_core(
            mf_, mo_coeff_, lo_coeff_, fock_, s1e_,
            frozen=frozen, frag_prescreen=frag_prescreen_,
            verbose_imp=verbose_imp, ccsd_t=ccsd_t, dcsd=dcsd,
            profile_info=None, profile_pass=profile_pass,
            pt2_fragment_method=pt2_fragment_method,
            sos_c_os=sos_c_os,
        )

    _, vjp_fn = jax.vjp(fn, mf, mo_coeff, lo_coeff, fock, s1e, frag_prescreen)
    return vjp_fn(ybar)


_impurity_solve_jax.defvjp(_impurity_solve_jax_fwd, _impurity_solve_jax_bwd)


def _impurity_solve_core(mf, mo_coeff, lo_coeff, fock, s1e, frozen=None,
                         frag_prescreen=None,
                         verbose_imp=0, ccsd_t=False, dcsd=False,
                         profile_info=None, profile_pass=None,
                         pt2_fragment_method='mp2',
                         sos_c_os=lno_base.DOMAIN_SOS_MP2_C_OS):
    '''Numerical core of :func:`impurity_solve`; see that function for docs.

    Takes ``fock`` and ``s1e`` directly (rather than the parent ``eris``
    container) so the function is callable from a ``jax.custom_vjp`` whose
    diff inputs must be plain pytrees of arrays.
    '''
    log = logger.new_logger(mf)
    maskocc = mf.mo_occ > lno_base.THRESH_OCC
    nocc = numpy.count_nonzero(maskocc)
    nmo = mf.mo_occ.size

    frozen, maskact = get_maskact(frozen, nmo)

    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]
    nlo = lo_coeff.shape[1]
    prjlo = reduce(np.dot, (lo_coeff.T, s1e, orbactocc))

    log.info('    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir',
             nlo, nactocc+nactvir, nmo, nactocc, nactvir)

    # solve impurity problem
    if dcsd:
        mcc = RDCSD(mf, mo_coeff=mo_coeff, frozen=frozen)
    else:
        mcc = RCCSD(mf, mo_coeff=mo_coeff, frozen=frozen)
    diis_space = int(os.environ.get('PYSCFAD_LNO_CCSD_DIIS_SPACE', '3'))
    if diis_space <= 0:
        mcc.diis = False
    else:
        mcc.incore_complete = True
        mcc.diis_space = diis_space
    mcc._domain_atmlst = None if frag_prescreen is None else frag_prescreen.get('extended_primary_domain')
    mcc.e_hf = mf.e_tot  #avoid MP2 recompute e_hf
    mcc.profile_pass = profile_pass
    mcc.lno_ccsd_t_timing = _should_print_t_timing(
        verbose_imp, getattr(getattr(mf, 'mol', None), 'verbose', 0),
    )
    total_start = time.perf_counter()
    phase_start = time.perf_counter()
    imp_eris = mcc.ao2mo(fockao=fock)
    phase_times = {'ao2mo_s': time.perf_counter() - phase_start}

    # Method-matched PT2 reference energy.  DLNO correction subtracts this
    # value before adding the selected full/domain correction.
    phase_start = time.perf_counter()
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    elcorr_pt2 = _pt2_fragment_energy(
        imp_eris, t2, prjlo, pt2_fragment_method, sos_c_os
    )
    phase_times['mp2_s'] = time.perf_counter() - phase_start

    # CCSD fragment energy
    phase_start = time.perf_counter()
    t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
    elcorr_cc = ccsd_fragment_energy(imp_eris, t1, t2, prjlo)
    phase_times['ccsd_s'] = time.perf_counter() - phase_start

    if ccsd_t and not dcsd:
        #for tests
        #from pyscfad.lno import ccsd_t_slow
        #elcorr_cc_t = ccsd_t_slow.kernel(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        #elcorr_cc_t = ccsd_t_slow.iterative_kernel(
        #    mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        #from pyscfad.cc import gccsd_t
        #elcorr_cc_t = gccsd_t.kernel(mcc, prjlo, t1=t1, t2=t2)
        phase_start = time.perf_counter()
        elcorr_cc_t = ccsd_t_mod.kernel(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        phase_times['triples_s'] = time.perf_counter() - phase_start
        if (profile_pass != 'backward replay' and
                (_should_print_forward_t_energy(verbose_imp, profile_pass) or
                 mcc.lno_ccsd_t_timing)):
            _print_forward_t_energy(elcorr_cc_t, phase_times['triples_s'])
    else:
        elcorr_cc_t = 0.
        phase_times['triples_s'] = 0.0

    phase_times['total_s'] = time.perf_counter() - total_start
    if profile_info is not None:
        profile_info['solver_occ'] = int(nactocc)
        profile_info['solver_vir'] = int(nactvir)
        profile_info['solver_mo'] = int(nactocc + nactvir)
        profile_info['phase_times'] = phase_times

    t1 = t2 = imp_eris = mcc = None
    del log
    return (elcorr_pt2, elcorr_cc, elcorr_cc_t)

def get_maskact(frozen, nmo):
    if frozen is None:
        frozen = 0
    elif len(frozen) == 0:
        frozen = 0

    if numpy.isscalar(frozen):
        maskact = numpy.hstack([numpy.zeros(frozen,dtype=bool),
                                numpy.ones(nmo-frozen,dtype=bool)])
    else:
        maskact = numpy.array([i not in frozen for i in range(nmo)])
    return frozen, maskact

def _fragment_pq_contractions(amp, ovov):
    """Compute the two fragment contractions in mp2/ccsd_fragment_energy as
    explicit GEMMs.  Equivalent to::

        2*einsum('pjab,qajb->pq', amp, ovov) - einsum('pjab,qbja->pq', amp, ovov)

    Reshaping to a single matmul avoids XLA selecting a Triton kernel whose
    shared-memory request exceeds the per-block limit for large virtual
    spaces.
    """
    p = amp.shape[0]
    q = ovov.shape[0]
    amp_flat = amp.reshape(p, -1)
    ovov_qjab = ovov.transpose(0, 2, 1, 3).reshape(q, -1)
    ovov_qjab_alt = ovov.transpose(0, 2, 3, 1).reshape(q, -1)
    return 2.0 * (amp_flat @ ovov_qjab.T) - (amp_flat @ ovov_qjab_alt.T)


def _is_sos_pt2_fragment_method(method):
    method = str(method).lower().replace('_', '-')
    return method in _SOS_PT2_FRAGMENT_METHODS


def _pt2_fragment_energy(eris, t2, prj, method='mp2',
                         sos_c_os=lno_base.DOMAIN_SOS_MP2_C_OS):
    if _is_sos_pt2_fragment_method(method):
        return sos_mp2_fragment_energy(eris, t2, prj, c_os=sos_c_os)
    return mp2_fragment_energy(eris, t2, prj)


def mp2_fragment_energy(eris, t2, prj):
    """Compute the MP2 fragment energy contribution.

    eij[p,q] = 2 * einsum('pjab,qajb->pq', t2, ovov)
             -     einsum('pjab,qbja->pq', t2, ovov)
    e2       = einsum('ij,ij', eij, prj.T @ prj)
    """
    ovov = np.asarray(eris.ovov)
    if _USE_CUSTOM_VJP_MP2_FRAG:
        return _mp2_fragment_energy_jax(ovov, t2, prj)
    m = np.dot(prj.T, prj)
    eij = _fragment_pq_contractions(t2, ovov)
    return np.einsum('ij,ij', eij, m)


def sos_mp2_fragment_energy(eris, t2, prj,
                            c_os=lno_base.DOMAIN_SOS_MP2_C_OS):
    """Compute the scaled opposite-spin MP2 fragment energy contribution.

    eij[p,q] = c_os * einsum('pjab,qajb->pq', t2, ovov)
    e2       = einsum('ij,ij', eij, prj.T @ prj)
    """
    ovov = np.asarray(eris.ovov)
    return _sos_mp2_fragment_energy_jax(ovov, t2, prj, c_os)


# ---------------------------------------------------------------------------
# MP2 fragment energy wrapped as jax.custom_vjp.  Inputs are (ovov, t2, prj);
# output is a scalar.  Hand-coded backward avoids retaining the eij matrix
# and the two _fragment_pq_contractions reshaped GEMMs in the JAX trace.
# ---------------------------------------------------------------------------
@custom_vjp
def _mp2_fragment_energy_jax(ovov, t2, prj):
    m = np.dot(prj.T, prj)
    eij = _fragment_pq_contractions(t2, ovov)
    return np.einsum('ij,ij', eij, m)


def _mp2_fragment_energy_jax_fwd(ovov, t2, prj):
    m = np.dot(prj.T, prj)
    eij = _fragment_pq_contractions(t2, ovov)
    e2 = np.einsum('ij,ij', eij, m)
    return e2, (ovov, t2, prj, m, eij)


def _mp2_fragment_energy_jax_bwd(res, bar_e2):
    ovov, t2, prj, m, eij = res
    if _is_zero_cot(bar_e2):
        # Shapes: ovov (nocc, nvir, nocc, nvir), t2 (nocc, nocc, nvir, nvir),
        # prj (nlo, nocc).
        return (np.zeros_like(ovov), np.zeros_like(t2), np.zeros_like(prj))
    bar_e2 = np.asarray(bar_e2)

    # e2 = einsum('ij,ij', eij, m)
    # -> bar_eij[p,q] = bar_e2 * m[p,q],  bar_m[p,q] = bar_e2 * eij[p,q]
    bar_eij = bar_e2 * m
    bar_m = bar_e2 * eij

    # eij[p,q] = 2*sum_{j,a,b} t2[p,j,a,b] * ovov[q,a,j,b]
    #         -   sum_{j,a,b} t2[p,j,a,b] * ovov[q,b,j,a]
    # -> bar_t2[p,j,a,b] = sum_q bar_eij[p,q] * (2*ovov[q,a,j,b] - ovov[q,b,j,a])
    # -> bar_ovov[q,a,j,b] += 2 * sum_p bar_eij[p,q] * t2[p,j,a,b]
    # -> bar_ovov[q,b,j,a] += -   sum_p bar_eij[p,q] * t2[p,j,a,b]
    bar_t2 = (
        2 * np.einsum('pq,qajb->pjab', bar_eij, ovov)
        -     np.einsum('pq,qbja->pjab', bar_eij, ovov)
    )
    bar_ovov = (
        2 * np.einsum('pq,pjab->qajb', bar_eij, t2)
        -     np.einsum('pq,pjab->qbja', bar_eij, t2)
    )

    # m = prj.T @ prj -> m[p,q] = sum_l prj[l,p] * prj[l,q]
    # -> bar_prj[l,p] = sum_q (bar_m[p,q] + bar_m[q,p]) * prj[l,q]
    bar_prj = np.dot(prj, bar_m + bar_m.T)

    return bar_ovov, bar_t2, bar_prj


_mp2_fragment_energy_jax.defvjp(
    _mp2_fragment_energy_jax_fwd,
    _mp2_fragment_energy_jax_bwd,
)


@partial(custom_vjp, nondiff_argnums=(3,))
def _sos_mp2_fragment_energy_jax(ovov, t2, prj, c_os):
    m = np.dot(prj.T, prj)
    eij = np.einsum('pjab,qajb->pq', t2, ovov)
    return c_os * np.einsum('ij,ij', eij, m)


def _sos_mp2_fragment_energy_jax_fwd(ovov, t2, prj, c_os):
    m = np.dot(prj.T, prj)
    eij = np.einsum('pjab,qajb->pq', t2, ovov)
    e2 = c_os * np.einsum('ij,ij', eij, m)
    return e2, (ovov, t2, prj, m, eij)


def _sos_mp2_fragment_energy_jax_bwd(c_os, res, bar_e2):
    ovov, t2, prj, m, eij = res
    if _is_zero_cot(bar_e2):
        return (np.zeros_like(ovov), np.zeros_like(t2), np.zeros_like(prj))
    bar_e2 = np.asarray(bar_e2) * c_os

    # e2/c_os = einsum('ij,ij', eij, m)
    # eij[p,q] = sum_{j,a,b} t2[p,j,a,b] * ovov[q,a,j,b]
    bar_eij = bar_e2 * m
    bar_m = bar_e2 * eij

    bar_t2 = np.einsum('pq,qajb->pjab', bar_eij, ovov)
    bar_ovov = np.einsum('pq,pjab->qajb', bar_eij, t2)
    bar_prj = np.dot(prj, bar_m + bar_m.T)

    return bar_ovov, bar_t2, bar_prj


_sos_mp2_fragment_energy_jax.defvjp(
    _sos_mp2_fragment_energy_jax_fwd,
    _sos_mp2_fragment_energy_jax_bwd,
)

_USE_CUSTOM_VJP_CCSD_FRAG = True


# ---------------------------------------------------------------------------
# CCSD fragment energy wrapped as jax.custom_vjp.
#
# Forward:
#     m[p,q]    = (prj.T @ prj)[p,q]
#     eij1[p,q] = 2 * Σ_a t1[p,a] fov[q,a]
#     tau[i,j,a,b]   = t1[i,a] t1[j,b] + t2[i,j,a,b]
#     eij2[p,q] = 2 Σ_{j,a,b} tau[p,j,a,b] ovov[q,a,j,b]
#               -   Σ_{j,a,b} tau[p,j,a,b] ovov[q,b,j,a]
#     e2        = Σ_{p,q} (eij1+eij2)[p,q] m[p,q]
#
# Backward (closed-form chain rule).  Let ē = bar_e2 (scalar).
#   bar_eij[p,q] = ē m[p,q]
#   bar_m[p,q]   = ē (eij1+eij2)[p,q]
#
#   From eij1:
#     bar_t1   += 2 bar_eij @ fov               # 'ij,ja->ia'
#     bar_fov   = 2 bar_eij.T @ t1              # 'ij,ia->ja'
#
#   From eij2 (same shape as the MP2 fragment energy bwd, swap t2 → tau):
#     bar_tau  = 2 einsum('pq,qajb->pjab', bar_eij, ovov)
#              -   einsum('pq,qbja->pjab', bar_eij, ovov)
#     bar_ovov = 2 einsum('pq,pjab->qajb', bar_eij, tau)
#              -   einsum('pq,pjab->qbja', bar_eij, tau)
#
#   From tau = t1 t1 + t2:
#     bar_t2  = bar_tau
#     bar_t1 += einsum('ijab,jb->ia', bar_tau, t1)
#            +  einsum('jiba,jb->ia', bar_tau, t1)
#
#   From m = prj.T @ prj:
#     bar_prj = prj @ (bar_m + bar_m.T)
# ---------------------------------------------------------------------------

@custom_vjp
def _ccsd_fragment_energy_jax(t1, t2, fov, ovov, prj):
    m = np.dot(prj.T, prj)
    eij = 2 * np.einsum('ia,ja->ij', t1, fov)
    tau = np.einsum('ia,jb->ijab', t1, t1) + t2
    eij = eij + _fragment_pq_contractions(tau, ovov)
    return np.einsum('ij,ij', eij, m)


def _ccsd_fragment_energy_jax_fwd(t1, t2, fov, ovov, prj):
    m = np.dot(prj.T, prj)
    eij1 = 2 * np.einsum('ia,ja->ij', t1, fov)
    tau = np.einsum('ia,jb->ijab', t1, t1) + t2
    eij2 = _fragment_pq_contractions(tau, ovov)
    e2 = np.einsum('ij,ij', eij1 + eij2, m)
    # ``tau`` and ``eij1+eij2`` are reused in the bwd; save them to skip the
    # einsum/transpose retraces.
    return e2, (t1, t2, fov, ovov, prj, tau, m, eij1 + eij2)


def _ccsd_fragment_energy_jax_bwd(res, bar_e2):
    t1, t2, fov, ovov, prj, tau, m, eij = res

    bar_eij = bar_e2 * m
    bar_m = bar_e2 * eij

    # eij1 contributions
    bar_t1 = 2 * np.einsum('ij,ja->ia', bar_eij, fov)
    bar_fov = 2 * np.einsum('ij,ia->ja', bar_eij, t1)

    # eij2 contributions to tau and ovov
    bar_tau = (
        2 * np.einsum('pq,qajb->pjab', bar_eij, ovov)
        -     np.einsum('pq,qbja->pjab', bar_eij, ovov)
    )
    bar_ovov = (
        2 * np.einsum('pq,pjab->qajb', bar_eij, tau)
        -     np.einsum('pq,pjab->qbja', bar_eij, tau)
    )

    # tau = t1 t1 + t2
    bar_t2 = bar_tau
    bar_t1 = bar_t1 + np.einsum('ijab,jb->ia', bar_tau, t1)
    bar_t1 = bar_t1 + np.einsum('jiba,jb->ia', bar_tau, t1)

    # m = prj.T @ prj
    bar_prj = np.dot(prj, bar_m + bar_m.T)

    return bar_t1, bar_t2, bar_fov, bar_ovov, bar_prj


_ccsd_fragment_energy_jax.defvjp(
    _ccsd_fragment_energy_jax_fwd,
    _ccsd_fragment_energy_jax_bwd,
)


def ccsd_fragment_energy(eris, t1, t2, prj):
    nocc = t1.shape[0]
    fov = eris.fock[:nocc,nocc:]
    ovov = np.asarray(eris.ovov)
    if _USE_CUSTOM_VJP_CCSD_FRAG:
        return _ccsd_fragment_energy_jax(t1, t2, fov, ovov, prj)
    m = np.dot(prj.T, prj)
    eij = 2*np.einsum('ia,ja->ij', t1, fov)
    tau = np.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    eij += _fragment_pq_contractions(tau, ovov)
    e2 = np.einsum('ij,ij', eij, m)
    return e2

class LNOCCSD(lno_base.LNO):
    def __init__(self, mf, thresh=1e-4, frozen=None, fock=None, s1e=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, fock=fock, s1e=s1e, **kwargs)
        self.efrag_cc = None
        self.efrag_pt2 = None
        self.efrag_pt2_domain = None
        self.efrag_cc_t = None
        self.ccsd_t = False
        self.dcsd = False
        self.pt2_fragment_method = 'mp2'
        self.pt2_fragment_sos_c_os = lno_base.DOMAIN_SOS_MP2_C_OS

    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None,
                       frag_prescreen=None, profile_info=None):
        return impurity_solve(mf, mo_coeff, lo_coeff, eris=eris, frozen=frozen,
                              frag_prescreen=frag_prescreen,
                              verbose_imp=self.verbose_imp, ccsd_t=self.ccsd_t,
                              dcsd=self.dcsd, profile_info=profile_info,
                              profile_pass=getattr(self, 'profile_pass', None),
                              pt2_fragment_method=getattr(
                                  self, 'pt2_fragment_method', 'mp2'),
                              sos_c_os=getattr(
                                  self, 'pt2_fragment_sos_c_os',
                                  lno_base.DOMAIN_SOS_MP2_C_OS))

    def _post_proc(self, frag_res, frag_wghtlist):
        ''' Post processing results returned by ``impurity_solve`` collected in ``frag_res``.
        '''
        efrag_pt2 = efrag_cc = efrag_cc_t = efrag_pt2_domain = 0.0
        has_domain_pt2 = False
        for i, res in enumerate(frag_res):
            if res is not None:
                efrag_pt2 += res[0] * frag_wghtlist[i]
                efrag_cc += res[1] * frag_wghtlist[i]
                efrag_cc_t += res[2] * frag_wghtlist[i]
                if len(res) > 3:
                    efrag_pt2_domain += res[3] * frag_wghtlist[i]
                    has_domain_pt2 = True
        self.efrag_pt2  = efrag_pt2
        self.efrag_pt2_domain = efrag_pt2_domain if has_domain_pt2 else None
        self.efrag_cc   = efrag_cc
        self.efrag_cc_t = efrag_cc_t

    @property
    def e_corr(self):
        return self.e_corr_ccsd + self.e_corr_ccsd_t

    @property
    def e_corr_ccsd(self):
        e_corr = self.efrag_cc
        return e_corr

    @property
    def e_corr_pt2(self):
        e_corr = self.efrag_pt2
        return e_corr

    @property
    def e_corr_pt2_domain(self):
        e_corr = self.efrag_pt2_domain
        if e_corr is None:
            e_corr = self.efrag_pt2
        return e_corr

    @property
    def e_corr_ccsd_t(self):
        e_corr = self.efrag_cc_t
        return e_corr

    @property
    def e_tot_ccsd(self):
        return self.e_corr_ccsd + self._scf.e_tot

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self._scf.e_tot

    def e_corr_pt2corrected(self, ept2):
        return self.e_corr - self.e_corr_pt2 + ept2

    def e_tot_pt2corrected(self, ept2):
        return self._scf.e_tot + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_pt2corrected(self, ept2):
        return self.e_corr_ccsd - self.e_corr_pt2 + ept2

    def e_tot_ccsd_pt2corrected(self, ept2):
        return self._scf.e_tot_ccsd + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_t_pt2corrected(self, ept2):
        return self.e_corr_ccsd_t - self.e_corr_pt2 + ept2

    def e_tot_ccsd_t_pt2corrected(self, ept2):
        return self._scf.e_tot_ccsd_t + self.e_corr_pt2corrected(ept2)

class LNOCCSD_T(LNOCCSD):
    def __init__(self, mf, thresh=1e-4, frozen=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, **kwargs)
        self.ccsd_t = True
