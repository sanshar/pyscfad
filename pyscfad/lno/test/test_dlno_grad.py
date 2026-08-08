import warnings

import jax
import jax.numpy as jnp
import numpy

from pyscfad import config, df, gto, scf
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_first_order_custom', True)
config.update('pyscfad_ccsd_implicit_diff', True)
config.update('pyscfad_dfccsd_custom_response', True)


def _build_water():
    mol = gto.Mole(
        atom='''
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        ''',
        basis='sto3g',
        verbose=0,
        max_memory=4000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _prepare_outcore_cderi(mol, cderi_file):
    with_df = df.DF(mol, incore=False)
    with_df._cderi_to_save = str(cderi_file)
    with_df.max_memory = mol.max_memory
    with_df.build()


def _density_fit_from_cderi(mol, cderi_file):
    mf = scf.RHF(mol)
    with_df = df.DF(mol, incore=False)
    with_df.max_memory = mol.max_memory
    with_df.auxmol = df.addons.make_auxmol(mol, with_df.auxbasis)
    with_df._cderi_to_save = str(cderi_file)
    with_df._cderi = numpy.zeros((0, 0))
    with_df._prefer_cderi_to_save = True
    return mf.density_fit(with_df=with_df)


def _build_local_orbitals_and_fragments(mf, thresh):
    lo_coeff = lnoccsd.LNOCCSD(mf, thresh=thresh, frozen=0).get_lo(lo_type='iao')
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def _make_solver(mf, thresh, dlno_data=None):
    if dlno_data is None:
        mycc = lnoccsd.LNOCCSD(mf, thresh=thresh, frozen=0)
    else:
        mycc = DLNOCCSD(mf, thresh=thresh, frozen=0,
                        dlno_prescreen_data=dlno_data)
    mycc.thresh_occ = thresh
    mycc.thresh_vir = thresh
    mycc.lo_type = 'iao'
    mycc.no_type = 'ie'
    mycc.ccsd_t = False
    return mycc


class _FakeEris:
    def __init__(self, ovov):
        self.ovov = ovov


def test_projected_sos_fragment_energy_matches_direct_os_term():
    ovov = jnp.asarray(
        numpy.arange(3 * 2 * 3 * 2, dtype=float).reshape(3, 2, 3, 2) / 23.0
        - 0.4
    )
    t2 = jnp.asarray(
        numpy.arange(3 * 3 * 2 * 2, dtype=float).reshape(3, 3, 2, 2) / 19.0
        - 0.7
    )
    prj = jnp.asarray([
        [0.8, 0.3, -0.2],
        [0.1, 0.5, 0.4],
    ])
    c_os = 1.3

    m = prj.T @ prj
    direct = jnp.einsum('pq,pjab,qajb->', m, t2, ovov)
    exchange = jnp.einsum('pq,pjab,qbja->', m, t2, ovov)

    e_sos = lnoccsd._sos_mp2_fragment_energy_jax(ovov, t2, prj, c_os)
    e_full = lnoccsd.mp2_fragment_energy(_FakeEris(ovov), t2, prj)

    assert abs(e_sos - c_os * direct) < 1e-12
    assert abs(e_full - (2.0 * direct - exchange)) < 1e-12
    assert abs(e_full - e_sos / c_os) > 1e-6

    def energy_t2(t2_):
        return lnoccsd._sos_mp2_fragment_energy_jax(ovov, t2_, prj, c_os)

    grad_t2 = jax.grad(energy_t2)(t2)
    idx = (1, 2, 0, 1)
    eps = 1e-5
    fd = (
        energy_t2(t2.at[idx].add(eps))
        - energy_t2(t2.at[idx].add(-eps))
    ) / (2.0 * eps)
    assert abs(grad_t2[idx] - fd) < 1e-7


def _build_static_topology(mol, thresh, cderi_file):
    mf = _density_fit_from_cderi(mol, cderi_file)
    mf.kernel()
    lo_coeff, frag_lolist = _build_local_orbitals_and_fragments(mf, thresh)
    topology = build_dlno_prescreen_data(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=0.0,
        pao_bp_domain_thr=0.0,
        domain_pao_thr=0.0,
        pair_energy_thr=0.0,
        multipole_order=2,
    )
    return frag_lolist, topology


def test_full_scope_mp2_correction_is_method_consistent_for_full_domains(tmp_path):
    mol = _build_water()
    thresh = 0.0
    cderi_file = tmp_path / 'water-cderi-full-scope.h5'
    _prepare_outcore_cderi(mol, cderi_file)
    frag_lolist, _ = _build_static_topology(mol, thresh, cderi_file)

    def build_mf(mol_):
        mf = _density_fit_from_cderi(mol_, cderi_file)
        mf.kernel()
        return mf

    common_kwargs = dict(
        build_mf=build_mf,
        frag_lolist=frag_lolist,
        mp2_correction_scope='full',
        frozen=0,
        thresh_occ=thresh,
        thresh_vir=thresh,
        lo_type='iao',
        no_type='ie',
        ccsd_t=False,
        lmo_bp_domain_thr=0.0,
        pao_bp_domain_thr=0.0,
        domain_pao_thr=0.0,
        pair_energy_thr=0.0,
        multipole_order=2,
    )
    e_uncorrected, _ = DLNOCCSD.value_and_grad(
        mol, include_mp2_correction=False, **common_kwargs
    )

    for method in ('mp2', 'sos'):
        e_corrected, _ = DLNOCCSD.value_and_grad(
            mol,
            include_mp2_correction=True,
            mp2_correction_method=method,
            **common_kwargs,
        )
        assert abs(e_corrected - e_uncorrected) < 1e-6


def test_dlno_ccsd_gradient_matches_parent_lno_for_full_domains(tmp_path):
    mol = _build_water()
    thresh = 0.0
    cderi_file = tmp_path / 'water-cderi.h5'
    _prepare_outcore_cderi(mol, cderi_file)
    static_frag_lolist, static_topology = _build_static_topology(
        mol, thresh, cderi_file
    )

    def lno_energy(mol_):
        mf = _density_fit_from_cderi(mol_, cderi_file)
        ehf = mf.kernel()
        lo_coeff, _ = _build_local_orbitals_and_fragments(mf, thresh)
        mycc = _make_solver(mf, thresh)
        mycc.kernel(orbloc=lo_coeff)
        return ehf + mycc.e_corr

    def dlno_energy(mol_):
        mf = _density_fit_from_cderi(mol_, cderi_file)
        ehf = mf.kernel()
        lo_coeff, _ = _build_local_orbitals_and_fragments(mf, thresh)
        dlno_data = rebuild_dlno_prescreen_data(
            mf, lo_coeff, static_topology, frozen=0
        )
        mycc = _make_solver(mf, thresh, dlno_data=dlno_data)
        mycc.kernel(frag_lolist=static_frag_lolist, orbloc=lo_coeff)
        return ehf + mycc.e_corr

    e_lno, g_lno = jax.value_and_grad(lno_energy)(mol)
    e_dlno, g_dlno = jax.value_and_grad(dlno_energy)(mol)

    assert abs(e_dlno - e_lno) < 1e-6
    assert numpy.max(numpy.abs(g_dlno.coords - g_lno.coords)) < 1e-6
