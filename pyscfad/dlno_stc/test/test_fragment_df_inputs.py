"""Molecular W -> LIS -> CCSD(T) response with cached integral pullbacks."""
from types import SimpleNamespace

import h5py
import jax
import numpy
import pytest

from pyscfad import config_update, df, scf
from pyscfad.lno import _df_direct, _df_shared, _df_symmetric
from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.adjoint import fix_restart_mo_phases
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.tools.linear_solver import eager_gmres


@pytest.fixture(scope='module')
def molecular_case(tmp_path_factory):
    native()
    mol = accept.molecule()
    mol.max_memory = 10000
    path = str(tmp_path_factory.mktemp('fragment_df')/'scf.h5')
    integrals = df.DF(mol, incore=False)
    integrals._cderi_to_save = path
    integrals.build()
    with h5py.File(path, 'r') as handle:
        assert set(handle) == {'j3c'}
    def build(current):
        mf = scf.RHF(current).density_fit()
        mf.with_df.attach_outcore_cderi(path)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with eager_gmres():
            mf.kernel()
        return fix_restart_mo_phases(mf)
    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        mf, pullback = jax.vjp(build, mol)
    domains = accept.selection(mf)
    controls = dict(accept.controls(), threads=1)
    static = ccsd.build_static_selections(mf, domains, matrix_kind='W',
        thresh_occ=1e-4, thresh_vir=3e-5,
        baseline_controls=controls, matrix_controls=controls, single_pass=True)
    group = ccsd.build_domain_groups(domains)[0]
    indices = tuple(i for i, _ in ccsd._group_cc_fragments(static, group))[:2]
    assert len(indices) == 2
    assert len(domains.fragments[group.representative_index].extended_ao_indices) < mol.nao
    return SimpleNamespace(mf=mf, pullback=pullback, domains=domains,
        static=static, group=group, indices=indices, controls=controls)


@pytest.mark.parametrize('triples_mode,integral_transform',
                         [('stochastic', 'rotate'), ('deterministic', 'df')])
@pytest.mark.parametrize('auxiliary_tolerance', [None, 1e-4])
def test_explicit_half_matches_default_direct_with_one_integral_pass(
        molecular_case, monkeypatch, triples_mode, integral_transform, auxiliary_tolerance):
    case = molecular_case
    controls = dict(mode=triples_mode, integral_transform=integral_transform)
    if triples_mode == 'stochastic':
        controls.update(nsamples=16, pilot_samples=16, global_seed=71)
    triples = ccsd.normalize_triples_controls(controls, ccsd_t=True)
    static = ccsd.build_static_selections(case.mf, case.domains, matrix_kind='W',
        thresh_occ=1e-4, thresh_vir=3e-5, baseline_controls=case.controls,
        matrix_controls=case.controls, single_pass=True)
    receipts = []
    close = _df_shared.LocalDFReverse.close
    def closed(scope):
        receipts.append(dict(scope.receipt))
        close(scope)
    monkeypatch.setattr(_df_shared.LocalDFReverse, 'close', closed)
    def run(mode):
        if mode is None:
            monkeypatch.delenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', raising=False)
        else:
            monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', mode)
        state = ccsd.driver._initialize(case.mf, case.domains, case.controls,
            with_grad=True, scf_pullback=case.pullback)
        records = ccsd._run_group(state, static, case.group,
            case.controls, case.controls, with_grad=True, verbose_imp=0,
            ccsd_t=True, triples_controls=triples, fragment_indices=case.indices,
            lis_aux_svd_tol=auxiliary_tolerance)
        bars = state['mf_bar'], state['common_bar']
        value, gradient = ccsd.driver._finish(state, with_grad=True, include_hf=False)
        return value, gradient, records, bars
    # Independent baseline: full, unsymmetrized pair adjoints and raw inputs.
    with monkeypatch.context() as original:
        original.setattr(_df_symmetric, 'same_orbital_pair', lambda slices: False)
        reference, ref_gradient, ref_records, ref_bars = run(None)
    assert receipts[-1]['raw_integral_builds'] == 2*(1+len(case.indices))
    reads = []
    generate = _df_direct._generate_local_raw_int3c_blocks
    def once(*args, **kwargs):
        reads.append(1)
        assert len(reads) == 1, 'LIS and backward must reuse H_ED'
        yield from generate(*args, **kwargs)
    monkeypatch.setattr(_df_direct, '_generate_local_raw_int3c_blocks', once)
    value, gradient, records, bars = run('half')
    assert reads == [1]
    receipt = receipts[-1]
    assert receipt['cache_mode'] == 'half'
    assert receipt['raw_source_passes'] == receipt['raw_integral_builds'] == receipt['half_builds'] == 1
    assert receipt['half_rotations'] == len(case.indices)
    assert receipt['half_releases'] == receipt['half_pullbacks'] == 1+len(case.indices)
    assert receipt['half_resident_bytes'] == 0
    assert receipt['consumers'] == 1+len(case.indices)
    assert receipt['symmetric_consumers'] == len(case.indices)
    assert receipt['symmetric_expanded_bytes'] > 0
    assert receipt['adjoint_peak_bytes'] > 0
    assert receipt['adjoint_resident_bytes'] == 0
    assert receipt['geometry_passes'] == receipt['metric_passes'] == 1
    numpy.testing.assert_allclose(value, reference, atol=3e-12, rtol=0)
    numpy.testing.assert_allclose(gradient.coords, ref_gradient.coords, atol=3e-9, rtol=3e-7)
    for got, expected in zip(jax.tree_util.tree_leaves(bars), jax.tree_util.tree_leaves(ref_bars)):
        if getattr(got, 'dtype', None) != jax.dtypes.float0:
            numpy.testing.assert_allclose(got, expected, atol=3e-9, rtol=3e-7)
    for got, expected in zip(records, ref_records):
        numpy.testing.assert_allclose((got.e_mp2_lis, got.e_ccsd, got.e_triples),
            (expected.e_mp2_lis, expected.e_ccsd, expected.e_triples), atol=3e-12, rtol=0)
        assert got.triples_samples == expected.triples_samples
        if auxiliary_tolerance is not None:
            info = got.auxiliary_compression
            assert info['retained_naux'] < info['original_naux']
            assert info['pullback_s'] > 0
    suffix = '' if auxiliary_tolerance is None else '_aux_svd'
    accept.record_result('half_df_'+triples_mode+suffix, {
        'energy_Eh': float(value), 'max_gradient_difference':
            float(numpy.max(numpy.abs(gradient.coords-ref_gradient.coords))),
        'receipt': receipts[-1]})
