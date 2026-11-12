"""Domain LIS selection, target weights, and independent coordinate checks."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update
from pyscfad.dlno_stc import ccsd, lis
from pyscfad.lno import _df_shared
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_fragment_df_inputs import molecular_case


def test_domain_lis_reaches_optional_amplitude_solver(molecular_case, monkeypatch):
    """The optional CC solver must receive local occupations and local ERIs."""
    from pyscfad.dlno_stc import stc_ccsd
    case = molecular_case
    state = ccsd.driver._initialize(case.mf, case.domains, case.controls, with_grad=False)
    _, host, _, frame = ccsd.driver._prepare(
        state, case.domains, case.group, case.controls, False, return_frame=True)
    index = case.indices[0]
    targets = dict(ccsd._group_cc_fragments(case.static, case.group))[index]
    forward = ccsd.backend.solve_matrices(host,
        {'fragment_id': index, 'target_indices': targets}, case.controls, matrix_kind='W')
    static = ccsd._select_current_fragment(state, case.static, index, case.group, frame, forward)
    matrices = {name: jnp.asarray(forward[name]) for name in ('occupied', 'virtual')}
    selected = lis.build_fragment_lis(case.mf, state['common'], static.lis_static,
        index, matrices=matrices, frame=frame,
        frame_fragment_index=case.group.representative_index)
    nocc = selected.active_occupied_coeff.shape[1]
    nvir = selected.active_virtual_coeff.shape[1]
    assert nocc < np.count_nonzero(case.mf.mo_occ > 0)
    calls = []

    def make_solver(controls, fragment_index):
        assert controls == {'seed': 71} and fragment_index == index
        def solve(mcc, eris, t1, t2, *, prjlo):
            assert mcc.nocc == nocc and mcc.nmo == nocc+nvir
            assert t1.shape == (nocc, nvir)
            assert eris.fock.shape == (nocc+nvir, nocc+nvir)
            assert prjlo.shape[1] == nocc
            calls.append(True)
            a, b = mcc.kernel(eris=eris, t1=t1, t2=t2)[1:]
            return a, b, {'seed': controls['seed']}
        return solve

    monkeypatch.setattr(stc_ccsd, 'make_solver', make_solver)
    args = (case.mf, state['common'], static, index, matrices, case.group)
    options = dict(verbose_imp=0, ccsd_t=False, frame=frame)
    expected = ccsd._solve_fragment(*args, **options)
    actual, diagnostics = ccsd._solve_fragment(*args, **options,
        cc_solver='stc', stc_cc_controls={'seed': 71})
    assert calls == [True] and diagnostics['ccsd']['seed'] == 71
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize('kind', ['D', 'W'])
@pytest.mark.parametrize('cutoff', [.45, 0., 1e-30])
def test_domain_selection_keeps_targets_and_stays_in_ed(molecular_case, monkeypatch, kind, cutoff):
    case = molecular_case
    common = ccsd.driver._initialize(case.mf, case.domains, case.controls, with_grad=False)['common']
    frame = lis.build_boys_ed_frame(common, case.domains, case.group.representative_index)
    no, nv = frame.occupied_coeff.shape[1], frame.virtual_coeff.shape[1]
    matrices = dict(occupied=jnp.diag(jnp.linspace(.1, 1., no)),
                    virtual=jnp.diag(jnp.linspace(.1, 1., nv)))
    if kind == 'W':
        matrices['virtual'] = matrices['virtual'].at[0, 0].set(-.8)
    full = cutoff == 0.
    assert not hasattr(lis, '_domain_maps')
    index = case.indices[0]
    selected = lis.build_fragment_lis_selection(case.mf, case.domains, index,
        common=common, frame=frame, matrices=matrices, matrix_kind=kind,
        thresh_occ=cutoff, thresh_vir=cutoff, lis_space='domain')
    fragments = [None]*len(case.static.fragments)
    fragments[index] = selected
    static = lis.LISSelections(case.domains, kind, cutoff, cutoff, 1e-6,
                               tuple(fragments), lis_space='domain')
    value = lis.build_fragment_lis(case.mf, common, static, index, matrices=matrices, frame=frame)
    co, cv = value.active_occupied_coeff, value.active_virtual_coeff
    overlap = common.s1e
    all_coeff = jnp.concatenate((co, cv), axis=1)
    np.testing.assert_allclose(all_coeff.T@overlap@all_coeff, np.eye(all_coeff.shape[1]), atol=2e-12)
    outside = np.setdiff1d(np.arange(case.mf.mol.nao), frame.extended_ao_indices)
    np.testing.assert_array_equal(all_coeff[outside], 0.)
    anchor = value.fragment_occupied_anchor
    projection = anchor.T@overlap@co
    np.testing.assert_allclose(anchor.T@overlap@anchor, np.eye(anchor.shape[1]), atol=2e-12)
    np.testing.assert_allclose(projection@projection.T, np.eye(anchor.shape[1]), atol=2e-12)
    assert value.frozen.size == 0
    assert value.mo_coeff.shape[1] == co.shape[1]+cv.shape[1]
    if full:
        assert (co.shape[1], cv.shape[1]) == (no, nv)
        assert no < common.occupied_coeff.shape[1]
    else:
        assert cv.shape[1] == np.count_nonzero(np.abs(np.diag(matrices['virtual'])) > cutoff)
        assert co.shape[1] <= no
    cc_static = replace(case.static, lis_static=static)
    assert ccsd._fragment_dimensions(cc_static, index) == (co.shape[1], cv.shape[1])
    with pytest.raises(ValueError, match="Only lis_space='domain'"):
        lis.build_fragment_lis(case.mf, common, replace(static, lis_space='global'), index,
                              matrices=matrices, frame=frame)


def test_domain_gradient_finite_difference_and_forward_one_pass(molecular_case, monkeypatch):
    case = molecular_case
    # Opt into shared H_ED reuse for the single-pass checks.
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'half')
    from pyscfad.lno.ccsd import RCCSD
    monkeypatch.setattr(RCCSD, 'conv_tol', 1e-11)
    monkeypatch.setattr(RCCSD, 'conv_tol_normt', 1e-9)
    static = ccsd.build_static_selections(case.mf, case.domains, matrix_kind='W',
        thresh_occ=1e-4, thresh_vir=3e-5, baseline_controls=case.controls,
        matrix_controls=case.controls, lis_space='domain', fragment_indices=case.indices,
        single_pass=False)
    receipts = []
    close = _df_shared.LocalDFReverse.close
    def closed(scope):
        receipts.append(dict(scope.receipt))
        close(scope)
    monkeypatch.setattr(_df_shared.LocalDFReverse, 'close', closed)
    options = dict(baseline_controls=case.controls, matrix_controls=case.controls,
        fragment_indices=case.indices, include_baseline=False, ccsd_t=True,
        triples_controls=dict(mode='deterministic', integral_transform='rotate'))
    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        value, gradient = ccsd.value_and_grad(case.mf.mol, accept.build_mf, static, **options)
        # Rebuild integrals and SCF at each displaced geometry; reuse only fixed
        # domain/rank/anchor labels, never the reference HDF5 integral source.
        step = 2e-4
        energies = []
        for sign in (1, -1):
            coords = np.array(case.mf.mol.coords)
            coords[1, 1] += sign*step
            mol = case.mf.mol.set_geom_(coords, unit='Bohr', inplace=False)
            energies.append(float(ccsd.kernel(accept.build_mf(mol), static, **options)))
    finite_difference = (energies[0]-energies[1])/(2*step)
    np.testing.assert_allclose(gradient.coords[1, 1], finite_difference, atol=2e-7, rtol=3e-4)
    assert len(receipts) == 3
    assert all(r['raw_source_passes'] == r['raw_integral_builds'] == r['half_builds'] == 1
               for r in receipts)
    assert all(r['half_rotations'] == len(case.indices) for r in receipts)
    assert receipts[0]['half_pullbacks'] == 1+len(case.indices)
    assert all(r['half_pullbacks'] == 0 for r in receipts[1:])
    accept.record_result('domain_finite_difference', dict(energy=float(value),
        analytic=float(gradient.coords[1, 1]), finite_difference=finite_difference,
        receipts=receipts))


def test_density_selection_conventional_triples_and_restart(molecular_case, monkeypatch, tmp_path):
    """The D prepass and semicanonical solver use the same domain-only layout."""
    from pyscfad.dlno_stc._restart import RestartManager
    case = molecular_case
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'half')
    static = ccsd.build_static_selections(case.mf, case.domains, matrix_kind='D',
        thresh_occ=1e-4, thresh_vir=3e-5, baseline_controls=case.controls,
        matrix_controls=case.controls, lis_space='domain', smooth=False,
        fragment_indices=case.indices, single_pass=False)
    manager = RestartManager(tmp_path, resume=False, method='domain-lis-test',
                             scientific_payload={'lis_space': 'domain'})
    try:
        manager.save_static(static)
    finally:
        manager.close()
    manager = RestartManager(tmp_path, resume=True, method='domain-lis-test',
                             scientific_payload={'lis_space': 'domain'})
    try:
        restored = manager.load_static(expected_type=type(static))
    finally:
        manager.close()
    assert restored.lis_static.lis_space == 'domain'
    assert all(restored.fragments[i].lis_space == 'domain' for i in case.indices)
    state = ccsd.driver._initialize(case.mf, case.domains, case.controls, with_grad=False)
    records = ccsd._run_group(state, restored, case.group, case.controls, case.controls,
        with_grad=False, verbose_imp=0, ccsd_t=True, include_baseline=False,
        fragment_indices=case.indices,
        triples_controls=ccsd.normalize_triples_controls(None, ccsd_t=True))
    assert len(records) == len(case.indices)
    assert all(np.isfinite([row.e_ccsd, row.e_mp2_lis, row.e_triples]).all() for row in records)
