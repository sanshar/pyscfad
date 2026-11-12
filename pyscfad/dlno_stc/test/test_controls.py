"""Fixed quadrature defaults and high-level stochastic error budgets."""
import numpy as np
import pytest
from types import SimpleNamespace
from pyscfad.dlno_stc import driver


def test_original_grid_returns_independent_arrays():
    from pyscfad.dlno_stc.controls import default_laplace_grid, DEFAULT_ENERGY_TOLERANCE
    roots, weights = default_laplace_grid()
    np.testing.assert_array_equal(roots, np.array([.003431, .023534, .088984, .275603, .757121, 1.906218]) * 2.6)
    np.testing.assert_array_equal(weights, np.array([.009348, .035196, .107559, .293035, .729094, 1.690608]) * 2.6)
    roots[:] = weights[:] = 0
    fresh_roots, fresh_weights = default_laplace_grid()
    assert np.all(fresh_roots > 0) and np.all(fresh_weights > 0)
    assert DEFAULT_ENERGY_TOLERANCE == 3e-4


@pytest.mark.parametrize('supplied_grid', [True, False])
@pytest.mark.parametrize('scope,tolerance,production,expected', [
    ('domain', 6e-4, None, 6e-4 / np.sqrt(2)),
    ('domain', None, None, 3e-4 / np.sqrt(2)),
    ('system', 6e-4, None, 6e-4), ('system', None, None, 3e-4),
    ('domain', None, 7, None), ('system', None, 7, None),
])
def test_high_level_budget_reaches_solver_without_mutating_caller(monkeypatch, scope, tolerance, production, expected, supplied_grid):
    controls = {'mode': 'stochastic', 'laplace_roots': np.array([1.]), 'laplace_weights': np.array([2.])}
    if not supplied_grid:
        del controls['laplace_roots'], controls['laplace_weights']
    if tolerance is not None:
        controls['energy_tolerance'] = tolerance
    if production is not None:
        controls['production_samples'] = production
    before = controls.copy()
    state = dict(weak_pairs=[(0, 1)],
                 occupied=[0], virtual=[1], energy=np.float64(0), variance=0.,
                 mf_bar=None, common_bar=None, mf=SimpleNamespace(verbose=0), common=None)
    monkeypatch.setattr(driver, '_initialize', lambda *a, **k: state)
    monkeypatch.setattr(driver, '_initialize_system', lambda *a, **k: state)
    monkeypatch.setattr(driver, '_prepare', lambda *a, **k: (None, {}, None))
    monkeypatch.setattr(driver, '_prepare_system', lambda *a, **k: (None, {}, None))
    monkeypatch.setattr(driver, 'weak_pair_energy_sum', lambda *a, **k: np.float64(-.25))
    from pyscfad.dlno_stc import DomainGroup
    monkeypatch.setattr(driver, 'build_domain_groups', lambda *a, **k:
        (DomainGroup((0,), 0, (0,)), DomainGroup((1,), 1, (0,))))
    monkeypatch.setattr(driver, '_finish', lambda s, **k: s['energy'])
    received = []
    def solver(inputs, metadata, solve_controls, backend, **kwargs):
        received.append(solve_controls.copy())
        return {'energy': np.float64(-1), 'energy_standard_error': 0.}
    monkeypatch.setattr(driver, 'run_backend', solver)
    static = SimpleNamespace(fragments=(SimpleNamespace(extended_atoms=()),)*2,
                             frag_lolist=((0,), (1,)))
    value = driver.kernel(None, static if scope == 'domain' else None, scope=scope, controls=controls)
    assert value == (-2.25 if scope == 'domain' else -1.)
    assert len(received) == (2 if scope == 'domain' else 1)
    for passed in received:
        assert passed.get('energy_tolerance') == expected
        assert passed.get('production_samples') == production
        if supplied_grid:
            np.testing.assert_array_equal(passed['laplace_roots'], [1.])
            np.testing.assert_array_equal(passed['laplace_weights'], [2.])
        else:
            np.testing.assert_array_equal(passed['laplace_roots'],
                np.array([.003431, .023534, .088984, .275603, .757121, 1.906218]) * 2.6)
            np.testing.assert_array_equal(passed['laplace_weights'],
                np.array([.009348, .035196, .107559, .293035, .729094, 1.690608]) * 2.6)
        if production is None:
            assert 'production_samples' not in passed
    assert controls.keys() == before.keys()
    for name in before:
        np.testing.assert_array_equal(controls[name], before[name])


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_invalid_budget_rejected_before_system_preparation(bad):
    with pytest.raises(ValueError, match='energy_tolerance'):
        driver.kernel(None, scope='system', controls={
            'mode': 'stochastic', 'energy_tolerance': bad,
            'laplace_roots': [1.], 'laplace_weights': [1.]})


@pytest.mark.parametrize('supplied', ['laplace_roots', 'laplace_weights'])
def test_partial_grid_rejected_before_reference_use(supplied):
    missing = 'laplace_weights' if supplied == 'laplace_roots' else 'laplace_roots'
    with pytest.raises(ValueError, match=missing):
        driver.kernel(None, scope='system', controls={'mode': 'stochastic', supplied: [1.]})
