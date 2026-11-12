"""MP2 retains the correlation metric through its immediate pullback."""
import jax.scipy.linalg
import numpy as np
import pytest
import scipy.linalg
from pyscf import df as pyscf_df, gto as pyscf_gto

from pyscfad import dlno_stc, gto
from pyscfad.dlno_stc.test.test_correlation_auxbasis import (
    RIFIT, _controls, _independent_mp2_total, _mf, molecule,
)
from pyscfad.lno import _df_shared


@pytest.mark.parametrize('scope', ['system', 'domain'])
@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('cache_mode', [None, 'direct', 'half'])
def test_one_metric_and_factorization_per_mp2_group(molecule, monkeypatch,
                                                  scope, with_grad, cache_mode):
    if cache_mode is None:
        monkeypatch.delenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', raising=False)
    else:
        monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', cache_mode)
    mf = _mf(molecule)
    static = (dlno_stc.build_domain_selections(
        mf, frozen=1, force_full_domains=True, boys_grouping='molecule')
        if scope == 'domain' else None)
    expected = _independent_mp2_total(molecule)
    aux = pyscf_df.addons.make_auxmol(molecule.to_pyscf(), RIFIT)
    counts = dict(metric=0, cholesky=0)
    receipts = []

    def counted_intor(original):
        def intor(mol, name, *args, **kwargs):
            if name in ('int2c2e', 'int2c2e_sph') and mol._basis == aux._basis:
                counts['metric'] += 1
            return original(mol, name, *args, **kwargs)
        return intor

    for cls in (pyscf_gto.Mole, gto.Mole):
        monkeypatch.setattr(cls, 'intor', counted_intor(cls.intor))

    def counted_cholesky(original):
        def cholesky(a, *args, **kwargs):
            if a.shape == (aux.nao, aux.nao):
                counts['cholesky'] += 1
            return original(a, *args, **kwargs)
        return cholesky

    for module in (np.linalg, scipy.linalg, jax.scipy.linalg):
        monkeypatch.setattr(module, 'cholesky', counted_cholesky(module.cholesky))
    finish = _df_shared.LocalDFReverse.finish

    def finished(shared):
        coords = finish(shared)
        receipts.append(dict(shared.receipt))
        return coords

    monkeypatch.setattr(_df_shared.LocalDFReverse, 'finish', finished)
    options = dict(scope=scope, frozen=1 if scope == 'system' else None, controls=_controls(),
                   correlation_auxbasis=RIFIT)
    if with_grad:
        energy, gradient = dlno_stc.value_and_grad(
            molecule, _mf, static, include_hf=True, **options)
        np.testing.assert_allclose(np.sum(gradient.coords, axis=0), 0,
                                   atol=2e-8, rtol=0)
    else:
        energy = mf.e_tot + dlno_stc.kernel(mf, static, **options)
    np.testing.assert_allclose(energy, expected, atol=3e-10, rtol=0)
    assert counts == dict(metric=1, cholesky=1)
    assert len(receipts) == 1
    assert receipts[0]['geometry_passes'] == int(with_grad)
    assert receipts[0]['metric_passes'] == int(with_grad)
    half = scope == 'domain' and cache_mode == 'half'
    assert receipts[0]['half_builds'] == int(half)
    assert receipts[0]['raw_source_passes'] == 1 + int(with_grad and not half)
    assert receipts[0]['half_pullbacks'] == int(with_grad and half)
    if with_grad:
        assert receipts[0]['half_resident_bytes'] == 0
    assert _df_shared._ACTIVE.get() is None
