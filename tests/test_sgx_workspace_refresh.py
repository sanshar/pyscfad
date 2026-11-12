from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import lib
from pyscfad.sgx import _sgx_grad


def test_workspace_refresh_after_coulomb_allocations(monkeypatch):
    # Peptide dimensions, with metadata only: no integrals or large AO arrays.
    nao, natm, ngrids = 3408, 154, 100000
    mol = SimpleNamespace(nao=nao, natm=natm, nbas=1,
        nao_nr=lambda: nao, ao_loc_nr=lambda: np.array([0, nao], dtype=np.int32))
    data = _sgx_grad._SGXData.__new__(_sgx_grad._SGXData)
    data._mol = mol
    data._grids = SimpleNamespace(mol=mol, weights=range(ngrids),
        coords=SimpleNamespace(shape=(ngrids, 3)), non0tab=np.ones((1, 1)),
        atm_idx=np.zeros(1), cutoff=1e-15)
    data.max_memory = 1500.  # Limit recorded before J/cache allocations.
    data._response_workspace_bytes = 1_000_000_000
    data._response_total_memory_mb = 16000.
    data._generated_plan = SimpleNamespace(grid_batch_points=128,
        scratch_bytes=lambda *a, **kw: 0)
    monkeypatch.setattr(lib, 'current_memory', lambda: (2000., 0.))
    monkeypatch.setattr(lib, 'num_threads', lambda: 12)
    monkeypatch.setattr(_sgx_grad, '_system_available_bytes', lambda: None)

    block = data.get_loop_data(with_pair_mask=False)[-1]
    plan = data._response_grid_plan
    assert block >= 896 and block % 896 == 0
    reserved = plan['grid_bytes'] + plan['fixed_reserve_bytes'] + plan['thread_scratch_bytes']
    assert reserved <= data._response_workspace_bytes
    assert reserved <= .8 * plan['live_headroom_bytes']

    # A refreshed upstream cap must still respect genuine process exhaustion.
    monkeypatch.setattr(lib, 'current_memory', lambda: (15999., 0.))
    with pytest.raises(MemoryError, match='grid block'):
        data.get_loop_data(with_pair_mask=False)

    monkeypatch.setattr(lib, 'current_memory', lambda: (2000., 0.))
    data._response_workspace_bytes = 100_000_000
    with pytest.raises(MemoryError, match='grid block'):
        data.get_loop_data(with_pair_mask=False)
