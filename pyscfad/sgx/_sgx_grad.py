# Copyright 2018-2019 The PySCF Developers. All Rights Reserved.
# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0.
#
# Adapted from PySCF pyscf/sgx/grad/rhf.py, get_k_grad_only
# (author Kyle Bystrom), and pyscf/sgx/sgx_jk.py.
"""Bilinear SGX response with shared grids and a cached overlap factorization."""
import numpy as np
import scipy.linalg
import time
from math import lcm
from pathlib import Path
from pyscf import lib
from pyscf.dft import numint as ni
from pyscf.sgx import sgx as native_sgx, sgx_jk
from pyscf.sgx.grad.rhf import scalable_grids_response_cc
from pyscfad.lib._threading import fixed_threads


def _system_available_bytes():
    """Reclaimable host headroom, when the OS exposes it (Linux)."""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _aligned_grid_block(points, ngrids, alignment):
    if points >= ngrids:
        return ngrids
    block = int(points)//alignment*alignment
    if block < alignment:
        raise MemoryError('SGX workspace cannot hold one aligned native grid block')
    return block


def _grid_plan(nao, natm, ngrids, nset, grad, threads, workspace, headroom,
               native_scratch=0, native_batch=None):
    """Target about four actual native grid jobs per thread within live memory.

    Generated kernels use 128-point jobs; libcint uses PySCF's 224-point jobs.
    Panel boundaries preserve both native tiles and PySCF screening/AO masks,
    including when a derivative falls back to libcint with an older library.
    """
    if native_batch is None:
        native_batch = sgx_jk.SGX_BLKSIZE
    alignment = lcm(native_batch, sgx_jk.SGX_BLKSIZE, sgx_jk.BLKSIZE)
    arrays = (10 + 7 * nset) if grad else (4 + 3 * nset)
    per_grid = 8 * (arrays * nao + 3 * natm)
    # Arrays allocated after loop planning, plus the native per-thread K
    # reduction buffers. Cached metric/overlap arrays are already in RSS.
    fixed = 8 * (((5*nset + 4) if grad else (2*nset + 3))*nao**2
                 + (4*ngrids if grad else 0))
    thread_scratch = 8 * threads * nset * (3 if grad else 1) * nao * native_batch
    thread_scratch = max(thread_scratch, native_scratch)
    available = max(0, min(int(workspace), int(.8 * headroom))-fixed-thread_scratch)
    jobs_target = max(4*threads, (256*1024**2)//(per_grid*native_batch))
    points = min(available//per_grid, max(alignment, jobs_target*native_batch))
    block = _aligned_grid_block(points, ngrids, alignment)
    return dict(block_points=int(block), native_batch_points=native_batch,
                block_alignment_points=alignment,
                threads=int(threads), target_grid_jobs=int(jobs_target),
                selected_grid_jobs=int((block+native_batch-1)//native_batch),
                bytes_per_grid=int(per_grid), grid_bytes=int(block*per_grid),
                fixed_reserve_bytes=int(fixed), thread_scratch_bytes=int(thread_scratch),
                workspace_bytes=int(workspace), live_headroom_bytes=int(headroom),
                usable_grid_bytes=int(available))


class _SGXData(sgx_jk.SGXData):
    """PySCF's sparse/native kernels with bounded grid panels and reusable LU."""
    def get_loop_data(self, nset=1, with_pair_mask=True, grad=False):
        # PySCF subtracts current RSS; refresh after J and cache allocations.
        self.max_memory = lib.current_memory()[0] + self._response_workspace_bytes / 1e6
        nbins, mask, pairs, loc, native_block = super().get_loop_data(nset, with_pair_mask, grad)
        from pyscfad.lib import _scf_response_cache
        from pyscfad.lib import _solver_memory
        total = _scf_response_cache.total_memory_mb()
        if total is None:
            total = self._response_total_memory_mb
        headroom = max(0, (total-lib.current_memory()[0])*1e6)
        host_available = _system_available_bytes()
        if host_available is not None:
            headroom = min(headroom, host_available)
        solver_bytes = _solver_memory.reserved_bytes()
        headroom = max(0, headroom-solver_bytes)
        generated = getattr(self, '_generated_plan', None)
        native_batch = (generated.grid_batch_points if generated is not None
                        else sgx_jk.SGX_BLKSIZE)
        plan = _grid_plan(self.mol.nao, self.mol.natm, len(self.grids.weights),
                          nset, grad, lib.num_threads(), self._response_workspace_bytes,
                          headroom, native_scratch=(
                              generated.scratch_bytes(nset, derivative=grad)*lib.num_threads()
                              if generated is not None else 0), native_batch=native_batch)
        block = _aligned_grid_block(min(plan['block_points'], native_block),
                                    len(self.grids.weights), plan['block_alignment_points'])
        plan.update(block_points=block, grid_bytes=block*plan['bytes_per_grid'],
                    solver_reserve_bytes=solver_bytes,
                    selected_grid_jobs=(block+native_batch-1)//native_batch)
        self._response_grid_plan = plan
        return nbins, mask, pairs, loc, block

    def _build_ovlp_fit(self):
        self._response_overlap_lu = None
        if not self.fit_ovlp:
            self._overlap_correction_matrix = np.identity(self.mol.nao)
            return
        if not self.sym_ovlp:
            raise NotImplementedError('SGX response requires symmetric overlap fitting')
        sn = np.zeros((self.mol.nao, self.mol.nao))
        nbins, mask, pairs, loc, block = self.get_loop_data()
        grids, mol = self.grids, self.mol
        for p, q in lib.prange(0, len(grids.weights), block):
            ao = mol.eval_gto('GTOval', grids.coords[p:q], non0tab=mask[p // sgx_jk.BLKSIZE:])
            sgx_jk._dot_ao_ao_sparse(ao, ao, grids.weights[p:q], nbins,
                                    mask[p // sgx_jk.BLKSIZE:], pairs, loc,
                                    hermi=self.hermi, out=sn)
        self._response_overlap_lu = scipy.linalg.lu_factor(sn, overwrite_a=True, check_finite=False)
        proj = scipy.linalg.lu_solve(self._response_overlap_lu,
                                     mol.intor_symmetric('int1e_ovlp'), check_finite=False)
        proj *= .5
        proj.flat[::mol.nao + 1] += .5
        self._overlap_correction_matrix = proj


class SGX(native_sgx.SGX):
    @fixed_threads()
    def build(self, *args, **kwargs):
        return super().build(*args, **kwargs)

    @fixed_threads()
    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        if omega is not None or not self.optk or not self.dfj:
            return super().get_jk(dm, hermi, vhfopt, with_j, with_k,
                                  direct_scf_tol, omega)
        from . import _dfj
        vj = _dfj.get_j(self, dm, hermi, direct_scf_tol) if with_j else None
        vk = get_k(self, dm, hermi, direct_scf_tol) if with_k else None
        return vj, vk

    @fixed_threads()
    def _build_pjs(self, direct_scf_tol):
        data = _SGXData(self.mol, self.grids, fit_ovlp=self.fit_ovlp,
                        sym_ovlp=self._symm_ovlp_fit, max_memory=self.max_memory,
                        direct_scf_tol=direct_scf_tol, vtol=self.sgx_tol_potential,
                        etol=self.sgx_tol_energy, hermi=1, bound_algo=self.bound_algo,
                        sgxopt=self._opt)
        data._response_workspace_bytes = self._response_workspace_bytes
        data._response_total_memory_mb = self._response_total_memory_mb
        data._integral_backend = getattr(self, '_integral_backend', 'auto')
        data._sgx_accuracy = getattr(self, '_sgx_accuracy', None)
        data.build()
        self._pjs_data = data


def _back_contract(ao, gv, weights, mask, loc, out):
    # A non-None mask selects the modern blocked native contraction. Keep
    # every density/output pair so K remains a fixed linear response map.
    output_mask = np.ones(((len(ao)+sgx_jk.BLKSIZE-1)//sgx_jk.BLKSIZE,
                           len(loc)-1), dtype=np.uint8)
    return sgx_jk._sgxdot_ao_gv_sparse(ao, gv, weights, mask, output_mask, loc, out=out)


# These are contraction subpanels only: integral tiles and their screening
# decisions are unchanged. Align boundaries with the existing AO mask.
_CONTRACTION_BLOCK = 16 * sgx_jk.BLKSIZE


def _active_ao(mask, loc, start, stop):
    if mask is None:
        return np.arange(loc[-1], dtype=np.int32)
    shells = np.any(mask[start//sgx_jk.BLKSIZE:
                         (stop+sgx_jk.BLKSIZE-1)//sgx_jk.BLKSIZE] != 0, axis=0)
    return np.flatnonzero(np.repeat(shells, np.diff(loc))).astype(np.int32)


def _front_contract(ao, dms, mask, loc, out=None):
    """AO-density product using the existing geometry mask and compact BLAS.

    Only AO rows of D are packed; every output column is retained, including
    for signed response densities. No density cutoff is introduced.
    """
    ngrids, nao = ao.shape
    result = np.ndarray((len(dms), nao, ngrids), dtype=ao.dtype, buffer=out)
    for p, q in lib.prange(0, ngrids, _CONTRACTION_BLOCK):
        active = _active_ao(mask, loc, p, q)
        if not len(active):
            result[:, :, p:q] = 0.
            continue
        packed_ao = np.ascontiguousarray(ao[p:q, active].T)
        for i, dm in enumerate(dms):
            packed_dm = np.ascontiguousarray(dm[active, :].T)
            result[i, :, p:q] = lib.dot(packed_dm, packed_ao)
    return result


def _geometry_contract(dao, fg, gv, dgv, projected, weights, mask, loc,
                       atom_ao, xed, dxed):
    """Form only needed AO-response columns and fuse their force reductions."""
    from . import _native
    ngrids, nao = dao.shape[1:]
    # GTOval_deriv1 is already contiguous in (component, AO, grid) order.
    dao_cart = np.ascontiguousarray(dao.transpose(0, 2, 1))
    grid_energy = np.zeros(ngrids)
    grid_motion = np.zeros((3, ngrids))
    for p, q in lib.prange(0, ngrids, _CONTRACTION_BLOCK):
        active = _active_ao(mask, loc, p, q)
        index = np.full(nao, -1, dtype=np.int32)
        index[active] = np.arange(len(active), dtype=np.int32)
        for i in range(len(fg)):
            j = i ^ 1
            tmp = (lib.dot(np.ascontiguousarray(projected[j, active, :]),
                           np.ascontiguousarray(gv[i, :, p:q])) if len(active)
                   else np.empty((0, q-p)))
            if _native.reduce_geometry(dao_cart, fg[j], gv[i], dgv[i],
                    weights, p, q, index, tmp, atom_ao, grid_energy, grid_motion):
                continue
            # Compatibility path for older native libraries or libcint-only
            # installations. It uses the same compact intermediate.
            tmp *= .5 * weights[p:q]
            wfg = fg[j, :, p:q] * (.5 * weights[p:q])
            atom_ao[:, active] -= np.einsum('xug,ug->xu', dao_cart[:, active, p:q], tmp)
            atom_ao -= np.einsum('xug,ug->xu', dgv[i, :, :, p:q], wfg)
            grid_energy[p:q] += np.einsum('ug,ug->g', fg[j, :, p:q], gv[i, :, p:q])
            grid_motion[:, p:q] -= np.einsum('ug,xug->xg', wfg, dgv[i, :, :, p:q])
            grid_motion[:, p:q] -= np.einsum('ug,xug->xg', tmp, dao_cart[:, active, p:q])
    xed += grid_energy
    dxed += grid_motion


def get_k(obj, dm, hermi, tol):
    """SGX contractions with a fixed reference or generated adaptive kernels."""
    mol, grids = obj.mol, obj.grids
    data = obj._pjs_data
    if data is None or data.mol is not mol or data.grids is not grids or data._itol != tol:
        obj._build_pjs(tol)
        data = obj._pjs_data
    if data.use_dm_screening:
        return sgx_jk.get_k_only(obj, dm, hermi, tol)
    from . import _native
    native_plan = _native.get_plan(data)
    dms = np.asarray(dm).reshape(-1, mol.nao, mol.nao)
    _, screen, _, loc, block = data.get_loop_data(nset=len(dms))
    proj = data._overlap_correction_matrix
    projected = lib.einsum('ki,xij->xkj', proj, dms)
    vk = np.zeros_like(dms)
    batch = (native_plan.contract if native_plan is not None else
             sgx_jk._gen_k_direct(mol, 's2', tol, data))
    ao = fg = None
    for p, q in lib.prange(0, len(grids.weights), block):
        coords, weights = grids.coords[p:q], grids.weights[p:q]
        mask = screen[p//sgx_jk.BLKSIZE:]
        ao = mol.eval_gto('GTOval', coords, non0tab=mask, out=ao)
        fg = _front_contract(ao, projected, mask, loc, out=fg)
        extra = ({'ao_l1': np.sum(np.abs(ao), axis=1)}
                 if native_plan is not None and native_plan.accuracy is not None else {})
        gv = batch(mol, coords, fg, weights, p//sgx_jk.SGX_BLKSIZE, **extra)
        _back_contract(ao, gv, weights, mask, loc, vk)
        del gv
    if data.sym_ovlp:
        vk = lib.einsum('ik,xij->xkj', proj, vk)
    if hermi == 1:
        vk = .5*(vk+vk.transpose(0, 2, 1))
    return vk.reshape(np.shape(dm))


def density_response(obj, bar, tol):
    """Apply response K with the requested precision but no density skipping.

    Density-dependent skipping is not homogeneous at tiny Krylov solution
    magnitudes. Mixed response retains the forward FP32 classification
    thresholds while disabling skipping; otherwise response uses static FP64.
    Mixed arithmetic remains approximately linear. An FP64 reference uses a
    separate backend configured with mixed_precision=False.
    """
    from . import _native
    data = obj._pjs_data
    if data is None or data.mol is not obj.mol or data.grids is not obj.grids or data._itol != tol:
        obj._build_pjs(tol)
        data = obj._pjs_data
    plan = _native.get_plan(data)
    accuracy = None if plan is None else plan.accuracy
    try:
        if plan is not None:
            plan.accuracy = (dict(accuracy, screening=False)
                if accuracy is not None and accuracy['mixed'] else None)
        return get_k(obj, bar, 1, tol)
    finally:
        if plan is not None:
            plan.accuracy = accuracy


def geometry(obj, dm, bar, tol):
    """d <bar,K(dm)> / dR in one shared traversal of the integral grid.

    Every quadratic product in PySCF's exchange-energy derivative is replaced
    by its two cross products. No plus/minus densities or subtraction of large
    quadratic gradients are needed. Both densities share AO evaluation,
    integral evaluation, overlap response and partition/grid-motion work.
    """
    return geometry_many(obj, [(dm, bar)], tol)


@fixed_threads()
def geometry_many(obj, contributions, tol):
    """Sum bilinear geometry seeds in one value/derivative integral traversal.

    Distinct densities retain their own contractions, but share integral,
    AO, overlap-fit and grid-motion work. The live grid planner includes
    the actual number of density/seed matrices in its workspace estimate.
    """
    mol, grids, data = obj.mol, obj.grids, obj._pjs_data
    if data.use_dm_screening or not obj._grid_response or obj._full_dm is not None:
        raise NotImplementedError('SGX response requires full grid response and fixed density-independent screening')
    contributions = [(d, b) for d, b in contributions if np.any(d) and np.any(b)]
    if not contributions:
        return np.zeros((mol.natm, 3))
    dms = np.stack([x for pair in contributions for x in pair])
    from . import _native
    native_plan = _native.get_plan(data)
    nbins, screen, pairs, loc, block = data.get_loop_data(nset=len(dms), grad=True)
    started = time.perf_counter()
    counts_before = native_plan.screening_stats() if native_plan is not None else {}
    timings = dict(ao_seconds=0., front_contraction_seconds=0.,
                   integral_contraction_seconds=0., back_contraction_seconds=0.,
                   geometry_contraction_seconds=0.)
    all_pairs = np.ones_like(pairs)
    all_pairs_active = bool(np.all(pairs))
    proj = data._overlap_correction_matrix
    projected = np.stack([lib.dot(proj, d) for d in dms])
    vk = np.zeros_like(dms) if obj.fit_ovlp else None
    atom_ao = np.zeros((3, mol.nao))
    result = np.zeros((mol.natm, 3))
    xed = np.zeros(len(grids.weights))
    dxed = np.zeros((3, len(grids.weights)))
    batch = (native_plan.contract if native_plan is not None else
             sgx_jk._gen_k_direct(mol, 's2', tol, data))
    batch_with_grad = (native_plan.contract_with_grad
                       if native_plan is not None and native_plan.has_derivatives else None)
    batch_grad = (None if batch_with_grad is not None else
                  sgx_jk._gen_k_direct(mol, 's1', tol, data, grad=True))
    eval_name = 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1'
    setup_seconds = time.perf_counter()-started
    for p, q in lib.prange(0, len(grids.weights), block):
        coords, weights = grids.coords[p:q], grids.weights[p:q]
        mask = screen[p // sgx_jk.BLKSIZE:(q + sgx_jk.BLKSIZE - 1) // sgx_jk.BLKSIZE]
        tick = time.perf_counter()
        ao4 = mol.eval_gto(eval_name, coords, non0tab=mask)
        ao, dao = ao4[0], ao4[1:4]
        timings['ao_seconds'] += time.perf_counter()-tick
        tick = time.perf_counter()
        fg = _front_contract(ao, projected, mask, loc)
        extra = ({'ao_l1': np.sum(np.abs(ao), axis=1)}
                 if native_plan is not None and native_plan.accuracy is not None else {})
        timings['front_contraction_seconds'] += time.perf_counter()-tick
        tick = time.perf_counter()
        if batch_with_grad is not None:
            gv, dgv = batch_with_grad(mol, coords, fg, weights, p // sgx_jk.SGX_BLKSIZE, **extra)
        else:
            gv = batch(mol, coords, fg, weights, p // sgx_jk.SGX_BLKSIZE, **extra)
            dgv = batch_grad(mol, coords, fg, weights, p // sgx_jk.SGX_BLKSIZE)
        timings['integral_contraction_seconds'] += time.perf_counter()-tick
        tick = time.perf_counter()
        if obj.fit_ovlp:
            _back_contract(ao, gv, weights, mask, loc, vk)
        timings['back_contraction_seconds'] += time.perf_counter()-tick
        tick = time.perf_counter()
        _geometry_contract(dao, fg, gv, dgv, projected, weights, mask, loc,
                           atom_ao, xed[p:q], dxed[:, p:q])
        timings['geometry_contraction_seconds'] += time.perf_counter()-tick
        del ao4, ao, dao, fg, gv, dgv
    tick = time.perf_counter()
    slices = mol.aoslice_by_atom()
    cmat = None
    if obj.fit_ovlp:
        cmat = np.zeros((mol.nao, mol.nao))
        for i in range(len(dms)):
            cmat += lib.dot(vk[i], dms[i ^ 1])
        cmat = scipy.linalg.lu_solve(data._response_overlap_lu, cmat,
                                     overwrite_b=True, check_finite=False)
        # Contract overlap derivatives by atom; never allocate 3 full AO matrices.
        for atom, (s0, s1, p, q) in enumerate(slices):
            if s0 == s1:
                continue
            ds = mol.intor('int1e_ipovlp', comp=3, shls_slice=(s0, s1, 0, mol.nbas))
            result[atom] -= lib.einsum('xuv,vu->x', ds, cmat[:, p:q])
            result[atom] -= lib.einsum('xuv,uv->x', ds, cmat[p:q])
        # Original nonsymmetrized overlap map is 2*proj-I.
        cmat = 2 * lib.dot(cmat, proj.T) - cmat
        cmat += cmat.T.copy()
    del vk, projected, dms
    timings['overlap_response_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter()
    for p, q, atoms, coords, weights, weights1, mask in scalable_grids_response_cc(grids, block):
        if cmat is not None:
            ao4 = mol.eval_gto(eval_name, coords, non0tab=mask)
            ao, dao = ao4[0], ao4[1:4]
            ca = ni._dot_ao_dm_sparse(ao, cmat, nbins, mask, all_pairs, loc)
            energy = xed[p:q] - .5 * ni._contract_rho_sparse(ca, ao, mask, loc)
            motion = np.stack([ni._contract_rho_sparse(ca, d, mask, loc) for d in dao])
            motion *= weights
            motion += 4 * dxed[:, p:q]
            # Equivalent to forming 3 AO-by-AO overlap derivative matrices
            # and contracting their atom rows, but only a grid-by-AO tile lives.
            ca_masked = ca if all_pairs_active else ni._dot_ao_dm_sparse(
                ao, cmat, nbins, mask, pairs, loc)
            for atom, (_, _, a, b) in enumerate(slices):
                result[atom] += lib.einsum('xgu,gu,g->x', dao[:, :, a:b], ca_masked[:, a:b], weights)
            del ao4, ao, dao, ca, ca_masked
        else:
            energy, motion = xed[p:q], 4 * dxed[:, p:q]
        result += lib.einsum('axg,g->ax', weights1, energy)
        for atom in range(mol.natm):
            result[atom] -= motion[:, atoms == atom].sum(axis=1)
    for atom, (_, _, p, q) in enumerate(slices):
        result[atom] += 4 * atom_ao[:, p:q].sum(axis=1)
    timings['grid_response_seconds'] = time.perf_counter()-tick
    counts_after = native_plan.screening_stats() if native_plan is not None else {}
    data._geometry_stats = dict(seed_pairs=len(contributions),
        grid_points=len(grids.weights), block_points=block,
        accuracy=(dict(native_plan.accuracy)
                  if native_plan is not None and native_plan.accuracy is not None else None),
        grid_plan=dict(data._response_grid_plan),
        screening_counts={k: v-counts_before[k] for k, v in counts_after.items()},
        screening_counts_available=(native_plan is not None and native_plan.accuracy is not None),
        setup_seconds=setup_seconds, **timings,
        total_seconds=time.perf_counter()-started)
    return .5 * result
