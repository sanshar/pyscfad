from pyscfad import config, gto, scf, sgx, dlno_stc
from pyscf.data.elements import chemcore

config.update('pyscfad_moleintor_opt', True)

mol = gto.Mole(
    #atom = 'water_8.xyz',
    #atom = 'water_16.xyz',
    #atom='C50H76N14O13S1.xyz',
    atom = "C30H50CIN7O4S.xyz",
    basis='cc-pvtz',
    max_memory=20000,
    verbose=4);

mol.build(trace_exp=False, trace_ctr_coeff=False)
ncore = chemcore(mol)

def build_mf(mol):
    return sgx.sgx_fit(scf.RHF(mol).density_fit(auxbasis='cc-pvtz-ri'),
                      auxbasis='def2-universal-jfit', max_memory=1000,
                      integral_backend='generated').run()

energy, gradient = dlno_stc.value_and_grad(
    mol, build_mf, include_hf=True, controls={'mode': 'deterministic'},
    build_static=lambda mf: dlno_stc.build_domain_selections(mf, frozen=ncore, boys_grouping='atom'))
print('HF+MP2 total energy (Ha):', float(energy))
print('Nuclear gradient (Ha/Bohr):\n', gradient.coords)
