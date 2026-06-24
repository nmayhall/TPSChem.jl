import numpy as np
import pyscf
import copy as cp
from pyscf import gto, ao2mo, fci


mol = gto.Mole()
R = 0.75
atom = [
        ["h",   (0.0,       0.0,        0.0)],   
        ["h",   (1*R,       0.0,        0.0)],   
        ["h",   (2*R,       0.0,        0.0)],   
        ["h",   (3*R,       0.0,        0.0)],   
        ["h",   (4*R,       0.0,        0.0)],   
        ["h",   (5*R,       0.0,        0.0)],   
        ]

basis = "sto-3g"
mol.atom = atom
mol.build()
mf = pyscf.scf.RHF(mol).run(conv_tol=1e-8)
h0 = mf.energy_nuc()
print(h0)
print(mf.mo_energy)
C = mf.mo_coeff    
norb = C.shape[1]
h1 = C.T @ mf.get_hcore(mol) @ C
h2 = pyscf.ao2mo.kernel(mol, C, aosym="s4",compact=False)
h2.shape = (norb, norb, norb, norb)
    
cisolver = pyscf.fci.FCI(mf)
print('E(FCI) = %.12f' % cisolver.kernel()[0])


np.save('h0.npy', h0)
np.save('h1.npy', h1)
np.save('h2.npy', h2)
