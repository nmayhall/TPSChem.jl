import numpy as np
import pyscf
import copy as cp
from pyscf import gto, ao2mo


mol = gto.Mole()
R = 0.75
atom = [
        ["h",   (-R,       0.0,        0.0)],   
        ["h",   ( R,       0.0,        0.0)],   
        ["h",   ( 0.0,      -R,        0.0)],   
        ["h",   ( 0.0,       R,        0.0)], 
        ["h",   ( 0.0,        0.0,      -R)],   
        ["h",   ( 0.0,        0.0,       R)],  
        ]

basis = "sto-3g"
mol.atom = atom
mol.build()
mf = pyscf.scf.RHF(mol).run(conv_tol=1e-8)
h0 = mf.energy_nuc()
print(mf.mo_energy)
C = mf.mo_coeff    
norb = C.shape[1]
h1 = C.T @ mf.get_hcore(mol) @ C
h2 = pyscf.ao2mo.kernel(mol, C, aosym="s4",compact=False)
h2.shape = (norb, norb, norb, norb)

np.save('h0.npy', h0)
np.save('h1.npy', h1)
np.save('h2.npy', h2)
