"""
PySCF-backed functionality for TPSChem, activated when PyCall is loaded.

These methods attach to the stubs declared in `TPSChem.ClusterMeanField`
(`pyscf_do_scf`, `pyscf_build_ints`, `pyscf_fci`, ...). Requires a Python
environment with `pyscf` installed.
"""
module TPSChemPyCallExt

using PyCall
using LinearAlgebra
using Printf

using TPSChem
using TPSChem.QCBase
using TPSChem.InCoreIntegrals
using TPSChem.RDM

import TPSChem.ClusterMeanField: pyscf_do_scf,
                                 make_pyscf_mole,
                                 pyscf_write_molden,
                                 pyscf_build_1e,
                                 pyscf_build_eri,
                                 pyscf_get_jk,
                                 pyscf_build_ints,
                                 pyscf_fci,
                                 pyscf_fci_rdm12s,
                                 get_nuclear_rep,
                                 localize,
                                 get_ovlp

include("PyscfFunctions.jl")

function pyscf_fci_rdm12s(ints_i::InCoreInts, na, nb, tol_ci, maxiter_ci; use_nosym=false)
    pyscf = pyimport("pyscf")
    pyimport("pyscf.fci")
    no = n_orb(ints_i)
    if use_nosym
        cisolver = pyscf.fci.direct_nosym.FCI()
    else
        cisolver = pyscf.fci.direct_spin1.FCI()
    end
    cisolver.max_cycle = maxiter_ci
    cisolver.conv_tol = tol_ci
    cisolver.conv_tol_residual = tol_ci
    e, vfci = cisolver.kernel(ints_i.h1, ints_i.h2, no, (na, nb), ecore=ints_i.h0)
    (d1a, d1b), (d2aa, d2ab, d2bb) = cisolver.make_rdm12s(vfci, no, (na, nb))
    return e, RDM1(d1a, d1b), RDM2(d2aa, d2ab, d2bb)
end

end
