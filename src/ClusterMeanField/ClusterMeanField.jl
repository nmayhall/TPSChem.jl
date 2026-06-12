module ClusterMeanField

using ..QCBase
using ..RDM

using LinearAlgebra
using Random
using Optim
using TensorOperations
using ..InCoreIntegrals
using Printf
using ..ActiveSpaceSolvers

export cmf_ci
export cmf_oo

export pyscf_do_scf
export make_pyscf_mole
export pyscf_write_molden
export pyscf_build_1e
export pyscf_build_eri
export pyscf_get_jk
export pyscf_build_ints
export pyscf_fci
export get_nuclear_rep
export localize
export get_ovlp

#Base.convert(::Type{Vector{MOCluster}}, in::Vector{MOCluster{N}}) where {N} = return Vector{MOCluster}(in)

# PySCF-backed functions live in ext/TPSChemPyCallExt.jl and are only available
# when PyCall is loaded in the user's environment. Stubs declared here so the
# extension can attach methods.
function pyscf_do_scf end
function make_pyscf_mole end
function pyscf_write_molden end
function pyscf_build_1e end
function pyscf_build_eri end
function pyscf_get_jk end
function pyscf_build_ints end
function pyscf_fci end
function pyscf_fci_rdm12s end
function get_nuclear_rep end
function localize end
function get_ovlp end

include("incore_cmf.jl")
include("direct_cmf.jl")

end
