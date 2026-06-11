"""
TPSChem: electronic structure in a tensor product state (TPS) basis.

Consolidates the former FermiCG ecosystem (FermiCG, QCBase, InCoreIntegrals,
BlockDavidson, RDM, ActiveSpaceSolvers, ClusterMeanField) into a single package.
Each former package lives on as a submodule of the same name.
"""
module TPSChem

#####################################
# External packages
#
using Compat
using KrylovKit
using LinearAlgebra
using Printf
using TimerOutputs
using OrderedCollections
using IterTools
using StaticArrays
using TensorOperations

using ThreadPools
using Distributed
using JLD2
using LinearMaps
using Random

#####################################
# Vendored submodules (former standalone packages), in dependency order
#
include("QCBase/QCBase.jl")
using .QCBase
include("InCoreIntegrals/InCoreIntegrals.jl")
using .InCoreIntegrals
include("BlockDavidson/BlockDavidson.jl")
using .BlockDavidson
include("RDM/RDM.jl")
using .RDM
include("ActiveSpaceSolvers/ActiveSpaceSolvers.jl")
using .ActiveSpaceSolvers
include("ClusterMeanField/ClusterMeanField.jl")

#####################################
# Core (former FermiCG)
#
include("core/Utils.jl")
include("core/hosvd.jl")
include("core/SymDenseMats.jl");

# Local data
include("core/type_ClusterOps.jl")
include("core/type_ClusterBasis.jl")
include("core/type_ClusterSubspace.jl")

#indexing
include("core/type_SparseIndex.jl")
include("core/type_ClusterConfig.jl")
include("core/type_TransferConfig.jl")
include("core/type_FockConfig.jl")
include("core/type_TuckerConfig.jl")
include("core/type_OperatorConfig.jl")
include("core/Indexing.jl")
include("core/build_local_quantities.jl")

include("core/type_AbstractState.jl")
include("core/type_BSstate.jl")
include("core/type_BSTstate.jl")
include("core/type_TPSCIstate.jl")

include("core/type_ClusteredTerm.jl")
include("core/type_ClusteredOperator.jl")

include("core/tucker_inner.jl")
include("core/tucker_build_dense_H_term.jl")
include("core/tucker_contract_dense_H_with_state.jl")
include("core/tucker_form_sigma_block_expand.jl")
include("core/tucker_outer.jl")
include("core/tucker_pt.jl")
include("core/bst.jl")
include("core/bst_helpers.jl")

include("core/tpsci_inner.jl")
include("core/tpsci_matvec_thread.jl")
include("core/tpsci_pt1_wavefunction.jl")
include("core/tpsci_pt2_energy.jl")
include("core/tpsci_outer.jl")
include("core/tpsci_helpers.jl")
include("core/tpsci.jl")

include("core/dense_inner.jl")
include("core/dense_outer.jl")
include("core/spt_variance.jl")

#
#####################################

export RDM
export ClusterMeanField
export InCoreInts
export Molecule
export Atom
export MOCluster
export ClusterBasis
export ClusterSubspace
export ClusteredOperator
export TPSCIstate
export BSTstate
export ClusterConfig
export FockConfig
export TuckerConfig
export n_orb
export add_subspace!
export add_fockconfig!
export expand_each_fock_space!
export block_sparse_tucker
export correlation_functions
end
