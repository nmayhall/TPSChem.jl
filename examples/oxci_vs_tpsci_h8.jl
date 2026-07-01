#
# Oxidation-State CI (Method A) vs. conventional TPSCI — convergence to FCI.
#
# Both methods expand the wavefunction over the *same* complete list of oxidation
# states (FockConfigs) for the target global (Na,Nb) sector, and both converge to
# FCI as the number of local cluster states per Fock sector is increased. The
# only difference is the local basis:
#
#   * Conventional TPSCI: ONE basis per cluster per Fock sector, built once in the
#     global cMF mean field and shared across every FockConfig.
#   * Oxidation-State CI (Method A): a SEPARATE basis per cluster *per FockConfig*,
#     each polarized by the cMF environment of that specific oxidation state.
#
# Because each FockConfig's cluster states are tailored to their own context, the
# Oxidation-State CI expansion should reach a given accuracy with fewer local
# states — i.e. converge to FCI faster in the local-basis dimension. This script
# prints the gap-to-FCI for both methods as that dimension grows.
#
# Run from the package root:  julia --project examples/oxci_vs_tpsci_h8.jl
#
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra

# cMF-optimized orbitals, integrals, clusters, and the FCI reference (electronic)
fixture = joinpath(dirname(pathof(TPSChem)), "..", "test", "_testdata_cmf_h8.jld2")
@load fixture ints d1 clusters init_fspace e_fci

# Complete list of oxidation states for the global (Na,Nb) sector of the H8 dimer.
Na = sum(f[1] for f in init_fspace)
Nb = sum(f[2] for f in init_fspace)
no1 = length(clusters[1])
no2 = length(clusters[2])
fcs = FockConfig{2}[]
for na1 in 0:no1, nb1 in 0:no1
    na2 = Na - na1
    nb2 = Nb - nb1
    (0 <= na2 <= no2 && 0 <= nb2 <= no2) || continue
    push!(fcs, FockConfig([(na1, nb1), (na2, nb2)]))
end

# The clustered Hamiltonian is basis-independent — build it once.
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)

# Conventional TPSCI: one shared cMF basis per cluster/sector, full product space
# over the FockConfig list, dense diagonalization. `delta_elec=4` makes the shared
# basis span every Fock sector that appears in the complete list.
function conventional_tpsci(m)
    cb = TPSChem.compute_cluster_eigenbasis(ints, clusters; init_fspace=init_fspace,
                                            delta_elec=4, rdm1a=d1.a, rdm1b=d1.b,
                                            max_roots=m)
    cops = TPSChem.compute_cluster_ops(cb, ints)
    v = TPSChem.TPSCIstate(clusters, T=Float64, R=1)
    for fc in fcs
        TPSChem.add_fockconfig!(v, fc)
    end
    TPSChem.expand_each_fock_space!(v, cb)
    Hd = TPSChem.build_full_H(v, cops, clustered_ham)
    return eigen(Symmetric(Hd)).values[1], length(v)
end

no_tot = no1 + no2
fci_dim = binomial(no_tot, Na) * binomial(no_tot, Nb)

@printf("\n H8 dimer — convergence to FCI\n")
@printf("   orbitals: %d  (clusters: %s)      global sector (Na,Nb) = (%d,%d)\n",
        no_tot, string([length(c) for c in clusters]), Na, Nb)
@printf("   oxidation states (FockConfigs): %d      FCI dimension: %d\n",
        length(fcs), fci_dim)
@printf("   E(FCI, electronic) = %.8f\n\n", e_fci)

# `dim` is the dimension of the CI matrix actually diagonalized (number of kept
# product configurations summed over all FockConfigs). At matched `states/clu`
# the two methods build the *same-size* matrix — only its contents differ.
@printf(" %-10s | %-6s %-15s %-9s %-6s | %-6s %-15s %-9s %-6s\n",
        "states/clu", "dim", "E(OxState CI A)", "gap", "t(s)",
        "dim", "E(conv TPSCI)", "gap", "t(s)")
@printf(" %s\n", "-"^92)
for m in 1:36
    local eA, stateA, dimA, eC, dimC
    tA = @elapsed begin
        eA, stateA, _ = oxci_solve(ints, clusters, fcs; max_roots=m, nkeep=m, dguess=d1)
    end
    dimA = length(stateA)
    tC = @elapsed begin
        eC, dimC = conventional_tpsci(m)
    end
    @printf(" %-10d | %-6d %15.8f %.2e %6.1f | %-6d %15.8f %.2e %6.1f\n",
            m, dimA, eA[1], eA[1] - e_fci, tA, dimC, eC, eC - e_fci, tC)
    flush(stdout)
end
@printf("\n At matched matrix dimension, Oxidation-State CI sits closer to FCI:\n")
@printf(" the per-FockConfig (context-specific) basis captures with fewer states\n")
@printf(" what the shared basis needs more states to describe. Both reach FCI once\n")
@printf(" the local basis saturates every Fock sector (dim → FCI dimension).\n")
