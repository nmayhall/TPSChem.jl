"""
    generate_cluster_fock_ansatze(ref_fock, clusters, init_cluster_ansatz, delta_elec; verbose=0)

Generate all Fock sectors reachable within ±delta_elec electrons for each cluster.
"""
function generate_cluster_fock_ansatze(ref_fock,
                                        clusters::Vector{MOCluster},
                                        init_cluster_ansatz::Vector{<:Ansatz},
                                        delta_elec::Vector{Int},
                                        verbose=0)
    ansatze = Vector{Vector{Ansatz}}()
    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)")

    for i in 1:length(clusters)
        verbose == 0 || display(clusters[i])
        delta_e_i = delta_elec[i]
        ni = ref_fock[i][1] + ref_fock[i][2]
        max_e = 2 * length(clusters[i])
        min_e = 0
        sectors = []
        for nj in ni-delta_e_i:ni+delta_e_i
            nj <= max_e || continue
            nj >= min_e || continue
            naj = nj ÷ 2 + nj % 2
            nbj = nj ÷ 2
            if typeof(init_cluster_ansatz[i]) == FCIAnsatz
                push!(sectors, FCIAnsatz(init_cluster_ansatz[i].no, Int(naj), Int(nbj)))
            else
                error("Unsupported ansatz type: $(typeof(init_cluster_ansatz[i]))")
            end
        end
        append!(ansatze, [sectors])
    end
    return ansatze
end

"""
    generate_cluster_fock_ansatze_all(ref_fock, clusters, init_cluster_ansatz, delta_elec; verbose=0)

Generate all Fock sectors reachable within ±delta_elec electrons for each cluster (all spin configurations).
"""
function generate_cluster_fock_ansatze_all(ref_fock,
                                            clusters::Vector{MOCluster},
                                            init_cluster_ansatz::Vector{<:Ansatz},
                                            delta_elec::Vector{Int},
                                            verbose=0)
    ansatze = Vector{Vector{Ansatz}}()
    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)")

    for i in 1:length(clusters)
        verbose == 0 || display(clusters[i])
        delta_e_i = delta_elec[i]
        ni = ref_fock[i][1] + ref_fock[i][2]
        max_e = 2 * length(clusters[i])
        min_e = 0
        sectors = []
        for nj in ni-delta_e_i:ni+delta_e_i
            nj <= max_e || continue
            nj >= min_e || continue
            naj = nj ÷ 2 + nj % 2
            nbj = nj ÷ 2
            if typeof(init_cluster_ansatz[i]) == FCIAnsatz
                push!(sectors, FCIAnsatz(init_cluster_ansatz[i].no, Int(naj), Int(nbj)))
            else
                error("Unsupported ansatz type: $(typeof(init_cluster_ansatz[i]))")
            end
        end
        append!(ansatze, [sectors])
    end
    return ansatze
end

"""
    invariant_orbital_rotations(cluster::Ansatz)

Return pairs of orbitals that are invariant to orbital rotation for the given cluster ansatz.
"""
function invariant_orbital_rotations(cluster::Ansatz)
    invar_pairs = []
    if typeof(cluster) == FCIAnsatz
        for a in 1:cluster.no
            for b in a+1:cluster.no
                push!(invar_pairs, (a, b))
            end
        end
    else
        error("Unsupported ansatz type: $(typeof(cluster))")
    end
    return invar_pairs
end

"""
    invariant_orbital_rotations(init_cluster_ansatz::Vector{<:Ansatz})

Return per-cluster lists of orbital pairs invariant to rotation.
"""
function invariant_orbital_rotations(init_cluster_ansatz::Vector{<:Ansatz})
    invar_pairs = []
    for cl in init_cluster_ansatz
        if typeof(cl) == FCIAnsatz
            pairs = []
            for a in 1:cl.no
                for b in a+1:cl.no
                    push!(pairs, (a, b))
                end
            end
            push!(invar_pairs, pairs)
        else
            error("Unsupported ansatz type: $(typeof(cl))")
        end
    end
    return invar_pairs
end
