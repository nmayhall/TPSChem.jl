using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using LinearAlgebra
using Printf
using Test
using JLD2

@testset "Oxidation-State CI h8 (2 clusters)" begin
    @load "_testdata_cmf_h8.jld2" ints d1 clusters init_fspace e_fci

    fc0 = FockConfig(init_fspace)            # [(2,2),(2,2)]
    fcs = [fc0,
           FockConfig([(3, 2), (1, 2)]),      # α CT 2→1
           FockConfig([(1, 2), (3, 2)]),      # α CT 1→2
           FockConfig([(2, 3), (2, 1)]),      # β CT 2→1
           FockConfig([(2, 1), (2, 3)])]      # β CT 1→2

    #
    # 1) single-FockConfig limit must reproduce the CMF-CI energy
    e_cmf, _ = oxci_cmf_solutions(ints, clusters, [fc0], max_roots=1, dguess=d1)
    e0s, _, _ = oxci_solve(ints, clusters, [fc0], max_roots=1, dguess=d1)
    @test isapprox(e0s[1] + ints.h0, e_cmf[1], atol=1e-8)

    #
    # 2) Method A over the full list: bound chain FCI ≤ Union ≤ Method A ≤ CMF
    e0, state, env = oxci_solve(ints, clusters, fcs, max_roots=1, dguess=d1)
    eA, _ = oxci_union_benchmark(clusters, fcs, env.union_bases, env.cluster_ops, env.clustered_ham)
    # note: the fixture's e_fci is electronic (no h0), same convention as eA/e0
    @printf(" h8: E(FCI) %12.8f  E(Union) %12.8f  E(MethodA) %12.8f  E(CMF) %12.8f\n",
            e_fci, eA[1], e0[1], e_cmf[1] - ints.h0)
    @test e_fci <= eA[1] + 1e-9
    @test eA[1] <= e0[1] + 1e-9
    @test e0[1] + ints.h0 <= e_cmf[1] + 1e-9

    #
    # 3) invariance to the union stacking order (working basis is scaffolding)
    e0r, _, _ = oxci_solve(ints, clusters, reverse(fcs), max_roots=1, dguess=d1)
    @test isapprox(e0[1], e0r[1], atol=1e-8)

    #
    # 4) Method B (subspace) ALS on a richer union (max_roots=2, one TPS per FockConfig):
    #    monotone, lowers its Method A start, still bounded by its union-product benchmark
    e0b, stateb, envb = oxci_solve(ints, clusters, fcs, max_roots=2, nkeep=1, dguess=d1)
    eAb, _ = oxci_union_benchmark(clusters, fcs, envb.union_bases, envb.cluster_ops, envb.clustered_ham)
    e1_hist = oxci_variational_sweep!(stateb, envb.cluster_ops, envb.clustered_ham, max_iter=30, tol=1e-10)
    e1 = e1_hist[end]
    @printf(" h8: E(MethodB) %12.8f (MethodA start %12.8f, Union %12.8f)\n", e1, e0b[1], eAb[1])
    @test all(diff(e1_hist) .<= 1e-9)         # monotone non-increasing
    @test e1 <= e0b[1] + 1e-9                  # at least as good as Method A
    @test eAb[1] <= e1 + 1e-9                  # within the union span, Union bounds it

    #
    # 5) Method B (rank growth) rank growth: one extra TPS per FockConfig. On h8 with
    #    max_roots=2 the rank-2 CP expansion saturates the union product space,
    #    so the energy must land on union-product benchmark exactly.
    blocks, c = oxci_split_blocks(stateb)
    e2_hist = oxci_rank_growth!(blocks, c, envb.cluster_ops, envb.clustered_ham, tol=1e-10)
    @printf(" h8: E(MethodB-rank)  %12.8f (Union %12.8f)\n", e2_hist[end], eAb[1])
    @test e2_hist[end] <= e1 + 1e-9
    @test eAb[1] <= e2_hist[end] + 1e-9
    @test isapprox(e2_hist[end], eAb[1], atol=1e-7)   # saturated CP rank limit
    @test length(blocks) == 10

    #
    # 6) Method B (full relaxation): residual-augmented union — must not rise above 1a, and on
    #    this system pushes below the *initial* union's union-product benchmark bound
    e1b_hist, _, _ = oxci_variational_relax(ints, clusters, fcs, max_roots=2, cycles=3,
                                   dguess=d1, tol=1e-9)
    @printf(" h8: E(MethodB-relax) %12.8f (MethodB %12.8f)\n", e1b_hist[end], e1)
    @test e1b_hist[end] <= e1 + 1e-8
    @test e_fci <= e1b_hist[end] + 1e-9       # still variational
end

@testset "Oxidation-State CI h12 (5 clusters, spectator overlaps)" begin
    @load "_testdata_cmf_h12_64bit.jld2" ints d1 clusters init_fspace

    fc0 = FockConfig(init_fspace)             # [(1,1),(1,1),(2,2),(1,1),(1,1)]
    # CT between clusters 1 and 2 — clusters 3,4,5 are genuine spectators whose
    # sector appears in several parents (non-orthogonal cross-parent states)
    fcs = [fc0,
           FockConfig([(0, 1), (2, 1), (2, 2), (1, 1), (1, 1)]),
           FockConfig([(2, 1), (0, 1), (2, 2), (1, 1), (1, 1)])]

    e_cmf, _ = oxci_cmf_solutions(ints, clusters, [fc0], max_roots=1, dguess=d1)
    e0, state, env = oxci_solve(ints, clusters, fcs, max_roots=1, dguess=d1)
    eA, _ = oxci_union_benchmark(clusters, fcs, env.union_bases, env.cluster_ops, env.clustered_ham)
    @printf(" h12: E(Union) %12.8f  E(MethodA) %12.8f  E(CMF) %12.8f\n",
            eA[1] + ints.h0, e0[1] + ints.h0, e_cmf[1])
    @test eA[1] <= e0[1] + 1e-9
    @test e0[1] + ints.h0 <= e_cmf[1] + 1e-9

    # invariance to union stacking order with genuine cross-parent overlaps
    e0r, _, _ = oxci_solve(ints, clusters, reverse(fcs), max_roots=1, dguess=d1)
    @test isapprox(e0[1], e0r[1], atol=1e-8)

    # ALS with spectators present (richer union, one TPS per FockConfig)
    e0b, stateb, envb = oxci_solve(ints, clusters, fcs, max_roots=2, nkeep=1, dguess=d1)
    eAb, _ = oxci_union_benchmark(clusters, fcs, envb.union_bases, envb.cluster_ops, envb.clustered_ham)
    e1_hist = oxci_variational_sweep!(stateb, envb.cluster_ops, envb.clustered_ham, max_iter=20, tol=1e-10)
    @printf(" h12: E(MethodB) %12.8f (MethodA start %12.8f, Union %12.8f)\n",
            e1_hist[end], e0b[1], eAb[1])
    @test all(diff(e1_hist) .<= 1e-9)
    @test e1_hist[end] <= e0b[1] + 1e-9
    @test eAb[1] <= e1_hist[end] + 1e-9
end

@testset "Oxidation-State CI (Method A) → FCI as local states grow (h8)" begin
    @load "_testdata_cmf_h8.jld2" ints d1 clusters init_fspace e_fci

    # Complete FockConfig list for the target global (Na,Nb) sector: with every
    # oxidation state present *and* the local basis taken complete, Method A spans
    # the full Hilbert space and is exact. Truncating the local basis (max_roots)
    # is the convergence knob the ansatz is built around.
    Na = sum(f[1] for f in init_fspace); Nb = sum(f[2] for f in init_fspace)
    no1 = length(clusters[1]); no2 = length(clusters[2])
    fcs = FockConfig{2}[]
    for na1 in 0:no1, nb1 in 0:no1
        na2 = Na - na1; nb2 = Nb - nb1
        (0 <= na2 <= no2 && 0 <= nb2 <= no2) || continue
        push!(fcs, FockConfig([(na1, nb1), (na2, nb2)]))
    end

    # Sweep the number of local eigenstates kept per cluster per FockConfig.
    energies = Float64[]
    for m in 1:4
        e, _, _ = oxci_solve(ints, clusters, fcs; max_roots=m, nkeep=m, dguess=d1)
        push!(energies, e[1])
        @printf(" h8 Method A: nstates/cluster=%d  E=%14.8f  gap-to-FCI=%.2e\n",
                m, e[1], e[1] - e_fci)
    end

    # variational: every truncation is an upper bound on FCI
    @test all(energies .>= e_fci - 1e-9)
    # nested local bases ⇒ enlarging the basis can only lower the energy
    @test all(diff(energies) .<= 1e-9)
    # and it is genuinely converging downward toward FCI
    @test energies[end] < energies[1] - 1e-6
    @test (energies[end] - e_fci) < (energies[1] - e_fci)
end
