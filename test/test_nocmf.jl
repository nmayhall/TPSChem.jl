using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using LinearAlgebra
using Printf
using Test
using JLD2

@testset "NO-CMF h8 (2 clusters)" begin
    @load "_testdata_cmf_h8.jld2" ints d1 clusters init_fspace e_fci

    fc0 = FockConfig(init_fspace)            # [(2,2),(2,2)]
    fcs = [fc0,
           FockConfig([(3, 2), (1, 2)]),      # α CT 2→1
           FockConfig([(1, 2), (3, 2)]),      # α CT 1→2
           FockConfig([(2, 3), (2, 1)]),      # β CT 2→1
           FockConfig([(2, 1), (2, 3)])]      # β CT 1→2

    #
    # 1) single-FockConfig limit must reproduce the CMF-CI energy
    e_cmf, _ = nocmf_cmf_solutions(ints, clusters, [fc0], max_roots=1, dguess=d1)
    e0s, _, _ = nocmf_level0(ints, clusters, [fc0], max_roots=1, dguess=d1)
    @test isapprox(e0s[1] + ints.h0, e_cmf[1], atol=1e-8)

    #
    # 2) level 0 over the full list: bound chain FCI ≤ RouteA ≤ level0 ≤ CMF
    e0, state, env = nocmf_level0(ints, clusters, fcs, max_roots=1, dguess=d1)
    eA, _ = nocmf_routeA(clusters, fcs, env.union_bases, env.cluster_ops, env.clustered_ham)
    # note: the fixture's e_fci is electronic (no h0), same convention as eA/e0
    @printf(" h8: E(FCI) %12.8f  E(RouteA) %12.8f  E(lvl0) %12.8f  E(CMF) %12.8f\n",
            e_fci, eA[1], e0[1], e_cmf[1] - ints.h0)
    @test e_fci <= eA[1] + 1e-9
    @test eA[1] <= e0[1] + 1e-9
    @test e0[1] + ints.h0 <= e_cmf[1] + 1e-9

    #
    # 3) invariance to the union stacking order (working basis is scaffolding)
    e0r, _, _ = nocmf_level0(ints, clusters, reverse(fcs), max_roots=1, dguess=d1)
    @test isapprox(e0[1], e0r[1], atol=1e-8)

    #
    # 4) level 1a ALS on a richer union (max_roots=2, one TPS per FockConfig):
    #    monotone, lowers its level-0 start, still bounded by its Route A
    e0b, stateb, envb = nocmf_level0(ints, clusters, fcs, max_roots=2, nkeep=1, dguess=d1)
    eAb, _ = nocmf_routeA(clusters, fcs, envb.union_bases, envb.cluster_ops, envb.clustered_ham)
    e1_hist = nocmf_optimize!(stateb, envb.cluster_ops, envb.clustered_ham, max_iter=30, tol=1e-10)
    e1 = e1_hist[end]
    @printf(" h8: E(lvl1a) %12.8f (lvl0 start %12.8f, RouteA %12.8f)\n", e1, e0b[1], eAb[1])
    @test all(diff(e1_hist) .<= 1e-9)         # monotone non-increasing
    @test e1 <= e0b[1] + 1e-9                  # at least as good as level 0
    @test eAb[1] <= e1 + 1e-9                  # within the union span, RouteA bounds it
end

@testset "NO-CMF h12 (5 clusters, spectator overlaps)" begin
    @load "_testdata_cmf_h12_64bit.jld2" ints d1 clusters init_fspace

    fc0 = FockConfig(init_fspace)             # [(1,1),(1,1),(2,2),(1,1),(1,1)]
    # CT between clusters 1 and 2 — clusters 3,4,5 are genuine spectators whose
    # sector appears in several parents (non-orthogonal cross-parent states)
    fcs = [fc0,
           FockConfig([(0, 1), (2, 1), (2, 2), (1, 1), (1, 1)]),
           FockConfig([(2, 1), (0, 1), (2, 2), (1, 1), (1, 1)])]

    e_cmf, _ = nocmf_cmf_solutions(ints, clusters, [fc0], max_roots=1, dguess=d1)
    e0, state, env = nocmf_level0(ints, clusters, fcs, max_roots=1, dguess=d1)
    eA, _ = nocmf_routeA(clusters, fcs, env.union_bases, env.cluster_ops, env.clustered_ham)
    @printf(" h12: E(RouteA) %12.8f  E(lvl0) %12.8f  E(CMF) %12.8f\n",
            eA[1] + ints.h0, e0[1] + ints.h0, e_cmf[1])
    @test eA[1] <= e0[1] + 1e-9
    @test e0[1] + ints.h0 <= e_cmf[1] + 1e-9

    # invariance to union stacking order with genuine cross-parent overlaps
    e0r, _, _ = nocmf_level0(ints, clusters, reverse(fcs), max_roots=1, dguess=d1)
    @test isapprox(e0[1], e0r[1], atol=1e-8)

    # ALS with spectators present (richer union, one TPS per FockConfig)
    e0b, stateb, envb = nocmf_level0(ints, clusters, fcs, max_roots=2, nkeep=1, dguess=d1)
    eAb, _ = nocmf_routeA(clusters, fcs, envb.union_bases, envb.cluster_ops, envb.clustered_ham)
    e1_hist = nocmf_optimize!(stateb, envb.cluster_ops, envb.clustered_ham, max_iter=20, tol=1e-10)
    @printf(" h12: E(lvl1a) %12.8f (lvl0 start %12.8f, RouteA %12.8f)\n",
            e1_hist[end], e0b[1], eAb[1])
    @test all(diff(e1_hist) .<= 1e-9)
    @test e1_hist[end] <= e0b[1] + 1e-9
    @test eAb[1] <= e1_hist[end] + 1e-9
end
