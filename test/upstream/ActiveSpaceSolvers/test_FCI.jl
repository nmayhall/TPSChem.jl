using TPSChem.QCBase
using TPSChem.ActiveSpaceSolvers
using TPSChem.InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ

@testset "FCI" begin
#function run()
   
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")


    ints = InCoreInts(h0, h1, h2)
    n_elec_a = 3
    n_elec_b = 3

    norb = size(h1,1)
    ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)

    display(ansatz)

    # test build_H_matrix
    Hmat = build_H_matrix(ints, ansatz)
    @time e,v = Arpack.eigs(Hmat, nev = 10, which=:SR)
    ref_e = -3.155304800477 # from pyscf in generate_integrals.py
    @test isapprox(e[1], ref_e , atol=1e-10)
   
    ref = [-3.1553048004765447, -2.849049024311162, -2.5973991507252805]
    
    Hmap = LinearMap(ints, ansatz)
    e, v = Arpack.eigs(Hmap, nev=3, which=:SR)
    println(e)
    println(ref)
    @test all(isapprox.(e, ref, atol=1e-10))


    Hmap = LinearMap(ints, ansatz)
    e = v' * Matrix(Hmap * v)
    display(e)
    @test all(isapprox.(diag(e), ref, atol=1e-10))

    solver = SolverSettings(nroots=3, package="arpack")
    println(solver)
    solution = solve(ints, ansatz, solver)
    display(solution)
    @test all(isapprox.(solution.energies, ref, atol=1e-10))

    # davidson
    #
    solver = SolverSettings(nroots=3, package="davidson")
    println(solver)
    solution = solve(ints, ansatz, solver)
    display(solution)
    @test all(isapprox.(solution.energies, ref, atol=1e-10))


    # test 1RDMs

    op_ca_aa = compute_operator_ca_aa(solution, solution)
    op_ca_bb = compute_operator_ca_bb(solution, solution)
    for i in 1:3
        da = op_ca_aa[:,:,i,i]
        db = op_ca_bb[:,:,i,i]
        rdm1a, rdm1b = ActiveSpaceSolvers.compute_1rdm(solution, root=i)
        @printf(" Trace = %12.8f %12.8f\n", tr(da), tr(db))
        @printf(" Trace = %12.8f %12.8f\n", tr(rdm1a), tr(rdm1b))
        good = true
        good = good && isapprox(tr(da), tr(rdm1a), atol=1e-10)
        good = good && isapprox(tr(db), tr(rdm1b), atol=1e-10)
        good = good && isapprox(tr(da), 3.0, atol=1e-10)
        good = good && isapprox(tr(db), 3.0, atol=1e-10)
        @test good
    end
    #op_ca_ab = compute_operator_ca_ab(solution, solution)

    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)
    
    no = n_orb(ints)
    e = ints.h0
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (da[p,q] + db[p,q])
    end

    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * daa[p,q,r,s]
        e += .5 * ints.h2[p,q,r,s] * dbb[p,q,r,s]
        e +=      ints.h2[p,q,r,s] * dab[p,q,r,s]
    end
    
    @test isapprox(e, ref_e) 
    # test S2
    
    S2 = solution' * build_S2_matrix(ansatz) * solution
    for i in 1:3
        @printf(" %4i S^2 = %12.8f\n", i, S2[i,i])
    end
    @test all(isapprox.(diag(S2), [0,2,0], atol=1e-8))

    
    # test SVD
    a = svd_state(solution, 3, 3, 1e-3) 

    # this is not yet working for some reason
    #
    #solver = SolverSettings(nroots=3, package="krylovkit")
    #println(solver)
    #solution = solve(ansatz, ints, solver)
    #display(solution)
    #@test all(isapprox.(solution.energies.+ints.h0, ref, atol=1e-10))

    # string stuff
    #display(ActiveSpaceSolvers.StringCI.string_to_index("110010"))
    @test ActiveSpaceSolvers.FCI.string_to_index("110010") == 19
    
    # test single dimension cases
    ansatz = FCIAnsatz(norb, norb, norb)
    solution = solve(ints, ansatz, solver)
    display(solution)

    ansatz = FCIAnsatz(norb, norb, 0)
    solution = solve(ints, ansatz, solver)
    display(solution)
    
    ansatz = FCIAnsatz(norb, 0, norb)
    solution = solve(ints, ansatz, solver)
    display(solution)
    @test isapprox(solution.energies[1], -0.131427758762, atol=1e-9)
    
    ansatz = FCIAnsatz(norb, 0, 0)
    solution = solve(ints, ansatz, solver)
    display(solution)
end
#run()
