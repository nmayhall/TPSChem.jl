using TPSChem.QCBase
using TPSChem.ActiveSpaceSolvers
using TPSChem.InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ

#function run()
@testset "h4" begin
 
    h0 = npzread("h4_ccpvdz/h0.npy")
    h1 = npzread("h4_ccpvdz/h1.npy")
    h2 = npzread("h4_ccpvdz/h2.npy")


    ints = InCoreInts(h0, h1, h2)

    ints = subset(ints, 1:8)

    #k = rand(8,8);
    #U = exp(k-k')
    #ints = orbital_rotation(ints,U)
    n_elec_a = 3
    n_elec_b = 3

    norb = n_orb(ints)
    ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)

    display(ansatz)

    # test build_H_matrix
    #@test isapprox(e[1], ref_e , atol=1e-10)
  
    ref = [-2.22833756, -1.9548734, -1.84003465]
    
    Hmap = LinearMap(ints, ansatz)
    e, v = Arpack.eigs(Hmap, nev=3, which=:SR)
    println(e)
    #println(ref)
    #@test all(isapprox.(e, ref, atol=1e-10))


    Hmap = LinearMap(ints, ansatz)
    e = v' * Matrix(Hmap * v)
    display(e)
    #@test all(isapprox.(diag(e), ref, atol=1e-10))

    solver = SolverSettings(nroots=20, package="arpack")
    println(solver)
    @time solution = solve(ints, ansatz, solver)
    display(solution)

    S2matvec = 

    S2 = build_S2_matrix(solution.ansatz)
    s2a = solution.vectors' * S2 * solution.vectors
    s2b = solution.vectors' * apply_S2_matrix(solution.ansatz, solution.vectors)
   
    @test isapprox(norm(s2a-s2b), 0, atol=1e-12)

    s2a = diag(solution.vectors' * S2 * solution.vectors)
    s2 = compute_s2(solution)

    for r in 1:solver.nroots
        display(round(s2[r]))
        @test isapprox(s2a[r], s2[r], atol=1e-10)
    end
    @test round(s2[1]) == 0
    @test round(s2[2]) == 2
    @test round(s2[3]) == 0
    @test round(s2[4]) == 0
    @test round(s2[5]) == 2
    

    # S-
    v1 = solution.vectors[:,1:15]
    @time v2, ansatz2 = apply_sminus(v1, solution.ansatz)
    v3, ansatz3 = apply_sminus(v2, ansatz2)

    Hmap2 = LinearMap(ints, ansatz2)
    Hmap3 = LinearMap(ints, ansatz3)
    println(" new e")
    e1 = diag(v1'*Matrix((Hmap*v1)))
    e2 = diag(v2'*Matrix((Hmap2*v2)))
    e3 = diag(v3'*Matrix((Hmap3*v3)))

    display(e1)
    println()
    display(e2)
    println()
    display(e3)
    println()
    @test isapprox(e1[2], e2[1], atol=1e-12)
    @test isapprox(e1[15], e3[1], atol=1e-12)

    # S+
    v1 = solution.vectors[:,1:15]
    @time v2, ansatz2 = apply_splus(v1, solution.ansatz)
    v3, ansatz3 = apply_splus(v2, ansatz2)

    Hmap2 = LinearMap(ints, ansatz2)
    Hmap3 = LinearMap(ints, ansatz3)
    println(" new e")
    e1 = diag(v1'*Matrix(Hmap*v1))
    e2 = diag(v2'*Matrix(Hmap2*v2))
    
    e3 = diag(v3'*Matrix(Hmap3*v3))
    display(ansatz)
    display(e1)
    display(ansatz2)
    display(e2)
    display(ansatz3)
    display(e3)
    @test isapprox(e1[2], e2[1], atol=1e-12)
    @test isapprox(e1[15], e3[1], atol=1e-12)
end
#run()
