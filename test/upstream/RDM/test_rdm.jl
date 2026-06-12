using TPSChem.QCBase
using TPSChem.RDM  
using Test
using NPZ
using Printf
using TPSChem.InCoreIntegrals
using JLD2
using LinearAlgebra
using Random


@testset "RDM" begin
    @load "h6_sto3g/rdms_33.jld2" 
    d11 = RDM1(6)
    d22 = RDM2(6)
    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)
    
    d11.a .= d1.a
    d11.b .= d1.b
    d22.aa .= d2.aa
    d22.ab .= d2.ab
    d22.bb .= d2.bb
    @test isapprox(compute_energy(ints, d1, d2), e_ref)
    @test isapprox(compute_energy(ints, d11, d22), e_ref)
    @test isapprox(compute_energy(ints, ssRDM1(d1), ssRDM2(d2)), e_ref)
    
    display(norm(d1.a - RDM1(d2).a))
    display(norm(d1.b - RDM1(d2).b))
    @test isapprox(norm(d1.a - RDM1(d2).a),0,atol=1e-14)
    @test isapprox(norm(d1.b - RDM1(d2).b),0,atol=1e-14)

    fa, fb = RDM.compute_fock(ints, d1)
    e,C = eigen(fa+fb)
    display(e)
    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
    
    g = build_orbital_gradient(ints, d1, d2)
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
   
    println()
    println()
    @load "h6_sto3g/rdms_42.jld2" 
    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)
    @test isapprox(compute_energy(ints, d1, d2), e_ref)
    @test isapprox(compute_energy(ints, ssRDM1(d1), ssRDM2(d2)), e_ref)

#    no = n_orb(ints)
#    #d1.a .= Matrix(1.0I,no,no)
#    #d1.b .= Matrix(1.0I,no,no)
#    display(tr(d1))
#    display(tr(d1.a))
#    display(tr(d1.b))
#    display(tr(RDM1(d2)))
#    display(tr(RDM1(RDM2(d1))))
#    
#    tmp1 = d1.a / tr(d1.a)
#    tmp2 = RDM1(d2).a / tr(RDM1(d2).a)
#    tmp3 = RDM1(RDM2(d1)).a / tr(RDM1(RDM2(d1)).a)
#    println(" tmp1 - tmp2")
#    display(tmp1-tmp2)
#    println(" tmp1 - tmp3")
#    display(tmp1-tmp3)
#    @test isapprox(norm(d1.a - RDM1(RDM2(d1)).a),0,atol=1e-14)
   
    @test isapprox(norm(d1.a - RDM1(d2).a),0,atol=1e-14)
    @test isapprox(norm(d1.b - RDM1(d2).b),0,atol=1e-14)
    @test isapprox(tr(d1.a), 4, atol=1e-14)
    @test isapprox(tr(d1.b), 2, atol=1e-14)
    @test isapprox(tr(d1), 6, atol=1e-14)
    @test isapprox(tr(d2), 30, atol=1e-12)
    println()
    fa, fb = RDM.compute_fock(ints, d1)
    e,C = eigen(fa+fb)
    display(e)

    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
    
    g = build_orbital_gradient(ints, d1, d2)
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)


    no = n_orb(ints)
    G = rand(no,no)
    G = G - G'
    U = exp(G)
    e0 = compute_energy(ints, d1, d2)
    ints_1 = orbital_rotation(ints, U)
    d1_1 = orbital_rotation(d1, U)
    d2_1 = orbital_rotation(d2, U)
    e1 = compute_energy(ints_1, d1_1, d2_1)
    @test isapprox(e0, e1, atol=1e-12)
end


function numgrad()
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")
    
    ints = InCoreInts(h0, h1, h2)

    ansatz = FCIAnsatz(6,3,2)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
    display(solution)
    Random.seed!(1)
    v = rand(size(solution.vectors)...)

    v = v/norm(v)
    solution.vectors .= v

    rdm1a, rdm1b = compute_1rdm(solution)
    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)
    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)
    e1 = compute_energy(ints, d1, d2)
    e2 = compute_energy(ints, ssRDM1(d1), ssRDM2(d2))

    @printf(" %12.8f\n", e1)
    @printf(" %12.8f\n", e2)

    no = n_orb(ints)
    k = RDM.pack_gradient(zeros(no,no), no)
    grad = deepcopy(k)

    stepsize=1e-6
    for i in 1:length(k)
        ki = deepcopy(k)
        ki[i] += stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e1 = compute_energy(intsi, d1, d2)
        
        ki = deepcopy(k)
        ki[i] -= stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e2 = compute_energy(intsi, d1, d2)
        
        grad[i] = (e1-e2)/(2*stepsize)
    end
    println(" Numerical Gradient: ")
    display(grad)
    
    println(" Analytical Gradient: ")
    g = build_orbital_gradient(ints, d1, d2)
    display(g)

    println()
    display(norm(g-grad))
    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    display(g)

    println()
    display(norm(g-grad))
end
