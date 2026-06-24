using TPSChem.QCBase
using TPSChem.RDM  
using TPSChem.ActiveSpaceSolvers
using Test
using NPZ
using Printf
using TPSChem.InCoreIntegrals
using JLD2
using LinearAlgebra
using Random


function generate_test_data()
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")
    
    ints = InCoreInts(h0, h1, h2)

    ansatz = FCIAnsatz(6,3,3)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
    display(solution)
    rdm1a, rdm1b = compute_1rdm(solution)
    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)
    e_ref = solution.energies[1]
    @save "h6_sto3g/rdms_33.jld2" e_ref ints da db daa dbb dab


    
    ansatz = FCIAnsatz(6,4,2)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
    display(solution)
    e_ref = solution.energies[1]
    
    rdm1a, rdm1b = compute_1rdm(solution)
    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)

    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)

    @test isapprox(compute_energy(ints, d1, d2), solution.energies[1])
    @test isapprox(compute_energy(ints, ssRDM1(d1), ssRDM2(d2)), solution.energies[1])

    @save "h6_sto3g/rdms_42.jld2" e_ref ints da db daa dbb dab
end
