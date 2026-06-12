using TPSChem.QCBase  
using TPSChem.RDM  
using TPSChem.ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.ActiveSpaceSolvers

function numgrad(ints, d1::RDM1, d2::RDM2)


    @printf(" Energy: %12.8f\n", compute_energy(ints, d1, d2))

    no = n_orb(ints)
    k = RDM.pack_gradient(zeros(no,no), no)
    grad = deepcopy(k)

    stepsize=1e-5
    for i in 1:length(k)
        ki = zeros(no*(no-1)÷2) 
        ki[i] += stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e1 = compute_energy(intsi, d1, d2)
        
        ki = zeros(no*(no-1)÷2) 
        ki[i] -= stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e2 = compute_energy(intsi, d1, d2)
        
        grad[i] = (e1-e2)/(2*stepsize)
    end
    println(" Numerical Gradient: ")
    display(norm(grad))
    #display(grad)
    
    println(" Analytical Gradient: ")
    g = build_orbital_gradient(ints, d1, d2)
    display(norm(g))
    @printf("   Error: %12.8f\n",norm(g-grad))
    #display(g)

    
    println()
    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    #display(g)
    println(" Analytical Gradient: ")
    display(norm(g))
    @printf("   Error: %12.8f\n",norm(g-grad))
    println(" Now compare gradients:")
    println(" Numerical:")
    display(round.(ClusterMeanField.unpack_gradient(grad, no), digits=7))
    println(" Analytical:")
    display(round.(ClusterMeanField.unpack_gradient(g, no), digits=7))
    #error("here")
end

@testset "CMF" begin
    #h0 = npzread("h6_sto3g/h0.npy")
    #h1 = npzread("h6_sto3g/h1.npy")
    #h2 = npzread("h6_sto3g/h2.npy")
    
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,3,3)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

    ClusterMeanField.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
    Cl = ClusterMeanField.localize(mf.mo_coeff,"lowdin",mf)
    ClusterMeanField.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = ClusterMeanField.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = RDM1(rdm_mf, rdm_mf)

    e_fci = -3.155304800477
    e_scf = -3.09169726403968
   
    sol = solve(ints, FCIAnsatz(6,3,3), SolverSettings())
    display(sol)
    
    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    f1 = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
    
    @test isapprox(f1[1], -2.97293813654926351, atol=1e-10)
    
    e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, 
                              verbose=0, gconv=1e-6, method="cg",sequential=true)
    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)

    ansatze=[FCIAnsatz(2,1,1),FCIAnsatz(2,1,1),FCIAnsatz(2,1,1)]
    e_cmf, U_n, d1_n = ClusterMeanField.cmf_oo_newton(ints, clusters, init_fspace, ansatze,rdm1, maxiter_oo = 400,
                           tol_oo=1e-6, 
                           tol_d1=1e-9, 
                           tol_ci=1e-11,
                           verbose=4, 
                           zero_intra_rots = true,
                           sequential=true)
    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)

    e_cmf, U_n, d1_n = ClusterMeanField.cmf_oo_newton(ints, clusters, init_fspace, ansatze,rdm1, maxiter_oo = 400,
                           tol_oo=1e-6, 
                           tol_d1=1e-9, 
                           tol_ci=1e-11,
                           verbose=4, 
                           zero_intra_rots = false,
                           sequential=true)
    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)
    e_cmf, U, d1 = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace,ansatze, rdm1,
                           maxiter_oo   = 500,
                           maxiter_ci   = 200,
                           maxiter_d1   = 200,
                           verbose      = 0,
                           tol_oo       = 1e-6,
                           tol_d1       = 1e-9,
                           tol_ci       = 1e-11,
                           sequential   = true,
                           diis_start   = 1,
                           max_ss_size  = 24)
    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)


    e_cmf, U, d1 = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace,ansatze, rdm1,
                           maxiter_oo   = 500,
                           maxiter_ci   = 200,
                           maxiter_d1   = 200,
                           verbose      = 0,
                           tol_oo       = 1e-6,
                           tol_d1       = 1e-9,
                           tol_ci       = 1e-11,
                           diis_start   = 1,
                           max_ss_size  = 24,
                           zero_intra_rots = false,
                           sequential=true)
    e_cmf, U, d1 = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace,ansatze, rdm1,
                           maxiter_oo   = 500,
                           maxiter_ci   = 200,
                           maxiter_d1   = 200,
                           verbose      = 0,
                           tol_oo       = 1e-6,
                           tol_d1       = 1e-9,
                           tol_ci       = 1e-11,
                           diis_start   = 1,
                           max_ss_size  = 24,
                           zero_intra_rots = true,
                           orb_hessian=false,
                           sequential=true)
    e_cmf, U, d1 = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace,ansatze, rdm1,
                           maxiter_oo   = 500,
                           maxiter_ci   = 200,
                           maxiter_d1   = 200,
                           verbose      = 0,
                           tol_oo       = 1e-6,
                           tol_d1       = 1e-9,
                           tol_ci       = 1e-11,
                           diis_start   = 1,
                           max_ss_size  = 24,
                           zero_intra_rots = false,
                           orb_hessian=false,
                           sequential=true)
    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)
    Ccmf = Cl*U
    ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
end
    