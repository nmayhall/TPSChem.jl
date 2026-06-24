using TPSChem.QCBase  
using TPSChem.RDM  
using TPSChem.ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.ActiveSpaceSolvers
using JLD2
 
@testset "HESSIAN" begin

    atoms = []
    push!(atoms,Atom(1,"H",[0,0.0,0]))
    push!(atoms,Atom(2,"H",[50,0.0,0]))
    push!(atoms,Atom(3,"H",[100,100.0,0]))
    push!(atoms,Atom(4,"H",[150,100,0]))
    push!(atoms,Atom(5,"H",[200,200,0]))
    push!(atoms,Atom(6,"H",[250,200,0]))

    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    @printf(" HF  Energy: %12.8f\n", mf.e_tot)

    C = mf.mo_coeff
    rdm_mf = mf.mo_coeff'*mf.get_ovlp()*mf.make_rdm1()*mf.get_ovlp()*mf.mo_coeff / 2
    rdm1 = RDM1(rdm_mf, rdm_mf)
    @printf(" Should be E(HF):  %12.8f\n", compute_energy(ints, rdm1, RDM2(rdm1)))

    Cl = ClusterMeanField.localize(mf.mo_coeff,"lowdin",mf)
    S = ClusterMeanField.get_ovlp(mf)
    U =  C' * S * Cl

    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    rdm1 = orbital_rotation(rdm1,U)
    println(" done.")
    flush(stdout)

    init_fspace = [(1,1),(1,1),(1,1)]
    cluster   =[(1:2),(3:4),(5:6)]
    clusters = [MOCluster(i,collect(cluster[i])) for i = 1:length(cluster)]
    display(clusters)
    n = n_orb(ints)
    kappa=zeros(n*(n-1)÷2)
    e, rdm1_dict, rdm2_dict = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        maxiter_d1 = 100, 
                        maxiter_ci = 100, 
                        tol_d1     = 1e-9, 
                        tol_ci     = 1e-10, 
                        verbose    = 0, 
                        sequential = true)

    gd1, gd2 =ClusterMeanField.assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)

    hess_num_function=ClusterMeanField.orbital_hessian_finite_difference(ints,clusters,kappa,init_fspace,gd1)

    hess_num_grad= ClusterMeanField.orbital_hessian_numerical(ints,clusters,kappa,init_fspace,gd1)

    orbital_hessian=RDM.build_orbital_hessian(ints,gd1,gd2)

    num_grad=ClusterMeanField.orbital_gradient_numerical(ints, clusters, kappa, init_fspace, gd1)

    analytical_grad= build_orbital_gradient(ints, gd1, gd2)
    
    function mocluster_to_cluster(mo::MOCluster)
        return UnitRange{Int}(minimum(mo.orb_list), maximum(mo.orb_list))
    end
    transformed_cluster = [mocluster_to_cluster(mo) for mo in clusters]
    n=maximum([maximum(cluster) for cluster in transformed_cluster])
    projection_vector=ClusterMeanField.create_interaction_vector(transformed_cluster,n)
    projection_matrix=ClusterMeanField.create_projection_matrix(projection_vector)
    num_hess_function=hess_num_function.*projection_matrix
    num_hess_gradient=hess_num_grad.*projection_matrix
    orbital_hessian=orbital_hessian.*projection_matrix
    
  
    println(" Analytical Gradient: ")
    display(norm(analytical_grad))
    println("\n Numerical Gradient: ")
    display(norm(num_grad))
    @printf(" \n  Error: %12.8f\n",norm(analytical_grad-num_grad))
    @test isapprox(norm(analytical_grad-num_grad), 0.0, atol=1e-10)


    println(" \n Analytical Hessian: ")
    display(orbital_hessian)
    eigenvalue_hessian=eigvals(orbital_hessian)
    display(eigenvalue_hessian)
    @test all(x-> x>=0.0,eigenvalue_hessian)
    println("\n\n Finite difference Hessian (using function ): ")
    display(num_hess_function)
    println("\n\n Finite difference Hessian (using gradient ): ")
    display(num_hess_gradient)
    # @save "data_fd_hessian.jld2" ints clusters gd1 gd2 num_hess1 num_hess2 projection_matrix
    @test isapprox(norm(orbital_hessian-num_hess_function), 0.0, atol=1e-4)
    @test isapprox(norm(orbital_hessian-num_hess_gradient), 0.0, atol=1e-6)
    @test isapprox(norm(num_hess_function-num_hess_gradient), 0.0, atol=1e-4)
    @test isapprox(num_hess_gradient, orbital_hessian, atol=1e-8)
    @test isapprox(num_hess_function, orbital_hessian, atol=1e-4)
    @test isapprox(num_hess_function, num_hess_gradient, atol=1e-4)
end