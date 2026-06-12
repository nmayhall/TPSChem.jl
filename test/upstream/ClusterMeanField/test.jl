
using TPSChem.ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.ActiveSpaceSolvers

function run()
# 
    atoms = []
    #push!(atoms,Atom(1,"O",[0,0,0]))
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,1,0]))
    push!(atoms,Atom(4,"H",[3,1,0]))
    push!(atoms,Atom(5,"H",[4,2,0]))
    push!(atoms,Atom(6,"H",[5,2,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,3,3)
     @printf(" FCI Energy: %12.8f\n", e_fci)

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

    clusters    = [(1:3),(4:6)]
    init_fspace = [(2,1),(2,1)]

    clusters    = [(1:4),(5:6)]
    init_fspace = [(1,1),(1,1)]
    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]


    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf
    rdm1b = rdm_mf
    
    #sol = solve(ints, FCIAnsatz(6,3,3), SolverSettings())
    #display(sol)
    #ints.h2 .= 0.0

    e, d1a, d1b, d1_dict, d2_dict = cmf_ci(ints, clusters, init_fspace, rdm1a, rdm1b, 
                        verbose=1, sequential=false)
    
    
    rdm1, rdm2 = ClusterMeanField.assemble_full_rdm(clusters, d1_dict, d2_dict)
   
    norb = n_orb(ints)

    e2 = compute_energy(ints, rdm1, rdm2) 
    e3 = ClusterMeanField.compute_cmf_energy(ints, d1_dict, d2_dict, clusters) 
    @printf(" E1 = %12.8f E2 = %12.8f E3 = %12.8f\n", e, e2, e3)

    println(" Gradient Full")
    grad = ClusterMeanField.build_orbital_gradient(ints, rdm1, rdm2)
    hess = ClusterMeanField.build_orbital_hessian(ints, rdm1, rdm2)

    display(ClusterMeanField.unpack_gradient(grad, norb))

    g_anl = ClusterMeanField.orbital_gradient_analytical(ints, clusters, zeros(norb*(norb-1)÷2), init_fspace, d1a, d1b, ci_conv=1e-10)
    println(" Gradient Analytical")
    display(ClusterMeanField.unpack_gradient(g_anl, norb))

    g_num = ClusterMeanField.orbital_gradient_numerical(ints, clusters, zeros(norb*(norb-1)÷2), init_fspace, d1a, d1b, ci_conv=1e-10, stepsize=1e-4)
    println(" Gradient Numerical")
    display(ClusterMeanField.unpack_gradient(g_num, norb))

    display(ints.h2[1,2,3,4])
    display(ints.h2[2,1,3,4])
    display(ints.h2[1,2,4,3])
    display(ints.h2[2,1,4,3])
    display(ints.h2[3,4,1,2])
    display(ints.h2[3,4,2,1])
    display(ints.h2[4,3,1,2])
    display(ints.h2[4,3,2,1])
#    function grad_num(ints, rdm1, rdm2; stepsize=1e-5)
#        N = n_orb(ints)
#        rdm = InCoreInts(0.0, rdm1, rdm2)
#        for i in 1:length(k)
#            k = zeros(N*(N-1)÷2)
#            k[i] += stepsize
#            U = exp(ClusterMeanField.unpack_gradient(k, N))
#            
#            e1 = compute_energy(ints_i, rdm.h1, rdm.h2)
#        end
#    end

    #e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
    #                          verbose=0, gconv=1e-7, method="cg",sequential=true)
    
    #Ccmf = Cl*U
    #ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
  
end

run()
