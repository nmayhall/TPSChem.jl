using TPSChem.QCBase
using TPSChem.RDM
using TPSChem.ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.ActiveSpaceSolvers
using PyCall


    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    push!(atoms,Atom(7,"H",[6,0,0]))
    push!(atoms,Atom(8,"H",[7,0,0]))
    basis = "sto-3g"

    mol     = Molecule(0,3,atoms,basis)
    pymol = ClusterMeanField.make_pyscf_mole(mol)
    pyscf = pyimport("pyscf")
    mf = pyscf.scf.ROHF(pymol).run()

    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #ClusterMeanField.pyscf_fci(ints, 8,0)
    
    clusters    = [(1:3),(4:5),(6:8)]
    init_fspace = [(3,3),(2,0),(0,0)]
    

    # Broken!
    clusters    = [(1:4),(5:8)]
    init_fspace = [(4,3),(1,0)]

    
    
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    
    sol = solve(ints, FCIAnsatz(8,5,3), SolverSettings())
    display(sol)

    rdm_mf = mf.make_rdm1()
    rdm_mfa = rdm_mf[1,:,:]
    rdm_mfb = rdm_mf[2,:,:]
    rdm_mfa = mf.mo_coeff'*mf.get_ovlp()*rdm_mfa*mf.get_ovlp()*mf.mo_coeff
    rdm_mfb = mf.mo_coeff'*mf.get_ovlp()*rdm_mfb*mf.get_ovlp()*mf.mo_coeff
    rdm1 = RDM1(rdm_mfa, rdm_mfb)
    #display(rdm1)
    #display(round.(rdm1.a, digits=7))
    #display(round.(rdm1.b, digits=7))

    println(" ------------------------------------------------") 
    f1 = ClusterMeanField.cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
    @printf(" ROHF Energy: %12.8f\n", mf.e_tot)
    @printf(" CMF  Energy: %12.8f\n", f1[1])

    @printf(" These should match!\n")
    
    tmp1 = Dict{Integer, RDM1{Float64}}()
    tmp2 = Dict{Integer, RDM2{Float64}}()
    for ci in clusters
        tmp1[ci.idx] = RDM1(rdm1.a[ci.orb_list, ci.orb_list], rdm1.b[ci.orb_list, ci.orb_list])
        tmp2[ci.idx] = RDM2(tmp1[ci.idx])
    end
    e = compute_energy(ints, tmp1, tmp2, clusters)
    @printf("1RDM  Energy: %12.8f\n", e)
   
    e = compute_energy(ints, rdm1)
    @printf("2RDM  Energy: %12.8f\n", e)
    
    e = compute_energy(ints, f1[2], f1[3], clusters)
    @printf("3RDM  Energy: %12.8f\n", e)

    d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, f1[2], f1[3])
    e = compute_energy(ints, d1, d2)
    @printf("4RDM  Energy: %12.8f\n", e)
    
    e = compute_energy(ints, RDM1(RDM2(rdm1)))
    @printf("5RDM  Energy: %12.8f\n", e)
    
    d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, tmp1, tmp2)
    e = compute_energy(ints, d1, d2)
    @printf("6RDM  Energy: %12.8f\n", e)
    e_cmf, U, d1 = cmf_oo(ints, clusters, init_fspace, rdm1, 
                              verbose=0, gconv=1e-6, method="cg",sequential=true)

    C = mf.mo_coeff*U
    ClusterMeanField.pyscf_write_molden(mol, mf.mo_coeff, filename="scf.molden")
    ClusterMeanField.pyscf_write_molden(mol, C, filename="cmf.molden")
#end

