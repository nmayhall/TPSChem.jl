#
# Generate a CMF test/example fixture for a 1D hydrogen chain clustered into
# adjacent ATOM PAIRS (2 orbitals per cluster in STO-3G).
#
#   * Geometry: `natoms` H atoms equally spaced by `r` along the z-axis,
#     H_i at (0, 0, (i-1)*r).
#   * Clusters: [(1,2), (3,4), ...]  -> natoms/2 clusters of 2 orbitals each.
#   * Filling: half-filled neutral singlet, na = nb = natoms/2, each 2-orbital
#     cluster references (nα,nβ) = (1,1).
#   * Orbitals: Löwdin-localized, then CMF orbital-optimized (cmf_oo); the
#     integrals are saved in that optimized basis. `e_fci` is the exact STO-3G
#     FCI energy for the geometry.
#
# Saves `_testdata_cmf_hchain_<natoms>.jld2` with `ints d1 clusters init_fspace e_fci`,
# matching the format of the H8 fixture so it drops into the oxci tests/examples.
#
# Requires a working pyscf (PyCall). Run from the package root:
#   julia --project test/generate_test_data_hchain.jl
#
using TPSChem.QCBase
using TPSChem.RDM
using TPSChem.ClusterMeanField
using Printf
using JLD2

function generate_hchain_data(; natoms::Int=8, r::Float64=1.0, basis::String="sto-3g")
    iseven(natoms) || error("natoms must be even to form 2-atom clusters")

    # 1D chain along z
    atoms = [Atom(i, "H", [0.0, 0.0, (i - 1) * r]) for i in 1:natoms]

    na = natoms ÷ 2
    nb = natoms ÷ 2

    mol = Molecule(0, 1, atoms, basis)
    mf  = TPSChem.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = TPSChem.pyscf_build_ints(mol, mf.mo_coeff, zeros(nbas, nbas))
    e_fci, d1_fci, d2_fci = TPSChem.pyscf_fci(ints, na, nb)
    @printf(" FCI Energy (electronic): %12.8f\n", e_fci)

    # Löwdin-localize the canonical MOs, then rotate integrals into that basis
    C  = mf.mo_coeff
    Cl = TPSChem.localize(mf.mo_coeff, "lowdin", mf)
    S  = TPSChem.get_ovlp(mf)
    U  = C' * S * Cl
    ints = TPSChem.orbital_rotation(ints, U)

    # Adjacent atom pairs -> 2-orbital clusters
    nclusters = natoms ÷ 2
    orb_ranges  = [(2i - 1):(2i) for i in 1:nclusters]
    clusters    = [MOCluster(i, collect(orb_ranges[i])) for i in 1:nclusters]
    init_fspace = [(1, 1) for _ in 1:nclusters]
    display(clusters)

    # CMF orbital optimization at the reference Fock sectors
    d1 = RDM1(n_orb(ints))
    e_cmf, Uoo, d1 = cmf_oo(ints, clusters, init_fspace, d1,
                            max_iter_oo=100, verbose=0, gconv=1e-6, method="bfgs")
    ints = TPSChem.orbital_rotation(ints, Uoo)
    @printf(" CMF Energy (total):      %12.8f\n", e_cmf)

    fname = "_testdata_cmf_hchain_$(natoms).jld2"
    @save fname ints d1 clusters init_fspace e_fci
    @printf(" wrote %s   (%d clusters of 2 orbitals, r = %.3f)\n", fname, nclusters, r)
    return fname
end

generate_hchain_data(natoms=8, r=1.0)
