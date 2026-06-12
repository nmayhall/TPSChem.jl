#
# NO-CMF walkthrough on the H8 test system (2 clusters of 4 orbitals).
#
# The non-orthogonal CMF method solves a separate CMF-CI for each FockConfig in
# a user-chosen list (neutral + charge-transfer configs here), so each
# FockConfig gets cluster states polarized by its own mean-field environment,
# then couples one (or a few) tensor product states per FockConfig in a small
# CI. See docs/src/nocmf_design.md for the theory.
#
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using JLD2
using Printf

# CMF-optimized orbitals, integrals and clusters from the test fixture
fixture = joinpath(dirname(pathof(TPSChem)), "..", "test", "_testdata_cmf_h8.jld2")
@load fixture ints d1 clusters init_fspace e_fci

# FockConfig list: neutral reference + single charge transfers between clusters
fc0 = FockConfig(init_fspace)                  # [(2,2),(2,2)]
fcs = [fc0,
       FockConfig([(3, 2), (1, 2)]),            # α transfer 2→1
       FockConfig([(1, 2), (3, 2)]),            # α transfer 1→2
       FockConfig([(2, 3), (2, 1)]),            # β transfer 2→1
       FockConfig([(2, 1), (2, 3)])]            # β transfer 1→2

#
# Level 0: fixed per-FockConfig cMF states, coupled and diagonalized.
# max_roots controls how many cluster states per parent enter the union
# working basis (the coordinate system for matrix elements).
e0, state, env = nocmf_level0(ints, clusters, fcs, max_roots=2, nkeep=1, dguess=d1)

#
# Route A benchmark: TPSCI over the full product space of the union basis —
# a strict lower bound for every NO-CMF state in the union span.
eA, _ = nocmf_routeA(clusters, fcs, env.union_bases, env.cluster_ops, env.clustered_ham)

#
# Level 1a: "resonating CMF" — variationally relax each FockConfig's TPS in
# the presence of the others (ALS sweeps, monotone).
e1_hist = nocmf_optimize!(state, env.cluster_ops, env.clustered_ham, verbose=1)

#
# Level 2: grow the rank — add a second, independently optimized TPS per
# FockConfig (generalized eigenproblem handles the non-orthogonality).
blocks, coeffs = nocmf_split_blocks(state)
e2_hist = nocmf_rank_growth!(blocks, coeffs, env.cluster_ops, env.clustered_ham)

#
# Level 1b: escape the initial union span — augment the working basis with
# embedded-Hamiltonian residuals and re-optimize, repeatedly.
e1b_hist, _, _ = nocmf_level1b(ints, clusters, fcs, max_roots=2, cycles=3, dguess=d1)

@printf("\n  %-28s %16s %12s\n", "method", "E(elec)", "E-FCI / mH")
for (name, e) in [("CMF-CI (single FockConfig)", env.e_cmf[1] - ints.h0),
                  ("NO-CMF level 0",             e0[1]),
                  ("NO-CMF level 1a",            e1_hist[end]),
                  ("NO-CMF level 2 (rank 2)",    e2_hist[end]),
                  ("NO-CMF level 1b",            e1b_hist[end]),
                  ("Route A (union TPSCI)",      eA[1]),
                  ("FCI",                        e_fci)]
    @printf("  %-28s %16.10f %12.3f\n", name, e, (e - e_fci) * 1000)
end
