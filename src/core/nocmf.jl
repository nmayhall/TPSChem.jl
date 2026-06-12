"""
NO-CMF: non-orthogonal CMF state interaction.

Solve a separate CMF-CI (fixed orbitals, no orbital optimization) for each
FockConfig in a user-supplied list, so every FockConfig gets cluster states
polarized by its own mean-field environment. The resulting tensor product
states are coupled in a small CI and diagonalized.

Key structural facts (see docs/src/nocmf_design.md for the full derivation):

- TPS from *different* FockConfigs are exactly orthogonal (particle number on
  at least one cluster differs), and TPS within one FockConfig are orthonormal
  (products of one CMF solution's eigenstates), so the global CI is a standard
  eigenproblem. The "non-orthogonality" only appears inside Hamiltonian matrix
  elements: spectator clusters contribute overlap matrices instead of deltas.
- Matrix elements are evaluated through a per-cluster, per-sector orthonormal
  "union working basis" spanning all parents' states. ClusterOps are computed
  once in this basis; each parent's states are exact coefficient columns in it
  (the `factors`). The working basis is scaffolding only — it never enters the
  variational space.
- Each FockConfig block is stored as a Tucker block of an `SPTstate` whose
  factors are that parent's states in working-basis coordinates. The existing
  SPT sigma build already forms cross-block spectator overlaps explicitly
  (`contract_dense_H_with_state`), which is exactly what NO-CMF needs.

Energy convention: like `ci_solve`, all energies returned here are electronic
(no `ints.h0`); add `ints.h0` for totals.
"""

using LinearAlgebra
using OrderedCollections

"""
    nocmf_cmf_solutions(ints, clusters, fock_configs; max_roots=2, dguess=nothing, verbose=0, ...)

Converge a CMF-CI (no orbital optimization) for each FockConfig in
`fock_configs` and build that parent's cluster eigenbases (restricted to the
parent's own Fock sectors) in the embedding field of its converged CMF density.

# Returns
- `energies::Vector{T}`: CMF-CI energy per FockConfig (includes `ints.h0`,
  matching the `cmf_ci` convention)
- `parent_bases::Vector{Vector{ClusterBasis}}`: outer index = FockConfig
  (parent), inner index = cluster
"""
function nocmf_cmf_solutions(ints::InCoreInts{T}, clusters::Vector{MOCluster},
                             fock_configs::Vector{FockConfig{N}};
                             max_roots  = 2,
                             dguess     = nothing,
                             verbose    = 0,
                             maxiter_d1 = 40,
                             tol_d1     = 1e-7,
                             tol_ci     = 1e-9,
                             sequential = false) where {T,N}
    length(clusters) == N || throw(DimensionMismatch("clusters vs FockConfig length"))

    energies = Vector{T}()
    parent_bases = Vector{Vector{ClusterBasis{FCIAnsatz,T}}}()

    for fc in fock_configs
        fspace = [(Int(fc[i][1]), Int(fc[i][2])) for i in 1:N]
        verbose == 0 || @printf(" nocmf: CMF-CI for FockConfig %s\n", string(fspace))

        rdm1 = dguess === nothing ? RDM1(n_orb(ints)) : deepcopy(dguess)
        e, rdm1_dict, _ = ClusterMeanField.cmf_ci(ints, clusters, fspace, rdm1,
                                                  maxiter_d1 = maxiter_d1,
                                                  tol_d1     = tol_d1,
                                                  tol_ci     = tol_ci,
                                                  verbose    = verbose,
                                                  sequential = sequential)
        d1 = ClusterMeanField.assemble_full_rdm(clusters, rdm1_dict)

        cb = compute_cluster_eigenbasis(ints, clusters,
                                        init_fspace = fspace,
                                        delta_elec  = 0,
                                        rdm1a       = d1.a,
                                        rdm1b       = d1.b,
                                        max_roots   = max_roots,
                                        verbose     = verbose,
                                        T           = T)
        push!(energies, T(e))
        push!(parent_bases, cb)
    end
    return energies, parent_bases
end

"""
    nocmf_setup(ints, clusters, fock_configs; max_roots=1, svd_thresh=1e-8, verbose=0, ...)

Run the full NO-CMF preparation pipeline: per-FockConfig CMF-CI solutions,
union working basis + factors, ClusterOps in the working basis, and the
clustered Hamiltonian. Returns a NamedTuple
`(e_cmf, parent_bases, union_bases, factors, cluster_ops, clustered_ham)`.
"""
function nocmf_setup(ints::InCoreInts{T}, clusters::Vector{MOCluster},
                     fock_configs::Vector{FockConfig{N}};
                     max_roots  = 1,
                     svd_thresh = 1e-8,
                     verbose    = 0,
                     kwargs...) where {T,N}
    e_cmf, parent_bases = nocmf_cmf_solutions(ints, clusters, fock_configs;
                                              max_roots=max_roots, verbose=verbose, kwargs...)
    union_bases, factors, recon_err = build_union_basis(parent_bases, svd_thresh=svd_thresh, verbose=verbose)
    recon_err < 1e-6 || @warn(" nocmf_setup: union basis discards parent-state components", recon_err)
    cluster_ops = compute_cluster_ops(union_bases, ints)
    clustered_ham = extract_ClusteredTerms(ints, clusters)
    return (e_cmf=e_cmf, parent_bases=parent_bases, union_bases=union_bases,
            factors=factors, cluster_ops=cluster_ops, clustered_ham=clustered_ham)
end

"""
    nocmf_state(clusters, fock_configs, factors, union_bases; R=1, nkeep=nothing)

Build an `SPTstate` with one Tucker block per FockConfig. Block `p`'s factors
are parent `p`'s cluster states expressed in the union working basis
(`factors[p][i][sector]`), its TuckerConfig is the full union-sector range on
every cluster, and its core holds the CI coefficients over parent `p`'s kept
product states. Cores are initialized to zero — set a guess with
`set_vector!` or solve with `nocmf_ci_solve`.

`nkeep` truncates each block to the lowest `nkeep` cluster states per cluster
(e.g. `nkeep=1` gives one TPS per FockConfig — the level-1a ansatz — while the
union basis stays as rich as the parents allow, giving the ALS optimization
room to rotate).
"""
function nocmf_state(clusters::Vector{MOCluster},
                     fock_configs::Vector{FockConfig{N}},
                     factors,
                     union_bases::Vector{ClusterBasis{A,T}};
                     R=1, nkeep=nothing) where {T,N,A}
    length(unique(fock_configs)) == length(fock_configs) ||
        error("duplicate FockConfigs require the (level 2) generalized solver")

    # P spaces: the full union range for every sector present (Q is empty)
    p_spaces = Vector{ClusterSubspace}()
    for ci in clusters
        tss = ClusterSubspace(ci)
        for (sec, sol) in union_bases[ci.idx]
            tss[sec] = 1:size(sol.vectors, 2)
        end
        push!(p_spaces, tss)
    end
    q_spaces = [get_ortho_compliment(tss, union_bases[tss.cluster.idx]) for tss in p_spaces]

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}}}()
    state = SPTstate{T,N,R}(clusters, data, p_spaces, q_spaces)

    for (p, fc) in enumerate(fock_configs)
        add_fockconfig!(state, fc)

        blockfactors = Vector{Matrix{T}}()
        ranges = Vector{UnitRange{Int}}()
        for ci in clusters
            sec = fc[ci.idx]
            haskey(factors[p][ci.idx], sec) || error("missing factor for cluster $(ci.idx) sector $sec")
            U = factors[p][ci.idx][sec]
            nkeep === nothing || (U = U[:, 1:min(nkeep, size(U, 2))])
            size(U, 1) == size(union_bases[ci.idx][sec].vectors, 2) || error("factor/union dimension mismatch")
            push!(blockfactors, U)
            push!(ranges, 1:size(U, 1))
        end
        tconfig = TuckerConfig(ranges)
        core = ntuple(r -> zeros(T, ntuple(i -> size(blockfactors[i], 2), N)), R)
        state[fc][tconfig] = Tucker{T,N,R}(core, ntuple(i -> blockfactors[i], N))
    end
    return state
end

"""
    build_H_dense(state::SPTstate, cluster_ops, clustered_ham; nbody=4)

Build the dense Hamiltonian in the space spanned by `state`'s Tucker blocks
(core-coefficient basis) by applying `build_sigma!` to unit vectors. Intended
for the small NO-CMF variational spaces; dimension = `length(state)`.
"""
function build_H_dense(state::SPTstate{T,N,R}, cluster_ops, clustered_ham; nbody=4) where {T,N,R}
    len = length(state)
    Hd = zeros(T, len, len)
    for q in 1:len
        v = zeros(T, len)
        v[q] = one(T)
        vec_i = SPTstate(state, R=1)
        set_vector!(vec_i, v)
        sig = deepcopy(vec_i)
        zero!(sig)
        build_sigma!(sig, vec_i, cluster_ops, clustered_ham, nbody=nbody, verbose=0)
        Hd[:, q] .= get_vector(sig)[:, 1]
    end
    return Symmetric(0.5 .* (Hd .+ Hd'))
end

"""
    nocmf_ci_solve(state, cluster_ops, clustered_ham; verbose=0)

Diagonalize the Hamiltonian in the space spanned by `state`'s blocks (dense —
NO-CMF spaces are small by design). Returns `(energies, state)` with the `R`
lowest roots written into `state`'s cores. Energies are electronic (no h0).
"""
function nocmf_ci_solve(state::SPTstate{T,N,R}, cluster_ops, clustered_ham; verbose=0, nbody=4) where {T,N,R}
    Hd = build_H_dense(state, cluster_ops, clustered_ham, nbody=nbody)
    F = eigen(Hd)
    R <= length(F.values) || error("R=$R roots requested but dimension is $(length(F.values))")
    set_vector!(state, Matrix{T}(F.vectors[:, 1:R]))
    e = Vector{T}(F.values[1:R])
    if verbose > 0
        @printf(" NO-CMF CI dimension %5i\n", length(state))
        for r in 1:R
            @printf("   root %3i  E(elec) = %16.10f\n", r, e[r])
        end
    end
    return e, state
end

"""
    nocmf_level0(ints, clusters, fock_configs; max_roots=1, R=1, svd_thresh=1e-8, verbose=0, ...)

Level-0 NO-CMF: fixed per-FockConfig cMF cluster states, coupled and
diagonalized. Returns `(energies, state, env)` where `energies` are electronic
(add `ints.h0`), `state` is the solved `SPTstate`, and `env` is the
`nocmf_setup` NamedTuple (reusable for Route A / level 1a).
"""
function nocmf_level0(ints::InCoreInts{T}, clusters::Vector{MOCluster},
                      fock_configs::Vector{FockConfig{N}};
                      max_roots  = 1,
                      R          = 1,
                      nkeep      = nothing,
                      svd_thresh = 1e-8,
                      verbose    = 0,
                      kwargs...) where {T,N}
    env = nocmf_setup(ints, clusters, fock_configs;
                      max_roots=max_roots, svd_thresh=svd_thresh, verbose=verbose, kwargs...)
    state = nocmf_state(clusters, fock_configs, env.factors, env.union_bases, R=R, nkeep=nkeep)
    e, state = nocmf_ci_solve(state, env.cluster_ops, env.clustered_ham, verbose=verbose)
    return e, state, env
end

"""
    nocmf_expectation(state::SPTstate, cluster_ops, clustered_ham; nbody=4)

⟨Ψ|H|Ψ⟩/⟨Ψ|Ψ⟩ for the first root. Electronic energy (no h0).
"""
function nocmf_expectation(state::SPTstate{T,N,R}, cluster_ops, clustered_ham; nbody=4) where {T,N,R}
    sig = deepcopy(state)
    zero!(sig)
    build_sigma!(sig, state, cluster_ops, clustered_ham, nbody=nbody, verbose=0)
    v = get_vector(state)[:, 1]
    s = get_vector(sig)[:, 1]
    return dot(v, s) / dot(v, v)
end

"""
    nocmf_optimize!(state::SPTstate, cluster_ops, clustered_ham; max_iter=50, tol=1e-8, verbose=0)

Level-1a "resonating CMF": variationally optimize the cluster states defining
each FockConfig's TPS, restricted to the span of the union working basis, by
ALS sweeps. Requires rank-1 blocks (one TPS per FockConfig; build the state
with `max_roots=1`) and a single root.

Each update of (cluster j, FockConfig g) solves the bordered (d+1)-dimensional
generalized eigenproblem of the design doc §5: trial = Σ_q y_q |probe_q⟩ + w |rest⟩,
where probe_q replaces block g's cluster-j factor with the q-th union basis
vector and |rest⟩ is the current superposition of all other blocks. The lowest
eigenvalue simultaneously relaxes the factor, its CI weight, and the relative
weight of the rest — so the energy is monotonically non-increasing. After each
sweep the CI coefficients are re-solved with fixed factors (also variational).

With a single FockConfig the resonance terms vanish and each update is exactly
the CMF-CI embedded-cluster diagonalization (in the union subspace).

Returns the energy history (electronic, no h0), one entry per sweep, starting
with the initial energy. Mutates `state` (factors and cores).
"""
function nocmf_optimize!(state::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                         max_iter = 50,
                         tol      = 1e-8,
                         verbose  = 0,
                         nbody    = 4) where {T,N,R}
    R == 1 || error("nocmf_optimize! optimizes a single root (state-averaging NYI)")

    fcs = collect(keys(state.data))
    for fc in fcs
        length(state[fc]) == 1 || error("expected one TuckerConfig per FockConfig block")
        tuck = first(values(state[fc]))
        all(size(f, 2) == 1 for f in tuck.factors) ||
            error("level 1a requires rank-1 blocks (build the state with max_roots=1)")
    end

    # ensure we start from a solved coefficient vector
    if norm(get_vector(state)) < 1e-12
        nocmf_ci_solve(state, cluster_ops, clustered_ham, nbody=nbody)
    end

    e_hist = Vector{T}()
    push!(e_hist, T(nocmf_expectation(state, cluster_ops, clustered_ham, nbody=nbody)))
    verbose == 0 || @printf(" NO-CMF ALS initial E(elec) = %16.10f\n", e_hist[end])

    for iter in 1:max_iter
        for g in fcs
            tconfig, tuck_g = first(state[g])

            for j in 1:N
                d = size(tuck_g.factors[j], 1)

                # probe block: cluster j's slot opened to the full union range
                pfactors = ntuple(i -> i == j ? Matrix{T}(I, d, d) : tuck_g.factors[i], N)
                pcore = (zeros(T, ntuple(i -> i == j ? d : 1, N)),)
                pdata = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,1}}}()
                probe = SPTstate{T,N,1}(state.clusters, pdata, state.p_spaces, state.q_spaces)
                add_fockconfig!(probe, g)
                probe[g][tconfig] = Tucker{T,N,1}(pcore, pfactors)

                # A: embedded Hamiltonian of cluster j in block g (probe basis)
                A = Matrix(build_H_dense(probe, cluster_ops, clustered_ham, nbody=nbody))

                # rest of the wavefunction (all other blocks, current coefficients)
                # (build fresh — OrderedDict iteration breaks after delete!)
                rdata = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}}}()
                rest = SPTstate{T,N,R}(state.clusters, rdata, state.p_spaces, state.q_spaces)
                for f2 in fcs
                    f2 == g && continue
                    add_fockconfig!(rest, f2)
                    for (tc2, tk2) in state[f2]
                        rest[f2][tc2] = deepcopy(tk2)
                    end
                end

                y = zeros(T, d)
                w = one(T)
                if length(rest.data) == 0
                    # single block: plain embedded-cluster diagonalization (CMF-CI step)
                    F = eigen(Symmetric(A))
                    y .= F.vectors[:, 1]
                else
                    # v_q = ⟨probe_q|H|rest⟩ — half-projected sigma vector
                    sigp = deepcopy(probe)
                    zero!(sigp)
                    build_sigma!(sigp, rest, cluster_ops, clustered_ham, nbody=nbody, verbose=0)
                    v = get_vector(sigp)[:, 1]

                    # κ = ⟨rest|H|rest⟩, ρ = ⟨rest|rest⟩
                    sigr = deepcopy(rest)
                    zero!(sigr)
                    build_sigma!(sigr, rest, cluster_ops, clustered_ham, nbody=nbody, verbose=0)
                    rv = get_vector(rest)[:, 1]
                    κ = dot(rv, get_vector(sigr)[:, 1])
                    ρ = dot(rv, rv)
                    ρ > 1e-14 || error("rest of wavefunction has zero norm")

                    M = zeros(T, d + 1, d + 1)
                    M[1:d, 1:d] .= A
                    M[1:d, end] .= v
                    M[end, 1:d] .= v
                    M[end, end] = κ
                    Nm = Matrix{T}(I, d + 1, d + 1)
                    Nm[end, end] = ρ

                    F = eigen(Symmetric(M), Symmetric(Nm))
                    z = F.vectors[:, 1]
                    y .= z[1:d]
                    w = z[end]
                end

                # update block g: factor j ← y/|y|, core ← |y|; rescale the rest by w
                ny = norm(y)
                if ny > 1e-12
                    newfactors = ntuple(i -> i == j ? reshape(y ./ ny, d, 1) : tuck_g.factors[i], N)
                    newcore = ntuple(r -> fill(ny, ntuple(i -> 1, N)), R)
                    tuck_g = Tucker{T,N,R}(newcore, newfactors)
                    state[g][tconfig] = tuck_g
                else
                    # block decouples at this step: zero its weight, keep the factor
                    first(values(state[g])).core[1] .= zero(T)
                end
                for f2 in fcs
                    f2 == g && continue
                    first(values(state[f2])).core[1] .*= w
                end
            end
        end

        # re-solve CI coefficients with fixed factors (variational; reuses blocks)
        e, _ = nocmf_ci_solve(state, cluster_ops, clustered_ham, nbody=nbody)
        push!(e_hist, e[1])
        verbose == 0 || @printf(" NO-CMF ALS sweep %3i  E(elec) = %16.10f  ΔE = %9.2e\n",
                                iter, e_hist[end], e_hist[end] - e_hist[end-1])
        abs(e_hist[end] - e_hist[end-1]) > tol || break
    end
    return e_hist
end

"""
    nocmf_level1a(ints, clusters, fock_configs; svd_thresh=1e-8, max_iter=50, tol=1e-8, verbose=0, ...)

Level-1a NO-CMF: level 0 (one TPS per FockConfig) followed by ALS factor
optimization within the union working basis. Returns `(e_hist, state, env)`;
energies electronic (no h0).
"""
function nocmf_level1a(ints::InCoreInts{T}, clusters::Vector{MOCluster},
                       fock_configs::Vector{FockConfig{N}};
                       max_roots  = 4,
                       svd_thresh = 1e-8,
                       max_iter   = 50,
                       tol        = 1e-8,
                       verbose    = 0,
                       kwargs...) where {T,N}
    # union built from max_roots states per parent (optimization room);
    # the variational blocks keep one TPS per FockConfig (nkeep=1)
    e0, state, env = nocmf_level0(ints, clusters, fock_configs;
                                  max_roots=max_roots, nkeep=1, R=1,
                                  svd_thresh=svd_thresh, verbose=verbose, kwargs...)
    e_hist = nocmf_optimize!(state, env.cluster_ops, env.clustered_ham;
                             max_iter=max_iter, tol=tol, verbose=verbose)
    return e_hist, state, env
end

"""
    nocmf_routeA(clusters, fock_configs, union_bases, cluster_ops, clustered_ham; nroots=1)

Benchmark: standard TPSCI over the *full product space* of the union working
basis, restricted to the listed FockConfigs. Variationally contains every
NO-CMF state whose factors lie in the union span (level 0 and level 1a), so
its energy is a lower bound for those. Dense build — small systems only.
Returns `(energies, tpsci_state)`; energies are electronic (no h0).
"""
function nocmf_routeA(clusters::Vector{MOCluster},
                      fock_configs::Vector{FockConfig{N}},
                      union_bases::Vector{ClusterBasis{A,T}},
                      cluster_ops, clustered_ham;
                      nroots=1) where {T,N,A}
    v = TPSCIstate(clusters, T=T, R=nroots)
    for fc in unique(fock_configs)
        add_fockconfig!(v, fc)
    end
    expand_each_fock_space!(v, union_bases)
    Hd = build_full_H(v, cluster_ops, clustered_ham)
    F = eigen(Symmetric(Hd))
    set_vector!(v, Matrix{T}(F.vectors[:, 1:nroots]))
    return Vector{T}(F.values[1:nroots]), v
end
