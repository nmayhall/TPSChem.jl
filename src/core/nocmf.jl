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
- `rdm1s::Vector{RDM1{T}}`: converged CMF embedding density per FockConfig
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
    rdm1s = Vector{RDM1{T}}()

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
        push!(rdm1s, d1)
    end
    return energies, parent_bases, rdm1s
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
    e_cmf, parent_bases, rdm1s = nocmf_cmf_solutions(ints, clusters, fock_configs;
                                                     max_roots=max_roots, verbose=verbose, kwargs...)
    union_bases, factors, recon_err = build_union_basis(parent_bases, svd_thresh=svd_thresh, verbose=verbose)
    recon_err < 1e-6 || @warn(" nocmf_setup: union basis discards parent-state components", recon_err)
    cluster_ops = compute_cluster_ops(union_bases, ints)
    clustered_ham = extract_ClusteredTerms(ints, clusters)
    return (e_cmf=e_cmf, parent_bases=parent_bases, rdm1s=rdm1s, union_bases=union_bases,
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
    nocmf_level1b(ints, clusters, fock_configs; max_roots=4, cycles=3, res_thresh=1e-5, ...)

Level-1b: alternate (i) level-1a ALS in the current union working basis with
(ii) augmentation of the union by residual directions computed in each
cluster's determinant basis, until the residuals fall below `res_thresh` or
`cycles` is exhausted. The union span only grows and the converged factors are
carried over exactly, so the energy is non-increasing across cycles.

The augmentation direction is the *embedded-Hamiltonian* (diagonal-block)
residual `(H_emb - ε)|x⟩` of each converged factor, projected out of the
current union span — the resonance contribution to the exact gradient is not
included in the direction (the subsequent ALS still treats the enlarged space
fully variationally). Clusters whose determinant dimension exceeds `max_dim_H`
are skipped.

Returns `(e_hist, state, union_bases)`; energies electronic (no h0). Note the
final union basis differs from the level-0/1a one — Route A on the *initial*
union no longer bounds these energies (design doc §7).
"""
function nocmf_level1b(ints::InCoreInts{T}, clusters::Vector{MOCluster},
                       fock_configs::Vector{FockConfig{N}};
                       max_roots  = 4,
                       cycles     = 3,
                       res_thresh = 1e-5,
                       svd_thresh = 1e-8,
                       max_iter   = 50,
                       tol        = 1e-8,
                       max_dim_H  = 5000,
                       verbose    = 0,
                       kwargs...) where {T,N}
    e_cmf, parent_bases, rdm1s = nocmf_cmf_solutions(ints, clusters, fock_configs;
                                                     max_roots=max_roots, verbose=verbose, kwargs...)
    clustered_ham = extract_ClusteredTerms(ints, clusters)

    # residual directions accumulate here as an extra "parent"
    res_cb = [ClusterBasis(ci, T=T) for ci in clusters]
    have_res = false

    e_hist = Vector{T}()
    x_det = nothing       # per FockConfig: converged factors in determinant coordinates
    c_prev = nothing      # per FockConfig: converged core coefficient

    local state, union_bases, cluster_ops
    for cycle in 1:cycles
        stack = have_res ? vcat(parent_bases, [res_cb]) : parent_bases
        union_bases, factors, _ = build_union_basis(stack, svd_thresh=svd_thresh, verbose=verbose)
        cluster_ops = compute_cluster_ops(union_bases, ints)

        state = nocmf_state(clusters, fock_configs, factors, union_bases, R=1, nkeep=1)
        if x_det !== nothing
            # resume exactly from the previous cycle's converged factors
            for (p, fc) in enumerate(fock_configs)
                tc, tk = first(state[fc])
                newf = ntuple(N) do i
                    sec = fc[clusters[i].idx]
                    W = union_bases[i][sec].vectors
                    u = W' * x_det[p][i]
                    reshape(u ./ norm(u), size(W, 2), 1)
                end
                core = (fill(c_prev[p], ntuple(i -> 1, N)),)
                state[fc][tc] = Tucker{T,N,1}(core, newf)
            end
        end

        eh = nocmf_optimize!(state, cluster_ops, clustered_ham,
                             max_iter=max_iter, tol=tol, verbose=verbose)
        append!(e_hist, eh)

        # converged factors in determinant coordinates (for residuals and resume)
        x_det = Vector{Vector{Vector{T}}}()
        c_prev = Vector{T}()
        for (p, fc) in enumerate(fock_configs)
            tc, tk = first(state[fc])
            push!(x_det, [union_bases[i][fc[clusters[i].idx]].vectors * tk.factors[i][:, 1] for i in 1:N])
            push!(c_prev, tk.core[1][1])
        end

        cycle < cycles || break

        # embedded-Hamiltonian residuals, projected out of the current union
        maxres = zero(T)
        for (p, fc) in enumerate(fock_configs)
            for i in 1:N
                ci = clusters[i]
                sec = fc[ci.idx]
                sol = union_bases[i][sec]
                sol.ansatz.dim <= max_dim_H || continue
                W = sol.vectors
                size(W, 1) > size(W, 2) || continue    # union already spans the sector

                Xd = x_det[p][i]
                ints_i = subset(ints, ci.orb_list, rdm1s[p].a, rdm1s[p].b)
                Hm = build_H_matrix(ints_i, sol.ansatz)
                hx = Hm * Xd
                r = hx .- dot(Xd, hx) .* Xd
                r .-= W * (W' * r)
                nr = norm(r)
                maxres = max(maxres, nr)
                nr > res_thresh || continue

                r ./= nr
                if haskey(res_cb[i], sec)
                    old = res_cb[i][sec]
                    vecs = hcat(old.vectors, r)
                    res_cb[i][sec] = ActiveSpaceSolvers.Solution(sol.ansatz, zeros(T, size(vecs, 2)), vecs)
                else
                    res_cb[i][sec] = ActiveSpaceSolvers.Solution(sol.ansatz, zeros(T, 1), reshape(r, :, 1))
                end
                have_res = true
            end
        end
        verbose == 0 || @printf(" NO-CMF 1b cycle %2i  E(elec) = %16.10f  max residual = %9.2e\n",
                                cycle, e_hist[end], maxres)
        maxres > res_thresh || break
    end
    return e_hist, state, union_bases
end

#####################################################################
# Level 2: rank growth via repeated FockConfigs (generalized solve)
#
# The wavefunction is a list of rank-1 blocks (single-FockConfig SPTstates with
# unit cores) plus a coefficient vector. FockConfigs may repeat; same-FockConfig
# blocks are non-orthogonal, so the CI becomes a small generalized eigenproblem
# solved by canonical orthogonalization.
#####################################################################

# the single (FockConfig, TuckerConfig, Tucker) of a one-block state
function _only_block(b::SPTstate{T,N,R}) where {T,N,R}
    length(b.data) == 1 || error("expected a single-FockConfig block state")
    fc, tdict = first(b.data)
    length(tdict) == 1 || error("expected a single TuckerConfig block state")
    tc, tk = first(tdict)
    return fc, tc, tk
end

function _single_block_state(template::SPTstate{T,N,R}, fc::FockConfig{N}, tc::TuckerConfig{N},
                             tk::Tucker{T,N,R}) where {T,N,R}
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}}}()
    b = SPTstate{T,N,R}(template.clusters, data, template.p_spaces, template.q_spaces)
    add_fockconfig!(b, fc)
    b[fc][tc] = tk
    return b
end

"""
    nocmf_split_blocks(state::SPTstate{T,N,1})

Split a rank-1-block NO-CMF state into a list of normalized single-block basis
states (cores set to 1) and the corresponding coefficient vector.
"""
function nocmf_split_blocks(state::SPTstate{T,N,1}) where {T,N}
    blocks = Vector{SPTstate{T,N,1}}()
    coeffs = Vector{T}()
    for (fc, tdict) in state.data
        for (tc, tk) in tdict
            all(size(f, 2) == 1 for f in tk.factors) || error("rank-1 blocks expected")
            core = (ones(T, ntuple(i -> 1, N)),)
            push!(blocks, _single_block_state(state, fc, tc, Tucker{T,N,1}(core, deepcopy(tk.factors))))
            push!(coeffs, tk.core[1][1])
        end
    end
    return blocks, coeffs
end

"""
    nocmf_blocks_overlap(blocks)

Overlap matrix between rank-1 basis blocks. Different FockConfigs → exactly 0;
same FockConfig → product of cluster factor overlaps.
"""
function nocmf_blocks_overlap(blocks::Vector{SPTstate{T,N,1}}) where {T,N}
    nb = length(blocks)
    S = Matrix{T}(I, nb, nb)
    for a in 1:nb, b in a+1:nb
        fca, _, tka = _only_block(blocks[a])
        fcb, _, tkb = _only_block(blocks[b])
        fca == fcb || continue
        s = one(T)
        for i in 1:N
            s *= dot(tka.factors[i][:, 1], tkb.factors[i][:, 1])
        end
        S[a, b] = s
        S[b, a] = s
    end
    return S
end

"""
    nocmf_blocks_H(blocks, cluster_ops, clustered_ham; nbody=4)

Hamiltonian matrix between rank-1 basis blocks (electronic, no h0).
"""
function nocmf_blocks_H(blocks::Vector{SPTstate{T,N,1}}, cluster_ops, clustered_ham; nbody=4) where {T,N}
    nb = length(blocks)
    H = zeros(T, nb, nb)
    for a in 1:nb, b in a:nb
        H[a, b] = _block_H_element(blocks[a], blocks[b], cluster_ops, clustered_ham, nbody=nbody)
        H[b, a] = H[a, b]
    end
    return H
end

function _block_H_element(a::SPTstate{T,N,1}, b::SPTstate{T,N,1}, cluster_ops, clustered_ham; nbody=4) where {T,N}
    sig = deepcopy(a)
    zero!(sig)
    build_sigma!(sig, b, cluster_ops, clustered_ham, nbody=nbody, verbose=0)
    return get_vector(sig)[1, 1]
end

"""
    nocmf_gen_eig(H, S; lindep_thresh=1e-9)

Generalized symmetric eigenproblem by canonical orthogonalization: overlap
eigenvectors with eigenvalue below `lindep_thresh` are projected out. Returns
`(values, vectors, ndropped)` with vectors in the original (non-orthogonal)
block coordinates, normalized C'SC = 1.
"""
function nocmf_gen_eig(H::Matrix{T}, S::Matrix{T}; lindep_thresh=1e-9) where T
    F = eigen(Symmetric(S))
    keep = findall(F.values .> lindep_thresh)
    length(keep) > 0 || error("overlap matrix numerically singular")
    X = F.vectors[:, keep] * Diagonal(one(T) ./ sqrt.(F.values[keep]))
    Fh = eigen(Symmetric(X' * H * X))
    return Fh.values, X * Fh.vectors, size(S, 1) - length(keep)
end

"""
    nocmf_optimize_blocks!(blocks, coeffs, cluster_ops, clustered_ham;
                           active=eachindex(blocks), max_iter=20, tol=1e-8, ...)

Metric-corrected ALS over a list of rank-1 blocks with possibly repeated
FockConfigs. Identical to `nocmf_optimize!` except the bordered pencil's metric
gains the probe–rest overlap coupling `s_q = ⟨probe_q|rest⟩` (nonzero when other
blocks share the FockConfig of the block being updated), and the outer CI
re-solve is generalized. Updates with a near-singular metric are skipped.

`active` restricts which blocks are optimized (e.g. only a freshly added one).
Mutates `blocks` and `coeffs`; returns the per-sweep energy history.
"""
function nocmf_optimize_blocks!(blocks::Vector{SPTstate{T,N,1}}, coeffs::Vector{T},
                                cluster_ops, clustered_ham;
                                active        = eachindex(blocks),
                                max_iter      = 20,
                                tol           = 1e-8,
                                lindep_thresh = 1e-9,
                                verbose       = 0,
                                nbody         = 4) where {T,N}
    nb = length(blocks)
    nb == length(coeffs) || throw(DimensionMismatch)

    H = nocmf_blocks_H(blocks, cluster_ops, clustered_ham, nbody=nbody)
    S = nocmf_blocks_overlap(blocks)
    e, C, _ = nocmf_gen_eig(H, S, lindep_thresh=lindep_thresh)
    coeffs .= C[:, 1]

    e_hist = Vector{T}()
    push!(e_hist, e[1])
    verbose == 0 || @printf(" NO-CMF blocks ALS initial E(elec) = %16.10f\n", e_hist[end])

    for iter in 1:max_iter
        for bidx in active
            fcg, tcg, tkg = _only_block(blocks[bidx])
            others = [b2 for b2 in 1:nb if b2 != bidx]

            for j in 1:N
                d = size(tkg.factors[j], 1)

                pfactors = ntuple(i -> i == j ? Matrix{T}(I, d, d) : tkg.factors[i], N)
                pcore = (zeros(T, ntuple(i -> i == j ? d : 1, N)),)
                probe = _single_block_state(blocks[bidx], fcg, tcg, Tucker{T,N,1}(pcore, pfactors))

                A = Matrix(build_H_dense(probe, cluster_ops, clustered_ham, nbody=nbody))

                v = zeros(T, d)
                sv = zeros(T, d)
                for b2 in others
                    sig = deepcopy(probe)
                    zero!(sig)
                    build_sigma!(sig, blocks[b2], cluster_ops, clustered_ham, nbody=nbody, verbose=0)
                    v .+= coeffs[b2] .* get_vector(sig)[:, 1]

                    fcb, _, tkb = _only_block(blocks[b2])
                    fcb == fcg || continue
                    ov = one(T)
                    for i in 1:N
                        i == j && continue
                        ov *= dot(tkg.factors[i][:, 1], tkb.factors[i][:, 1])
                    end
                    sv .+= coeffs[b2] .* ov .* tkb.factors[j][:, 1]
                end
                co = coeffs[others]
                κ = dot(co, H[others, others] * co)
                ρ = dot(co, S[others, others] * co)

                if isempty(others)
                    F = eigen(Symmetric(A))
                    y = F.vectors[:, 1]
                    w = one(T)
                else
                    M = zeros(T, d + 1, d + 1)
                    M[1:d, 1:d] .= A
                    M[1:d, end] .= v
                    M[end, 1:d] .= v
                    M[end, end] = κ
                    Nm = Matrix{T}(I, d + 1, d + 1)
                    Nm[1:d, end] .= sv
                    Nm[end, 1:d] .= sv
                    Nm[end, end] = ρ
                    eigmin(Symmetric(Nm)) > lindep_thresh || continue   # near-dependent: skip

                    F = eigen(Symmetric(M), Symmetric(Nm))
                    z = F.vectors[:, 1]
                    y = z[1:d]
                    w = z[end]
                end

                ny = norm(y)
                ny > 1e-12 || continue
                newfactors = ntuple(i -> i == j ? reshape(y ./ ny, d, 1) : tkg.factors[i], N)
                tkg = Tucker{T,N,1}((ones(T, ntuple(i -> 1, N)),), newfactors)
                blocks[bidx] = _single_block_state(blocks[bidx], fcg, tcg, tkg)
                coeffs[bidx] = ny
                coeffs[others] .*= w
            end

            # refresh this block's H and S rows (off-block entries are unchanged)
            for b2 in 1:nb
                H[bidx, b2] = _block_H_element(blocks[bidx], blocks[b2], cluster_ops, clustered_ham, nbody=nbody)
                H[b2, bidx] = H[bidx, b2]
            end
            Srow = nocmf_blocks_overlap(blocks)
            S[bidx, :] .= Srow[bidx, :]
            S[:, bidx] .= Srow[:, bidx]
        end

        e, C, ndrop = nocmf_gen_eig(H, S, lindep_thresh=lindep_thresh)
        coeffs .= C[:, 1]
        push!(e_hist, e[1])
        verbose == 0 || @printf(" NO-CMF blocks ALS sweep %3i  E(elec) = %16.10f  ΔE = %9.2e  (S dropped %i)\n",
                                iter, e_hist[end], e_hist[end] - e_hist[end-1], ndrop)
        abs(e_hist[end] - e_hist[end-1]) > tol || break
    end
    return e_hist
end

"""
    nocmf_rank_growth!(blocks, coeffs, cluster_ops, clustered_ham;
                       add=nothing, max_iter=20, tol=1e-8, verbose=0, ...)

Level-2 greedy rank growth: for each FockConfig in `add` (default: each block's
FockConfig once), append a second rank-1 block in that FockConfig — initialized
from the existing block with its largest-dimension cluster factor replaced by an
orthogonal direction in the union span — then ALS-optimize the new block with
all previous blocks frozen, and finish with a global sweep. Exact in the rank
limit; near-linearly-dependent additions are projected out by the generalized
solve.

Returns the energy history across additions. Mutates `blocks`/`coeffs`.
"""
function nocmf_rank_growth!(blocks::Vector{SPTstate{T,N,1}}, coeffs::Vector{T},
                            cluster_ops, clustered_ham;
                            add           = nothing,
                            max_iter      = 20,
                            tol           = 1e-8,
                            lindep_thresh = 1e-9,
                            verbose       = 0,
                            nbody         = 4) where {T,N}
    candidates = add === nothing ? [_only_block(b)[1] for b in blocks] : add
    e_hist = Vector{T}()

    for fc in candidates
        # template: the first existing block in this FockConfig
        tidx = findfirst(b -> _only_block(b)[1] == fc, blocks)
        tidx === nothing && error("no existing block for FockConfig $fc to grow from")
        _, tc, tk = _only_block(blocks[tidx])

        # orthogonal initialization on the largest-dimension cluster
        ds = [size(tk.factors[i], 1) for i in 1:N]
        j = argmax(ds)
        if ds[j] == 1
            verbose == 0 || @printf(" nocmf_rank_growth!: no room to grow FockConfig %s — skipped\n", string(fc))
            continue
        end
        x = tk.factors[j][:, 1]
        xnew = zeros(T, ds[j])
        for q in 1:ds[j]
            eq = zeros(T, ds[j])
            eq[q] = one(T)
            xnew .= eq .- x .* dot(x, eq)
            norm(xnew) > 0.1 && break
        end
        xnew ./= norm(xnew)
        newfactors = ntuple(i -> i == j ? reshape(xnew, ds[j], 1) : deepcopy(tk.factors[i]), N)
        newblock = _single_block_state(blocks[tidx], fc, tc,
                                       Tucker{T,N,1}((ones(T, ntuple(i -> 1, N)),), newfactors))
        push!(blocks, newblock)
        push!(coeffs, zero(T))

        # optimize the new block with the others frozen, then a global polish
        eh = nocmf_optimize_blocks!(blocks, coeffs, cluster_ops, clustered_ham;
                                    active=[length(blocks)], max_iter=max_iter, tol=tol,
                                    lindep_thresh=lindep_thresh, verbose=verbose, nbody=nbody)
        append!(e_hist, eh)
        eh = nocmf_optimize_blocks!(blocks, coeffs, cluster_ops, clustered_ham;
                                    max_iter=max_iter, tol=tol,
                                    lindep_thresh=lindep_thresh, verbose=verbose, nbody=nbody)
        append!(e_hist, eh)
    end
    return e_hist
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
