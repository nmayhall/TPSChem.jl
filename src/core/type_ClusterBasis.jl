using .ActiveSpaceSolvers

"""
    cluster::MOCluster                            # Cluster to which basis belongs
    basis::Dict{Tuple,Matrix{T}}                # Basis vectors (nα, nβ)=>[I,s]

These basis coefficients map local slater determinants to local vectors
`(nα, nβ): 
V[αstring*βstring, cluster_state]`
"""
struct ClusterBasis{A,T}
    cluster::MOCluster
    basis::Dict{Tuple{Int16,Int16},Solution{A,T}}
end
ClusterBasis(ci::MOCluster; T::Type = Float64, A=FCIAnsatz) = ClusterBasis(ci, Dict{Tuple{Int16,Int16},Solution{A,T}}())

"""
    ClusterBasis(cb::ClusterBasis, T::Type)

Convert from one data type to another
"""
function ClusterBasis(cb::ClusterBasis{A,TT}, T::Type) where {A,TT}
    out = ClusterBasis{A,T}(cb.cluster, Dict{Tuple{Int16,Int16},Matrix{T}}())
    for (fspace,basis) in cb.basis
        out.basis[fspace] = Solution(basis.ansatz, Vector{T}(basis.energies), Matrix{T}(basis.vectors))
    end
    return out
end

Base.iterate(cb::ClusterBasis, state=1) = iterate(cb.basis, state)
Base.length(cb::ClusterBasis) = length(cb.basis)
Base.getindex(cb::ClusterBasis,i) = cb.basis[i] 
Base.setindex!(cb::ClusterBasis,val,key) = cb.basis[key] = val
Base.haskey(cb::ClusterBasis,key) = haskey(cb.basis, key)
function Base.display(cb::ClusterBasis) 
    @printf(" ClusterBasis for Cluster: %4i\n",cb.cluster.idx)
    norb = length(cb.cluster)
    sum_total_dim = 0
    sum_dim = 0
    
    T = eltype(norb)
    for (sector, vecs) in cb.basis
        na = T(sector[1])
        nb = T(sector[2])
        dim = size(vecs,2)
        total_dim = binomial(norb,na) * binomial(norb,nb) 
        sum_dim += dim
        sum_total_dim += total_dim
        
        @printf("   FockSector = (%2iα, %2iβ): Total Dim = %5i: Dim = %4i\n", sector[1],sector[2],total_dim, dim)
    end
       
    @printf("   -----------------------------\n")
    @printf("   Total Dim = %5i: Dim = %4i\n", sum_total_dim, sum_dim)
end



"""
    rotate!(cb::ClusterBasis, U::Dict{Tuple,Matrix{T}}) where {T} 

Rotate `cb` by unitary matrices in `U`
"""
function rotate!(cb::ClusterBasis,U::Dict{Tuple,Matrix{T}}) where {T} 
#={{{=#
    for (fspace,mat) in U
        cb[fspace].vectors .= cb[fspace] * mat
    end
end
#=}}}=#

    
function check_basis_orthogonality(basis::ClusterBasis; thresh=1e-12)
    for (fspace,mat) in basis
        if check_orthogonality(mat,thresh=thresh) == false
            println(" Cluster:", basis.cluster)
            println(" Fockspace:", fspace)
        end
    end
end

using LinearAlgebra

"""
    svd_orthonormalize(V::Matrix{T}; svd_thresh=1e-8)

Orthonormalize the columns of `V` by SVD, discarding directions with singular
value below `svd_thresh` (near-duplicate columns). Returns `(W, discarded)`
where `W` has orthonormal columns spanning `V` to within the threshold and
`discarded` is the vector of dropped singular values.
"""
function svd_orthonormalize(V::Matrix{T}; svd_thresh=1e-8) where T
    F = svd(V)
    nkeep = sum(F.S .> svd_thresh)
    return F.U[:, 1:nkeep], F.S[nkeep+1:end]
end

"""
Merge two Vector{ClusterBasis}: for each cluster, union the Fock sectors.
If a sector exists in both, keep the one with more roots (columns in vectors).
If you want to combine states from both (e.g. augment ground with CT states),
pass augment=true — this hcats vectors and re-orthogonalizes. By default the
combination uses a thresholded SVD (`svd_thresh`), dropping near-linearly-
dependent directions; pass svd_thresh=nothing to use plain QR (keeps all
columns, including numerical junk from near-duplicates).
"""
function merge_cluster_bases(cb1::Vector{T}, cb2::Vector{T}; augment=false, svd_thresh=1e-8) where T
    @assert length(cb1) == length(cb2)
    result = Vector{T}(undef, length(cb1))
    for i in eachindex(cb1)
        new_cb = TPSChem.ClusterBasis(cb1[i].cluster)
        # seed with all sectors from cb1
        for (fspace, sol) in cb1[i]
            new_cb[fspace] = sol
        end
        # merge sectors from cb2
        for (fspace, sol2) in cb2[i]
            if !haskey(new_cb, fspace)
                new_cb[fspace] = sol2
            elseif augment
                sol1 = new_cb[fspace]
                # concatenate vectors and re-orthogonalize
                combined = hcat(sol1.vectors, sol2.vectors)
                if svd_thresh === nothing
                    Q, _ = qr(combined)
                    n = size(combined, 2)
                    vecs = Matrix(Q)[:, 1:n]
                else
                    vecs, discarded = svd_orthonormalize(combined, svd_thresh=svd_thresh)
                    if length(discarded) > 0
                        @printf(" merge_cluster_bases: cluster %i sector (%i,%i) dropped %i directions, max sv %.1e\n",
                                cb1[i].cluster.idx, fspace[1], fspace[2], length(discarded), maximum(discarded))
                    end
                end
                n = size(vecs, 2)
                energies = vcat(sol1.energies, sol2.energies)[1:n]
                new_cb[fspace] = ActiveSpaceSolvers.Solution(sol1.ansatz, energies, vecs)
            else
                # keep whichever has more roots
                if size(sol2.vectors, 2) > size(new_cb[fspace].vectors, 2)
                    new_cb[fspace] = sol2
                end
            end
        end
        result[i] = new_cb
    end
    return result
end

"""
    build_union_basis(parent_bases::Vector{Vector{ClusterBasis{A,T}}}; svd_thresh=1e-8, verbose=0)

Build, per cluster and per Fock sector, an orthonormal "working" basis spanning
the cluster states of all parents (e.g. one parent per FockConfig-specific cMF
solution), together with the factor matrices expressing each parent's states in
the working basis. The working basis is scaffolding for computing ClusterOps
once; the parents' states remain the physical objects.

# Arguments
- `parent_bases`: outer index = parent, inner index = cluster
- `svd_thresh`: singular-value threshold for discarding near-duplicate directions

# Returns
- `union_bases::Vector{ClusterBasis{A,T}}`: per-cluster union basis. The
  `Solution.energies` of union sectors are placeholders (zeros) — the working
  basis has no eigenbasis meaning.
- `factors::Vector{Vector{Dict{Tuple{Int16,Int16},Matrix{T}}}}`:
  `factors[p][i][sector]` is the `d × m` matrix whose columns are parent `p`'s
  states on cluster `i` in `sector`, expressed in the union basis. Exact up to
  the discarded SVD directions; the worst reconstruction error is printed when
  `verbose > 0` and returned as the third value.
"""
function build_union_basis(parent_bases::Vector{Vector{ClusterBasis{A,T}}}; svd_thresh=1e-8, verbose=0) where {A,T}
    nparents = length(parent_bases)
    nparents > 0 || error("no parent bases given")
    nclusters = length(parent_bases[1])
    all(length(pb) == nclusters for pb in parent_bases) || error("inconsistent cluster counts")

    union_bases = Vector{ClusterBasis{A,T}}(undef, nclusters)
    factors = [[Dict{Tuple{Int16,Int16},Matrix{T}}() for i in 1:nclusters] for p in 1:nparents]
    max_recon_err = 0.0

    for i in 1:nclusters
        cluster = parent_bases[1][i].cluster
        all(pb[i].cluster.idx == cluster.idx for pb in parent_bases) || error("inconsistent cluster ordering")
        union_cb = ClusterBasis(cluster, T=T)

        # collect sectors over all parents
        sectors = Set{Tuple{Int16,Int16}}()
        for pb in parent_bases
            for (sec, _) in pb[i]
                push!(sectors, sec)
            end
        end

        for sec in sectors
            # stack all parents' vectors in this sector
            stacked = Vector{Matrix{T}}()
            for pb in parent_bases
                haskey(pb[i], sec) || continue
                push!(stacked, pb[i][sec].vectors)
            end
            V = hcat(stacked...)
            W, discarded = svd_orthonormalize(V, svd_thresh=svd_thresh)
            if verbose > 0 && length(discarded) > 0
                @printf(" build_union_basis: cluster %i sector (%i,%i) dropped %i of %i directions, max sv %.1e\n",
                        cluster.idx, sec[1], sec[2], length(discarded), size(V,2), maximum(discarded))
            end
            d = size(W, 2)

            ansatz = nothing
            for pb in parent_bases
                haskey(pb[i], sec) || continue
                ansatz === nothing || break
                ansatz = pb[i][sec].ansatz
            end
            union_cb[sec] = ActiveSpaceSolvers.Solution(ansatz, zeros(T, d), W)

            # exact factors: parent states as columns in the working basis
            for p in 1:nparents
                haskey(parent_bases[p][i], sec) || continue
                Vp = parent_bases[p][i][sec].vectors
                U = W' * Vp
                err = norm(Vp - W * U)
                max_recon_err = max(max_recon_err, err)
                factors[p][i][sec] = U
            end
        end
        union_bases[i] = union_cb
    end
    verbose == 0 || @printf(" build_union_basis: max factor reconstruction error %.2e\n", max_recon_err)
    return union_bases, factors, max_recon_err
end
