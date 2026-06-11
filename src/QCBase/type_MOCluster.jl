

"""
    idx::Int16
    orb_list::Vector{Int16}
"""
struct MOCluster
    idx::Int16
    orb_list::Vector{Int16}
end


function MOCluster(ci::Integer, c::UnitRange{<:Integer})
    return MOCluster(ci, collect(c))
end


"""
	length(c::MOCluster)

Return number of orbitals in `MOCluster`
"""
function Base.length(c::MOCluster)
    return length(c.orb_list)
end
"""
	dim_tot(c::MOCluster)

Return dimension of hilbert space spanned by number of orbitals in `MOCluster`. 
This is all sectors
"""
function dim_tot(c::MOCluster)
    return 2^(2*length(c))
end
"""
	dim_tot(c::MOCluster, na, nb)

Return dimension of hilbert space spanned by number of orbitals in `MOCluster`
with `na` and `nb` number of alpha/beta electrons.
"""
function dim_tot(c::MOCluster, na, nb)
    nc = length(c)
    T = eltype(nc)
    return binomial(nc, T(na))*binomial(nc, T(nb)) 
end
function Base.display(cv::Vector{MOCluster}) where N
    for c in cv
        display(c)
    end
end
function Base.display(c::MOCluster)
    @printf("IDX%03i:DIM%04i:" ,c.idx,dim_tot(c))
    for si in c.orb_list
        @printf("%03i|", si)
    end
    @printf("\n")
end
function Base.isless(ci::MOCluster, cj::MOCluster)
    return Base.isless(ci.idx, cj.idx)
end
function Base.isequal(ci::MOCluster, cj::MOCluster)
    return Base.isequal(ci.idx, cj.idx)
end
######################################################################################################

# possible_focksectors should go into TPSChem

"""
    possible_focksectors(c::MOCluster, delta_elec=nothing)
        
Get list of possible fock spaces accessible to the MOCluster

- `delta_elec::Vector{Int}`: (nα, nβ, Δ) restricts fock spaces to: (nα,nβ) ± Δ electron transitions
"""
function possible_focksectors(c::MOCluster; delta_elec::Tuple=())
    ref_a = nothing
    ref_b = nothing
    delta = nothing
    if length(delta_elec) != 0
        length(delta_elec) == 3 || throw(DimensionMismatch)
        ref_a = delta_elec[1]
        ref_b = delta_elec[2]
        delta = delta_elec[3]
    end

    no = length(c)
   
    fsectors::Vector{Tuple} = []
    for na in 0:no
        for nb in 0:no 
            if length(delta_elec) != 0
                if abs(na-ref_a)+abs(nb-ref_b) > delta
                    continue
                end
            end
            push!(fsectors,(na,nb))
        end
    end
    return fsectors
end





#function check_orthogonality(mat; thresh=1e-12)
#    Id = mat' * mat
#    if maximum(abs.(I-Id)) > thresh 
#        @warn("problem with orthogonality ", maximum(abs.(I-Id)))
#        return false
#    end
#    return true
#end

