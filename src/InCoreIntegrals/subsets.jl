using ..QCBase

"""
    subset(ints::InCoreInts, list)

Extract a subset of integrals acting on orbitals in list, returned as `InCoreInts` type.
Because a constant is necessarily a global quantity, we return an `InCoreInts` object
with h0 = 0.0. 

# Arguments
- `ints::InCoreInts`: Integrals for full system 
- `list`: list of orbital indices in subset
"""
function subset(ints::InCoreInts{T}, list) where {T}
    return InCoreInts{T}(0.0, view(ints.h1,list,list), view(ints.h2,list,list,list,list))
end

function InCoreIntegrals.subset(ints::InCoreInts, ci::MOCluster)
    return subset(ints, ci.orb_list) 
end


"""
    subset(ints::InCoreInts, list; rmd1a, rdm1b)

Extract a subset of integrals acting on orbitals in list, returned as `InCoreInts` type
and contract a 1rdm to give effectve 1 body interaction

# Arguments
- `ints::InCoreInts`: Integrals for full system 
- `list`: list of orbital indices in subset
- `rdm1a`: 1RDM for embedding α density to make CASCI hamiltonian
- `rdm1b`: 1RDM for embedding β density to make CASCI hamiltonian
"""
function subset(ints::InCoreInts{T}, list, rdm1a, rdm1b) where {T}
    ints_i = subset(ints, list)

    da = deepcopy(rdm1a)
    db = deepcopy(rdm1b)
    da[:,list] .= 0
    db[:,list] .= 0
    da[list,:] .= 0
    db[list,:] .= 0
    viirs = ints.h2[list, list,:,:]
    viqri = ints.h2[list, :, :, list]
    f = zeros(length(list),length(list))
    @tensor begin
        f[p,q] += viirs[p,q,r,s] * (da+db)[r,s]
        f[p,s] -= .5*viqri[p,q,r,s] * da[q,r]
        f[p,s] -= .5*viqri[p,q,r,s] * db[q,r]
    end
    ints_i.h1 .+= f
    
    return ints_i
end


