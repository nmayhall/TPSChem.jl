using ..QCBase
using TensorOperations
using LinearAlgebra

"""
    subset(ints::InCoreInts, list, rmd1a, rdm1b)
Extract a subset of integrals acting on orbitals in list, returned as `InCoreInts` type
and contract a 1rdm to give effectve 1 body interaction
# Arguments
- `ints::InCoreInts`: Integrals for full system 
- `list`: list of orbital indices in subset
- `rdm1a`: 1RDM for embedding α density to make CASCI hamiltonian
- `rdm1b`: 1RDM for embedding β density to make CASCI hamiltonian
"""
function InCoreIntegrals.subset(ints::InCoreInts, ci::MOCluster, rdm1::RDM1)
    list = ci.orb_list
    ints_i = subset(ints, list)
    da = deepcopy(rdm1.a)
    db = deepcopy(rdm1.b)
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
    #h0 = compute_energy(ints, RDM1(da,db))
    return InCoreInts(0.0, ints_i.h1, ints_i.h2) 
end


function InCoreIntegrals.subset(d::RDM1, ci::MOCluster)
    return RDM1(d.a[ci.orb_list, ci.orb_list], d.b[ci.orb_list, ci.orb_list])
end

function InCoreIntegrals.subset(d::RDM2, ci::MOCluster)
    return RDM2(d.aa[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list], d.ab[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list], d.bb[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] )
end
