using ..QCBase
using TensorOperations
using LinearAlgebra

"""
    compute_energy(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}) where T

Return energy defined by spin-summed  `rdm1` and `rdm2`.
# Arguments
- `ints`: InCoreInts object
- `d1`:   1 particle reduced density matrix
- `d2`:   2 particle reduced density matrix
"""
function QCBase.compute_energy(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}) where T
#={{{=#
    length(d1.rdm) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(d1)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * d1.rdm[p,q]
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * d2.rdm[p,q,r,s]
    end
    
    return e
end
#=}}}=#

"""
    compute_energy(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}) where T

Return energy defined by `rdm1` and `rdm2`.
# Arguments
- `ints`: InCoreInts object
- `d1`:   1 particle reduced density matrix
- `d2`:   2 particle reduced density matrix
"""
function QCBase.compute_energy(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}) where T
#={{{=#
    length(d1.a) == length(ints.h1) || throw(DimensionMismatch)
    length(d1.b) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(d1)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (d1.a[p,q] + d1.b[p,q])
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * d2.aa[p,q,r,s]
        e += .5 * ints.h2[p,q,r,s] * d2.bb[p,q,r,s]
        e +=      ints.h2[p,q,r,s] * d2.ab[p,q,r,s]
    end
    
    return e
end
#=}}}=#

"""
    compute_energy(ints::InCoreInts{T}, rdm1::RDM1{T}) where T

Return energy defined by `rdm1`.
# Arguments
- `ints`: InCoreInts object
- `rdm1`: 1 particle reduced density matrix
"""
function QCBase.compute_energy(ints::InCoreInts{T}, rdm1::RDM1{T}) where T
#={{{=#
    length(rdm1.a) == length(ints.h1) || throw(DimensionMismatch)
    length(rdm1.b) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(ints)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (rdm1.a[p,q] + rdm1.b[p,q])
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * rdm1.a[p,q] * rdm1.a[r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1.a[p,s] * rdm1.a[r,q]
        
        e += .5 * ints.h2[p,q,r,s] * rdm1.b[p,q] * rdm1.b[r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1.b[p,s] * rdm1.b[r,q]
        
        e += ints.h2[p,q,r,s] * rdm1.a[p,q] * rdm1.b[r,s]
    end
    
    return e
end
#=}}}=#


"""
    compute_fock(ints::InCoreInts, rdm1::RDM1)

Compute Fock Matrix
"""
function compute_fock(ints::InCoreInts, rdm1::RDM1)
#={{{=#
    fa = deepcopy(ints.h1)
    fb = deepcopy(ints.h1)
    @tensor begin
        #a
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.a[p,q] 
        fa[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1.a[p,q]
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.b[p,q] 
        
        #b
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.b[p,q] 
        fb[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1.b[p,q]
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.a[p,q]
        
    end
    return (fa,fb) 
end
#=}}}=#



function LinearAlgebra.tr(d::RDM1)
    return tr(d.a)+tr(d.b)
end
function LinearAlgebra.tr(d::RDM2)
    n = n_orb(d) 
    t = 0.0
    for p in 1:n
        for q in 1:n
            t += d.aa[p,p,q,q]
            t += d.ab[p,p,q,q]
            t += d.ab[q,q,p,p]
            t += d.bb[p,p,q,q]
        end
    end
    return t
end


