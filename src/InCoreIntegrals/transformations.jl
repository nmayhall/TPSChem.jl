
#"""
#    orbital_rotation!(ints::InCoreInts, U)
#
#Transform electronic integrals, by U
#i.e.,
#```math
#h_{pq} = U_{rp}h_{rs}U_{sq}
#```
#```math
#(pq|rs) = (tu|vw)U_{tp}U_{uq}U_{vr}U_{ws}
#```
#"""
#function orbital_rotation!(ints::InCoreInts{T}, U::Matrix{T}) where T
#    ints.h1 .= U' * ints.h1 * U
#    scr = Array{T}(undef, size(ints.h2)...)
#    @tensor begin
#        scr[p,q,r,s] = U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
#    end
#    ints.h2 .= scr
#end

@doc raw"""
    orbital_rotation(ints::InCoreInts, U)

Transform electronic integrals, by U
i.e.,
```math
h_{pq} = U_{rp}h_{rs}U_{sq}
```
```math
(pq|rs) = (tu|vw)U_{tp}U_{uq}U_{vr}U_{ws}
```
"""
function QCBase.orbital_rotation(ints::InCoreInts, U)
    @tensor begin
        h1[p,q] := U[r,p]*U[s,q]*ints.h1[r,s]
        # h2[p,q,r,s] := U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
        h2[p,q,r,s] := U[t,p]*ints.h2[t,q,r,s]
        h2[p,q,r,s] := U[t,q]*h2[p,t,r,s]
        h2[p,q,r,s] := U[t,r]*h2[p,q,t,s]
        h2[p,q,r,s] := U[t,s]*h2[p,q,r,t]
    end
    return InCoreInts(ints.h0,h1,h2)
end
