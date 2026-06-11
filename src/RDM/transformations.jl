
function QCBase.orbital_rotation(d::Union{RDM1, RDM2, ssRDM1, ssRDM2}, U)
    d2 = deepcopy(d)
    orbital_rotation!(d2,U)
    return d2
end

function orbital_rotation!(d2::RDM2{T}, U) where T
    @tensor begin
        d[p,q,r,s] := U[t,p] * d2.aa[t,q,r,s]
        d[p,q,r,s] := U[t,q] * d[p,t,r,s]
        d[p,q,r,s] := U[t,r] * d[p,q,t,s]
        d[p,q,r,s] := U[t,s] * d[p,q,r,t]
    end
    d2.aa .= d
    @tensor begin
        d[p,q,r,s] := U[t,p] * d2.ab[t,q,r,s]
        d[p,q,r,s] := U[t,q] * d[p,t,r,s]
        d[p,q,r,s] := U[t,r] * d[p,q,t,s]
        d[p,q,r,s] := U[t,s] * d[p,q,r,t]
    end
    d2.ab .= d
    @tensor begin
        d[p,q,r,s] := U[t,p] * d2.bb[t,q,r,s]
        d[p,q,r,s] := U[t,q] * d[p,t,r,s]
        d[p,q,r,s] := U[t,r] * d[p,q,t,s]
        d[p,q,r,s] := U[t,s] * d[p,q,r,t]
    end
    d2.bb .= d
end
function orbital_rotation!(d2::ssRDM2{T}, U) where T
    @tensor begin
        d[p,q,r,s] := U[t,p] * d2.rdm[t,q,r,s]
        d[p,q,r,s] := U[t,q] * d[p,t,r,s]
        d[p,q,r,s] := U[t,r] * d[p,q,t,s]
        d[p,q,r,s] := U[t,s] * d[p,q,r,t]
    end
    d2.rdm .= d
end
function orbital_rotation!(d::ssRDM1{T}, U) where T
    tmp = U'*d.rdm*U
    #@tensor begin
    #    d[p,q] := U[t,p] * d1.rdm[t,q]
    #    d[p,q] := U[t,q] * d[p,t]
    #end
    #d1.rdm .= d
    d.rdm .= tmp
end
function orbital_rotation!(d1::RDM1{T}, U) where T
    #d.a .= U'*d.a*U
    #d.b .= U'*d.b*U
    @tensor begin
        da[p,q] := U[t,p] * d1.a[t,q]
        db[p,q] := U[t,p] * d1.b[t,q]
        da[p,q] := U[t,q] * da[p,t]
        db[p,q] := U[t,q] * db[p,t]
    end
    d1.a .= da
    d1.b .= db
end
