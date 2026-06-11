using ..InCoreIntegrals


"""
    build_orbital_gradient(ints, rdm1::ssRDM1{T}, rdm2::ssRDM2{T})

Build the full orbital rotation hessian
g_{pq} = <[H,p'q-q'p]> for p<q

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM, NxN
- `rdm2`: Spin summed 2RDM, NxNxNxN
"""
function build_orbital_gradient(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}; verbose=0) where T
    verbose == 0 || println(" In build_orbital_gradient")
    
    N = n_orb(ints)
    G = zeros(N,N)
    g = zeros(N*(N-1)÷2)
   
    G = ints.h1*d1.rdm 
    V = ints.h2

    @tensor begin
        G[p,q] +=  V[r,p,s,t] * d2.rdm[r,q,s,t]
    end
    return pack_gradient(2*(G-G'),N)
end


"""
    build_orbital_gradient(ints, rdm1::D1, d2::RDM2)

Build the full orbital rotation hessian
g_{pq} = <[H,p'q-q'p]> for p<q

# Arguments
- `ints`: Integrals
- `d1`: Not spin-summed 
- `d2`: Not spin-summed 
"""
function build_orbital_gradient(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}; verbose=0) where T
    verbose == 0 || println(" In build_orbital_gradient")
    
    N = n_orb(ints)
    G = zeros(N,N)
    g = zeros(N*(N-1)÷2)
   
    G = ints.h1*(d1.a + d1.b)
    V = ints.h2

    @tensor begin
        G[s,t] +=  V[p,q,r,s] * d2.aa[p,q,r,t]
        G[s,t] +=  V[p,q,r,s] * d2.bb[p,q,r,t]
        G[s,t] +=  V[p,q,r,s] * d2.ab[p,q,r,t]
        G[s,t] +=  V[p,q,r,s] * d2.ab[r,t,p,q]
    end
    return pack_gradient(2*(G-G'),N)
end

"""
    build_orbital_hessian(ints, rdm1, rdm2)

Build the full orbital rotation hessian
H_{pq,rs} = <[[H,p'q-q'p], r's-s'r]>

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM
- `rdm2`: Spin summed 2RDM
"""
function build_orbital_hessian(ints::InCoreInts, rdm1, rdm2; verbose=0)
    verbose == 0 || println(" In build_orbital_hessian")
end


"""
    unpack_gradient(kappa,norb)

Unpack the orbital rotation parameter to full matrix 
"""
function unpack_gradient(kappa,norb)
    length(kappa) == norb*(norb-1)÷2 || throw(DimensionMismatch)
    K = zeros(norb,norb)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            K[i,j] = kappa[ind]
            K[j,i] = -kappa[ind]
            ind += 1
        end
    end
    return K
end


"""
    pack_gradient(K,norb)

Pack the orbital rotation parameter to upper-triangle
"""
function pack_gradient(K,norb)
    length(K) == norb*norb || throw(DimensionMismatch)
    kout = zeros(norb*(norb-1)÷2)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            kout[ind] = K[i,j]
            ind += 1
        end
    end
    return kout
end



