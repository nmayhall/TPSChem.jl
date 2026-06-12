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
    build_generalised_fock(ints, rdm1::ssRDM1{T}, rdm2::ssRDM2{T})

Build fock-like term f_{pq} = h_{pq} γ_{pq} + V_{r,p,s,t} Γ_{r,q,s,t}

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM, NxN
- `rdm2`: Spin summed 2RDM, NxNxNxN
"""
function build_generalised_fock(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}; verbose=0) where T
    verbose == 0 || println(" In build_generalised_fock")
    N = n_orb(ints)
    f = ints.h1 * d1.rdm
    V = ints.h2
    @tensor begin
        f[p,q] += V[p,r,s,t] * d2.rdm[q,r,s,t]
    end
    return f
end


"""
    build_orbital_hessian(ints, rdm1::ssRDM1{T}, rdm2::ssRDM2{T})

Build the full orbital rotation hessian
H_{pq,rs} = <[[H,p'q-q'p], r's-s'r]>

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM
- `rdm2`: Spin summed 2RDM
"""
function build_orbital_hessian(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}; verbose=0) where T
    verbose == 0 || println(" In build_orbital_hessian")
    N = n_orb(ints)
    A = zeros(N,N,N,N)
    Y = zeros(N,N,N,N)
    H = zeros(N,N,N,N)
    F = build_generalised_fock(ints, d1, d2)
    d_1 = d1.rdm
    d_2 = d2.rdm
    V = ints.h2
    h_1 = ints.h1
    I = Diagonal(ones(size(F, 1)))
    @tensor begin
        Y[p,q,r,s]  = d_2[p, m, r, n] * V[q, m, n, s]
        Y[p,q,r,s] += d_2[p, m, n, r] * V[q, m, n, s]
        Y[p,q,r,s] += d_2[p, r, m, n] * V[q, s, m, n]

        A[p,q,r,s]  = 2 * d_1[p,r] * h_1[q,s]
        A[p,q,r,s] -= (F[p,r] + F[r,p]) * I[q,s]
        A[p,q,r,s] += 2 * Y[p,q,r,s]

        H[p,q,r,s]  = A[p,q,r,s]
        H[p,q,r,s] -= A[p,q,s,r]
        H[p,q,r,s] -= A[q,p,r,s]
        H[p,q,r,s] += A[q,p,s,r]
    end
    return pack_hessian(H, N)
end


"""
    build_generalised_fock(ints, rdm1::RDM1{T}, rdm2::RDM2{T})

Build fock-like term for cmf (non spin-summed)
f_{pq} = h_{pq}*(d1.a+d1.b) + V_{p,q,r,s}*Γ_{t,q,r,s}

# Arguments
- `ints`: Integrals
- `rdm1`: Not spin-summed 1RDM
- `rdm2`: Not spin-summed 2RDM
"""
function build_generalised_fock(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}; verbose=0) where T
    verbose == 0 || println(" In build_generalised_fock_cmf")
    N = n_orb(ints)
    f = ints.h1 * (d1.a + d1.b)
    V = ints.h2
    @tensor begin
        f[p,t] +=  V[p,q,r,s] * d2.aa[t,q,r,s]
        f[p,t] +=  V[p,q,r,s] * d2.bb[t,q,r,s]
        f[p,t] +=  V[p,q,r,s] * d2.ab[t,q,r,s]
        f[p,t] +=  V[p,q,r,s] * d2.ab[r,s,t,q]
    end
    return f
end


"""
    build_orbital_hessian(ints, rdm1::RDM1{T}, rdm2::RDM2{T})

Build the full orbital rotation hessian
H_{pq,rs} = <[[H,p'q-q'p], r's-s'r]> (non spin-summed)

# Arguments
- `ints`: Integrals
- `rdm1`: Not spin-summed 1RDM
- `rdm2`: Not spin-summed 2RDM
"""
function build_orbital_hessian(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}; verbose=0) where T
    verbose == 0 || println(" In build_orbital_hessian")
    N = n_orb(ints)
    A = zeros(N,N,N,N)
    Y = zeros(N,N,N,N)
    d_2 = zeros(N,N,N,N)
    H = zeros(N,N,N,N)
    F = build_generalised_fock(ints, d1, d2)
    d_1 = d1.a + d1.b
    V = ints.h2
    h = ints.h1
    I = Diagonal(ones(size(F, 1)))
    for p in 1:N, q in 1:N, r in 1:N, s in 1:N
        d_2[p,q,r,s] = d2.aa[p,q,r,s] + d2.bb[p,q,r,s] + d2.ab[p,q,r,s] + d2.ab[r,s,p,q]
    end
    @tensor begin
        Y[p,q,r,s]  =  d_2[p, m, r, n] * V[q, m, n, s]
        Y[p,q,r,s] +=  d_2[p, m, n, r] * V[q, m, n, s]
        Y[p,q,r,s] +=  d_2[p, r, m, n] * V[q, s, m, n]
    end
    for p in 1:N, q in 1:N, r in 1:N, s in 1:N
        Y[r,s,p,q] = Y[p,q,r,s]
    end
    @tensor begin
        A[p,q,r,s]  = 2 * d_1[p,r] * h[q,s]
        A[p,q,r,s] -= F[p,r] * I[q,s]
        A[p,q,r,s] -= F[r,p] * I[q,s]
        A[p,q,r,s] += 2 * Y[p,q,r,s]
    end
    for p in 1:N, q in 1:N, r in 1:N, s in 1:N
        H[p,q,r,s] = A[p,q,r,s] - A[p,q,s,r] - A[q,p,r,s] + A[q,p,s,r]
    end
    return pack_hessian(H, N)
end


"""
    pack_hessian(H, norb)

Pack the rank-4 hessian tensor to a symmetric matrix of size n*(n-1)/2 × n*(n-1)/2.
"""
function pack_hessian(H, norb)
    size(H) == (norb, norb, norb, norb) || throw(DimensionMismatch)
    hout = zeros(norb*(norb-1)÷2, norb*(norb-1)÷2)
    ind_row = 1
    for i in 1:norb
        for j in i+1:norb
            ind_col = 1
            for k in 1:norb
                for l in k+1:norb
                    hout[ind_row, ind_col] = H[i, j, k, l]
                    hout[ind_col, ind_row] = hout[ind_row, ind_col]
                    ind_col += 1
                end
            end
            ind_row += 1
        end
    end
    return hout
end


"""
    unpack_hessian(h, norb)

Unpack a packed hessian matrix back to a rank-4 tensor.
"""
function unpack_hessian(h, norb)
    length(h) == (norb*(norb-1)÷2)^2 || throw(DimensionMismatch)
    H = zeros(norb, norb, norb, norb)
    ind_row = 1
    for i in 1:norb
        for j in i+1:norb
            ind_col = 1
            for k in 1:norb
                for l in k+1:norb
                    H[i, j, k, l] = h[ind_row, ind_col]
                    H[j, i, l, k] = H[i, j, k, l]
                    ind_col += 1
                end
            end
            ind_row += 1
        end
    end
    return H
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



