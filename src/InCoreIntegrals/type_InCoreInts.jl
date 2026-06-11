
"""
    -h0::Real                # constant energy shift
    -h1::Array{T,2}          # one electron integrals
    -h2::Array{T,4}          # two electron integrals (chemist's notation)

Type to hold a second quantized Hamiltonian coefficients in memory
"""
struct InCoreInts{T}
    h0::T
    h1::Array{T,2}
    h2::Array{T,4}
end

function InCoreInts(ints::InCoreInts{TT}, T::Type) where {TT}
    return InCoreInts{T}(T(ints.h0), Array{T,2}(ints.h1), Array{T,4}(ints.h2))
end

function QCBase.n_orb(ints::InCoreInts)
    return size(ints.h1,1)
end





