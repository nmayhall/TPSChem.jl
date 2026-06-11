using Printf

"""
This type contains both the Ansatz and the results.

    - ansatz::A
    - energies::Vector{T}
    - vectors::Matrix{T}

"""
struct Solution{A<:Ansatz, T<:AbstractFloat} 
    ansatz::A
    energies::Vector{T}
    vectors::Matrix{T}
end


Base.size(S::Solution) = size(S.vectors)

function Base.display(S::Solution)
    println()
    println(" Energies of Solution")
    display(S.ansatz)
    @printf(" %5s %12s\n", "State", "Energy")
    @printf("-------------------\n")
    for i in 1:length(S.energies)
        @printf(" %5i %16.12f\n", i, S.energies[i])
    end
    println()
end

Base.adjoint(S::Solution) = adjoint(S.vectors)
Base.:*(S::Solution, M) = S.vectors*M
Base.:*(M, S::Solution) = M*S.vectors
