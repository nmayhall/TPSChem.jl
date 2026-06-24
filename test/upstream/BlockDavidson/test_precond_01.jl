using TPSChem.BlockDavidson
using LinearAlgebra
using Random
using Test

@testset "BlockDavidson preconditioning" begin
    Random.seed!(2)

    N = 1_000
    nr = 4
    A = -10 .* diagm(rand(N)) + 0.001 * rand(N, N)
    A += A'

    v0 = qr(rand(N, 2)).Q[:, 1:nr]

    idx = sortperm(diag(A))
    v0 = Matrix(1.0I, N, N)[:, idx[1:nr]]


    println(" No preconditioning")
    dav1 = Davidson(A, max_iter=50, nroots=nr, tol=1e-8, v0=v0)
    dav2 = Davidson(A, max_iter=50, nroots=nr, tol=1e-8, v0=v0)
    e1, v1 = BlockDavidson.eigs(dav1)
    println()
    println(" Preconditioned")
    e2, v2 = BlockDavidson.eigs(dav2, Adiag=diag(A))
   
    flush(stdout)
    display(dav1.converged)
    display(dav2.converged)
    @test dav1.converged == false
    @test dav2.converged == true

end