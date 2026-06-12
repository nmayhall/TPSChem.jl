using TPSChem.BlockDavidson
using Random
using LinearAlgebra
using Test
using LinearMaps

@testset "BlockDavidson" begin
    Random.seed!(2)

    A = rand(500,500) .- .5;
    A += A';

    #F = eigen(A);
    #display(F.values)

    dav = Davidson(A)
    e,v = BlockDavidson.eigs(dav)
    e_ref = -18.260037795157675
    @test isapprox(e[1], -18.260037795157675)

    lmap = LinearMap(A)
    
    dav = Davidson(lmap)
    e,v = BlockDavidson.eigs(dav)
    e_ref = -18.260037795157675
    @test isapprox(e[1], -18.260037795157675)

    e_ref = [
             -18.260037795157675
             -17.9716411644818
             -17.47598854674256
             -17.184105784827796
             -16.939563610543257
             -16.83937885452674
            ]
    # now with more settings specified and roots
    dav = Davidson(lmap; max_iter=200, max_ss_vecs=8, tol=1e-6, nroots=6)
    e,v = BlockDavidson.eigs(dav)
    @test all(isapprox.(e, e_ref, atol=1e-10))

    display(v'*Matrix(lmap*v))
    # now test with initial guess
    e,v = BlockDavidson.eigs(Davidson(lmap; max_iter=2, max_ss_vecs=8, tol=1e-8, nroots=6, v0=v))
    @test all(isapprox.(e, e_ref, atol=1e-10))


    # # Test Complex
    println(" Now testing complex")
    flush(stdout)
    ndim = 100
    nvec = 6
    # types = [Float32, Float64, ComplexF32, ComplexF64]
    types = [Float64, ComplexF64]
    for T in types
        A = rand(T, ndim, ndim) .- 0.5
        A = A + A'

        e_ref, v_ref = eigen(A)
        e_ref = e_ref[1:nvec]


        mymatvec(v) = A * v
        
        lmap = LinearMap(mymatvec, ndim, ndim; ismutating=false, ishermitian=true)
        # display(lmap * rand(T,ndim, nvec))
        dav = Davidson(lmap; max_iter=200, max_ss_vecs=8, tol=1e-6, nroots=nvec, T=T)
        e, v = BlockDavidson.eigs(dav)
        @test all(isapprox.(e, e_ref, atol=1e-10))
    end

end
