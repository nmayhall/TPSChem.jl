using TPSChem
using Test

const UPSTREAM_TEST_ROOT = @__DIR__

function include_upstream_testset(package::AbstractString)
    pkgdir = joinpath(UPSTREAM_TEST_ROOT, package)
    cd(pkgdir) do
        include(joinpath(pkgdir, "runtests.jl"))
    end
end

@testset "Upstream standalone package tests" begin
    include_upstream_testset("QCBase")
    include_upstream_testset("InCoreIntegrals")
    include_upstream_testset("BlockDavidson")
    include_upstream_testset("RDM")
    include_upstream_testset("ActiveSpaceSolvers")

    if get(ENV, "TPSCHEM_TEST_PYSCF", "0") == "1"
        @info "Running PySCF-dependent ClusterMeanField upstream tests"
        include_upstream_testset("ClusterMeanField")
    else
        @testset "ClusterMeanField.jl upstream tests" begin
            @info "Skipping PySCF-dependent ClusterMeanField tests. Set TPSCHEM_TEST_PYSCF=1 to run them."
            @test true
        end
    end
end
