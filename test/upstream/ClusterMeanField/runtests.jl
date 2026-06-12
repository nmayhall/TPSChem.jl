using PyCall
using TPSChem.ClusterMeanField
using Test

@testset "ClusterMeanField.jl" begin
    include("test_cmf.jl")
    include("test_savg.jl")
    include("test_gd.jl")
    include("test_hessian.jl")
end
