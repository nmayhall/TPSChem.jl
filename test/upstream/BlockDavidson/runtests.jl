using TPSChem.BlockDavidson
using Test

@testset "BlockDavidson.jl" begin
    include("test01.jl")
    include("test_precond_01.jl")
end
