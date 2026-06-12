using TPSChem.RDM
using Test

@testset "RDM.jl" begin
    include("test_rdm.jl")
    include("test_hessian.jl")
end
