module InCoreIntegrals

# using
using TensorOperations
using ..QCBase

include("type_InCoreInts.jl")
include("transformations.jl")
include("computations.jl")
include("subsets.jl")

# exports
export InCoreInts
export subset 

end
