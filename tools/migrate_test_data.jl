# One-time migration of JLD2 test data from the pre-consolidation module layout
# (FermiCG, QCBase, RDM, ... as separate root packages) to TPSChem submodules.
#
# Usage: julia --project=. tools/migrate_test_data.jl test/*.jld2

using TPSChem
using JLD2

const TYPEMAP = Dict(
    "QCBase.MOCluster"                 => TPSChem.QCBase.MOCluster,
    "InCoreIntegrals.InCoreInts"       => TPSChem.InCoreIntegrals.InCoreInts,
    "RDM.RDM1"                         => TPSChem.RDM.RDM1,
    "RDM.RDM2"                         => TPSChem.RDM.RDM2,
    "ActiveSpaceSolvers.Solution"      => TPSChem.ActiveSpaceSolvers.Solution,
    "ActiveSpaceSolvers.FCI.FCIAnsatz" => TPSChem.ActiveSpaceSolvers.FCI.FCIAnsatz,
    "FermiCG.ClusterBasis"             => TPSChem.ClusterBasis,
    "FermiCG.ClusterOps"               => TPSChem.ClusterOps,
    "FermiCG.ClusteredOperator"        => TPSChem.ClusteredOperator,
    "FermiCG.ClusteredTerm"            => TPSChem.ClusteredTerm,
    "FermiCG.ClusteredTerm1B"          => TPSChem.ClusteredTerm1B,
    "FermiCG.ClusteredTerm2B"          => TPSChem.ClusteredTerm2B,
    "FermiCG.ClusteredTerm3B"          => TPSChem.ClusteredTerm3B,
    "FermiCG.ClusteredTerm4B"          => TPSChem.ClusteredTerm4B,
    "FermiCG.TransferConfig"           => TPSChem.TransferConfig,
)

# JLD2's Dict-based typemap does not apply stored type parameters; use the
# function form so e.g. RDM1 becomes RDM1{Float64}.
function typemap(f, typepath, params)
    if haskey(TYPEMAP, typepath)
        t = TYPEMAP[typepath]
        isempty(params) && return t
        try
            return t{params...}
        catch
            return JLD2.UnknownType{t, Tuple{params...}}
        end
    end
    return JLD2.default_typemap(f, typepath, params)
end

function check_clean(key, val)
    s = string(typeof(val))
    if occursin("Reconstructed", s)
        error("$key still contains reconstructed types: $s")
    end
end

for fn in ARGS
    println("migrating $fn")
    data = Dict{String,Any}()
    jldopen(fn, "r"; typemap=typemap) do f
        for k in keys(f)
            data[k] = f[k]
        end
    end
    for (k, v) in data
        check_clean(k, v)
    end
    tmp = fn * ".new"
    jldopen(tmp, "w") do f
        for (k, v) in data
            f[k] = v
        end
    end
    mv(tmp, fn, force=true)
end
println("done")
