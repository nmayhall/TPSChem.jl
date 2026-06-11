module QCBase

using Printf
using StaticArrays


include("type_Atom.jl")
include("type_Molecule.jl")
include("type_MOCluster.jl")


export Atom
export Molecule
export MOCluster

n_orb() = nothing
dim_tot() = nothing
write_xyz() = nothing
compute_energy() = nothing
orbital_rotation() = nothing


export n_orb
export dim_tot
export write_xyz
export compute_energy 
export orbital_rotation 
export possible_focksectors

end
