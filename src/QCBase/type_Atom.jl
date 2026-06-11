"""
    id::Integer             #index of atom in the molecule
    symbol::String          #Atomic ID (E.g. H, He, ...)
    xyz::Array{Float64,1}   #list of XYZ coordinates

Simply an Atom
"""
struct Atom
    id::Integer
    symbol::String
    xyz::Array{Float64,1}
end




