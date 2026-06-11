"""
    charge::Integer         #overall charge on molecule
    multiplicity::Integer   #2S+1
    atoms::Vector{Atom}     #Vector of `Atoms`
    basis::String           #Basis set
Molecule essentially as a Vector of atoms, number of electrons and basis set
"""
struct Molecule
    charge::Integer
    multiplicity::Integer
    atoms::Array{Atom,1}
    basis::String
end



function Base.display(mol::Molecule)
    @printf("%i\n\n",length(mol.atoms))
    for a in mol.atoms
        @printf("%4s %20.16f %20.16f %20.16f\n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3]) 
    end
end

function write_xyz(mol::Molecule; file="mol", append=true)
    xyz = @sprintf("%i\n\n",length(mol.atoms))
    for a in mol.atoms
        xyz *= @sprintf("%4s %20.16f %20.16f %20.16f\n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3])
    end
    open(file*".txt", "w") do file
        write(file, xyz)
    end
end
