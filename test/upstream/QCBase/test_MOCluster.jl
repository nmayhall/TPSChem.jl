using TPSChem.QCBase 
using Test

@testset "MOCluster" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    basis = "6-31g"

    mol     = Molecule(0, 1, atoms, basis)

    display(mol)
    QCBase.write_xyz(mol)
    
    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters1 = [MOCluster(i, collect(clusters[i])) for i = 1:length(clusters)]
    clusters2 = [MOCluster(i, clusters[i]) for i = 1:length(clusters)]


    display(length(clusters1[1]))
    @test length(clusters1[1]) == 2

    @test dim_tot(clusters1[1]) == 16

    for f in QCBase.possible_focksectors(clusters1[1])
        display(f)
    end
end
