using Distributed
#addprocs(3; exeflags="--project")
@everywhere using TPSChem
@everywhere using PyCall
@everywhere using Printf

@everywhere pydir = joinpath(dirname(dirname(pathof(TPSChem))), "tools", "python")
@everywhere pushfirst!(PyVector(pyimport("sys")."path"), pydir)
@everywhere ENV["PYTHON"] = Sys.which("python")


#for i in 1:10
@sync @distributed for i in 1:10
    pytest = pyimport("test_pytest")
    a = pytest.test_print(i)
    @printf("%4i %4i\n",i,a)
end

