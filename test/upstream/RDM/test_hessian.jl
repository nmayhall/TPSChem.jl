using TPSChem.QCBase
using TPSChem.RDM  
using Test
using JLD2
using LinearAlgebra



@testset "RDM" begin

    @load "data_fd_hessian.jld2"
    orbital_hessian=RDM.build_orbital_hessian(ints,gd1,gd2)
    num_hess_function=num_hess1.*projection_matrix# finite difference hessian using objective function
    num_hess_gradient=num_hess2.*projection_matrix# finite difference hessian using gradient
    orbital_hessian=orbital_hessian.*projection_matrix
    
    println(" \n Analytical Hessian: ")
    display(orbital_hessian)
    eigenvalue_hessian=eigvals(orbital_hessian)
    display(eigenvalue_hessian)
    @test all(x-> x>=0.0,eigenvalue_hessian)
    println("\n Finite difference Hessian")
    display(num_hess_function)
    display(num_hess_gradient)
    @test isapprox(num_hess_gradient, orbital_hessian, atol=1e-8)
    @test isapprox(num_hess_function, orbital_hessian, atol=1e-4)
    @test isapprox(num_hess_function, num_hess_gradient, atol=1e-4)
    @test isapprox(norm(orbital_hessian-num_hess_function), 0.0, atol=1e-4)
    @test isapprox(norm(orbital_hessian-num_hess_gradient), 0.0, atol=1e-8)
    @test isapprox(norm(num_hess_function-num_hess_gradient), 0.0, atol=1e-4)
end