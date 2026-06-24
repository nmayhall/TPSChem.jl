"""
    orbital_objective_function(ints, clusters, kappa, fspace, ansatze::Vector{Ansatz}, rdm::RDM1; ...)

Objective function to minimize in OO-CMF with explicit ansatze
"""
function orbital_objective_function(ints, clusters, kappa, fspace, ansatze::Vector{<:Ansatz}, rdm::RDM1{T};
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 0) where T
    norb = n_orb(ints)
    K = unpack_gradient(kappa, norb)
    U = exp(K)
    ints_tmp = orbital_rotation(ints, U)
    e, rdm1_dict, _ = cmf_ci(ints_tmp, clusters, fspace, ansatze, orbital_rotation(rdm, U), verbose=verbose)
    return e
end

"""
    orbital_gradient_numerical(ints, clusters, kappa, fspace, ansatze, d; ...)

Compute orbital gradient with finite difference (with explicit ansatze)
"""
function orbital_gradient_numerical(ints, clusters, kappa, fspace, ansatze::Vector{<:Ansatz}, d::RDM1;
                                    ci_conv  = 1e-10,
                                    verbose  = 0,
                                    stepsize = 1e-6)
    grad = zeros(size(kappa))
    for (ii, i) in enumerate(kappa)
        k1 = deepcopy(kappa); k1[ii] += stepsize
        e1 = orbital_objective_function(ints, clusters, k1, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose)
        k2 = deepcopy(kappa); k2[ii] -= stepsize
        e2 = orbital_objective_function(ints, clusters, k2, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose)
        grad[ii] = (e1 - e2) / (2 * stepsize)
    end
    return grad
end

"""
    orbital_hessian_finite_difference(ints, clusters, kappa, fspace, d::RDM1; ...)

Compute orbital hessian with finite difference for cMF using orbital energy
"""
function orbital_hessian_finite_difference(ints, clusters, kappa, fspace, d::RDM1;
                                           ci_conv  = 1e-10,
                                           verbose  = 0,
                                           stepsize = 1e-5)
    n = length(kappa)
    hessian = zeros(n, n)
    for i in 1:n
        k0 = deepcopy(kappa)
        k1 = deepcopy(kappa); k1[i] += stepsize
        k2 = deepcopy(kappa); k2[i] -= stepsize
        e0 = orbital_objective_function(ints, clusters, k0, fspace, d, ci_conv=ci_conv, verbose=verbose)
        e1 = orbital_objective_function(ints, clusters, k1, fspace, d, ci_conv=ci_conv, verbose=verbose)
        e2 = orbital_objective_function(ints, clusters, k2, fspace, d, ci_conv=ci_conv, verbose=verbose)
        grad = (e1 - e2) / (2 * stepsize)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)
        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, grad)
        for j in (i+1):n
            kpp = deepcopy(kappa); kpp[i] += stepsize; kpp[j] += stepsize
            kpm = deepcopy(kappa); kpm[i] += stepsize; kpm[j] -= stepsize
            kmp = deepcopy(kappa); kmp[i] -= stepsize; kmp[j] += stepsize
            kmm = deepcopy(kappa); kmm[i] -= stepsize; kmm[j] -= stepsize
            hessian[i,j] = (orbital_objective_function(ints, clusters, kpp, fspace, d, ci_conv=ci_conv, verbose=verbose) -
                            orbital_objective_function(ints, clusters, kpm, fspace, d, ci_conv=ci_conv, verbose=verbose) -
                            orbital_objective_function(ints, clusters, kmp, fspace, d, ci_conv=ci_conv, verbose=verbose) +
                            orbital_objective_function(ints, clusters, kmm, fspace, d, ci_conv=ci_conv, verbose=verbose)) / (4 * stepsize^2)
            hessian[j,i] = hessian[i,j]
        end
    end
    return hessian
end

"""
    orbital_hessian_finite_difference(ints, clusters, kappa, fspace, ansatze, d::RDM1; ...)

Compute orbital hessian with finite difference for cMF using orbital energy (with explicit ansatze)
"""
function orbital_hessian_finite_difference(ints, clusters, kappa, fspace, ansatze, d::RDM1;
                                           ci_conv  = 1e-8,
                                           verbose  = 0,
                                           stepsize = 1e-5)
    n = length(kappa)
    hessian = zeros(n, n)
    for i in 1:n
        k0 = deepcopy(kappa)
        k1 = deepcopy(kappa); k1[i] += stepsize
        k2 = deepcopy(kappa); k2[i] -= stepsize
        e0 = orbital_objective_function(ints, clusters, k0, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose)
        e1 = orbital_objective_function(ints, clusters, k1, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose)
        e2 = orbital_objective_function(ints, clusters, k2, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose)
        grad = (e1 - e2) / (2 * stepsize)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)
        for j in (i+1):n
            kpp = deepcopy(kappa); kpp[i] += stepsize; kpp[j] += stepsize
            kpm = deepcopy(kappa); kpm[i] += stepsize; kpm[j] -= stepsize
            kmp = deepcopy(kappa); kmp[i] -= stepsize; kmp[j] += stepsize
            kmm = deepcopy(kappa); kmm[i] -= stepsize; kmm[j] -= stepsize
            hessian[i,j] = (orbital_objective_function(ints, clusters, kpp, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose) -
                            orbital_objective_function(ints, clusters, kpm, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose) -
                            orbital_objective_function(ints, clusters, kmp, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose) +
                            orbital_objective_function(ints, clusters, kmm, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose)) / (4 * stepsize^2)
            hessian[j,i] = hessian[i,j]
        end
    end
    return hessian
end

"""
    orbital_hessian_numerical(ints, clusters, kappa, fspace, d::RDM1; ...)

Compute orbital hessian with finite difference for cMF using orbital gradient
"""
function orbital_hessian_numerical(ints, clusters, kappa, fspace, d::RDM1;
                                   ci_conv       = 1e-10,
                                   verbose       = 0,
                                   step_size     = 1e-5,
                                   zero_intra_rots = true,
                                   maxiter_ci    = 100,
                                   maxiter_d1    = 100,
                                   tol_oo        = 1e-6,
                                   tol_d1        = 1e-7,
                                   tol_ci        = 1e-8,
                                   alpha         = .1,
                                   sequential    = false)
    n = length(kappa)
    hessian = zeros(n, n)
    function step_numerical!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        if zero_intra_rots
            for ci in clusters
                K[ci.orb_list, ci.orb_list] .= 0
            end
        end
        Ui = exp(K)
        tmp_ints = orbital_rotation(ints, Ui)
        e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, d1,
                                          maxiter_d1 = maxiter_d1,
                                          maxiter_ci = maxiter_ci,
                                          tol_d1     = tol_d1,
                                          tol_ci     = tol_ci,
                                          verbose    = 0,
                                          sequential = sequential)
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        return build_orbital_gradient(tmp_ints, gd1, gd2)
    end
    for i in 1:n
        xp = deepcopy(kappa); xp[i] += step_size
        xm = deepcopy(kappa); xm[i] -= step_size
        gp = step_numerical!(ints, d, xp)
        gm = step_numerical!(ints, d, xm)
        gnum = (gp .- gm) ./ (2 * step_size)
        for j in 1:n
            hessian[i,j] = gnum[j]
        end
    end
    return hessian
end

"""
    orbital_hessian_numerical(ints, clusters, kappa, fspace, ansatze, d::RDM1; ...)

Compute orbital hessian with finite difference for cMF using orbital gradient (with explicit ansatze)
"""
function orbital_hessian_numerical(ints, clusters, kappa, fspace, ansatze, d::RDM1;
                                   verbose       = 0,
                                   step_size     = 5e-5,
                                   zero_intra_rots = true,
                                   maxiter_ci    = 100,
                                   maxiter_d1    = 100,
                                   tol_d1        = 1e-6,
                                   tol_ci        = 1e-8,
                                   sequential    = false)
    n = length(kappa)
    hessian = zeros(n, n)
    function step_numerical!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        tmp_ints = orbital_rotation(ints, Ui)
        e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, ansatze, d1,
                                          maxiter_d1 = maxiter_d1,
                                          maxiter_ci = maxiter_ci,
                                          tol_d1     = tol_d1,
                                          tol_ci     = tol_ci,
                                          verbose    = 0,
                                          sequential = sequential)
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        return build_orbital_gradient(tmp_ints, gd1, gd2)
    end
    for i in 1:n
        println(i)
        xp = deepcopy(kappa); xp[i] += step_size
        xm = deepcopy(kappa); xm[i] -= step_size
        gp = step_numerical!(ints, d, xp)
        gm = step_numerical!(ints, d, xm)
        gnum = (gp .- gm) ./ (2 * step_size)
        for j in 1:n
            hessian[i,j] = gnum[j]
        end
    end
    return hessian
end

"""
    orbital_hessian_fd_fci_solve(ints, n_elec_a, n_elec_b, kappa; ...)

Compute orbital hessian with finite difference for FCI using ActiveSpaceSolvers energies
"""
function orbital_hessian_fd_fci_solve(ints, n_elec_a, n_elec_b, kappa, verbose=0, stepsize=1e-4)
    n = length(kappa)
    hessian = zeros(n, n)
    function step!(ints, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        tmp_ints = orbital_rotation(ints, Ui)
        ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)
        solver = SolverSettings(nroots=3, tol=1e-6, maxiter=100)
        solution = solve(tmp_ints, ansatz, solver)
        return solution.energies
    end
    for i in 1:n
        k0 = deepcopy(kappa)
        k1 = deepcopy(kappa); k1[i] += stepsize
        k2 = deepcopy(kappa); k2[i] -= stepsize
        e0 = step!(ints, k0); e1 = step!(ints, k1); e2 = step!(ints, k2)
        hessian[i,i] = (e1[1] - 2*e0[1] + e2[1]) / (stepsize^2)
        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0[1], e1[1], e2[1], (e1[1]-e2[1])/(2*stepsize))
        for j in (i+1):n
            kpp = deepcopy(kappa); kpp[i] += stepsize; kpp[j] += stepsize
            kpm = deepcopy(kappa); kpm[i] += stepsize; kpm[j] -= stepsize
            kmp = deepcopy(kappa); kmp[i] -= stepsize; kmp[j] += stepsize
            kmm = deepcopy(kappa); kmm[i] -= stepsize; kmm[j] -= stepsize
            hessian[i,j] = (step!(ints,kpp)[1] - step!(ints,kmp)[1] - step!(ints,kpm)[1] + step!(ints,kmm)[1]) / (4*stepsize^2)
            hessian[j,i] = hessian[i,j]
        end
    end
    return hessian
end

"""
    orbital_hessian_fd_fci_rdm(ints, n_elec_a, n_elec_b, kappa; ...)

Compute orbital hessian with finite difference for FCI using RDM energy
"""
function orbital_hessian_fd_fci_rdm(ints, n_elec_a, n_elec_b, kappa, verbose=0, stepsize=1e-4)
    n = length(kappa)
    hessian = zeros(n, n)
    function step!(ints, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        tmp_ints = orbital_rotation(ints, Ui)
        ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)
        solver = SolverSettings(nroots=3, tol=1e-6, maxiter=100)
        solution = solve(tmp_ints, ansatz, solver)
        rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab = ActiveSpaceSolvers.compute_1rdm_2rdm(solution, root=1)
        d1 = RDM1(rdm1a, rdm1b)
        d2 = RDM2(rdm2aa, rdm2ab, rdm2bb)
        return compute_energy(ints, d1, d2)
    end
    for i in 1:n
        k0 = deepcopy(kappa)
        k1 = deepcopy(kappa); k1[i] += stepsize
        k2 = deepcopy(kappa); k2[i] -= stepsize
        e0 = step!(ints, k0); e1 = step!(ints, k1); e2 = step!(ints, k2)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)
        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, (e1-e2)/(2*stepsize))
        for j in (i+1):n
            kpp = deepcopy(kappa); kpp[i] += stepsize; kpp[j] += stepsize
            kpm = deepcopy(kappa); kpm[i] += stepsize; kpm[j] -= stepsize
            kmp = deepcopy(kappa); kmp[i] -= stepsize; kmp[j] += stepsize
            kmm = deepcopy(kappa); kmm[i] -= stepsize; kmm[j] -= stepsize
            hessian[i,j] = (step!(ints,kpp) - step!(ints,kmp) - step!(ints,kpm) + step!(ints,kmm)) / (4*stepsize^2)
            hessian[j,i] = hessian[i,j]
        end
    end
    return hessian
end

"""
    orbital_hessian_fd_cmf_rdm(ints, clusters, fspace, d1::RDM1, kappa; ...)

Compute orbital hessian with finite difference for cMF using RDM energy
"""
function orbital_hessian_fd_cmf_rdm(ints::InCoreInts{T}, clusters, fspace, d1::RDM1{T}, kappa,
                                     zero_intra_rots=true, verbose=0, stepsize=1e-5) where T
    n = length(kappa)
    hessian = zeros(n, n)
    function step!(ints, k, d1)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        if zero_intra_rots
            for ci in clusters
                K[ci.orb_list, ci.orb_list] .= 0
            end
        end
        Ui = exp(K)
        tmp_ints = orbital_rotation(ints, Ui)
        e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, d1,
                                          maxiter_d1 = 100, maxiter_ci = 100,
                                          tol_d1 = 1e-9, tol_ci = 1e-10,
                                          verbose = 0, sequential = true)
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        return compute_energy(tmp_ints, gd1, gd2)
    end
    for i in 1:n
        k0 = deepcopy(kappa)
        k1 = deepcopy(kappa); k1[i] += stepsize
        k2 = deepcopy(kappa); k2[i] -= stepsize
        e0 = step!(ints, k0, d1); e1 = step!(ints, k1, d1); e2 = step!(ints, k2, d1)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)
        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, (e1-e2)/(2*stepsize))
        for j in (i+1):n
            kpp = deepcopy(kappa); kpp[i] += stepsize; kpp[j] += stepsize
            kpm = deepcopy(kappa); kpm[i] += stepsize; kpm[j] -= stepsize
            kmp = deepcopy(kappa); kmp[i] -= stepsize; kmp[j] += stepsize
            kmm = deepcopy(kappa); kmm[i] -= stepsize; kmm[j] -= stepsize
            hessian[i,j] = (step!(ints,kpp,d1) - step!(ints,kmp,d1) - step!(ints,kpm,d1) + step!(ints,kmm,d1)) / (4*stepsize^2)
            hessian[j,i] = hessian[i,j]
        end
    end
    return hessian
end

"""
    create_interaction_vector(clusters, total_elements)

Compute the projection vector for redundant (intra-cluster) rotations.
"""
function create_interaction_vector(clusters, total_elements)
    blocks = [collect(t[1]:t[end]) for t in clusters]
    matrix = ones(total_elements, total_elements)
    for cluster in blocks
        for i in cluster, j in cluster
            matrix[i,j] = 0
        end
    end
    kout = zeros(total_elements*(total_elements-1)÷2)
    ind = 1
    for i in 1:total_elements
        for j in i+1:total_elements
            kout[ind] = matrix[i,j]
            ind += 1
        end
    end
    return kout
end

"""
    get_global_pair(local_pair_vector)

Compute the global pairs for which energy is invariant to orbital rotation.
"""
function get_global_pair(local_pair_vector)
    global_pair_vector = []
    offset = 0
    for sub_vector in local_pair_vector
        desired_sub_vector = [(x + offset, y + offset) for (i, (x, y)) in enumerate(sub_vector)]
        push!(global_pair_vector, desired_sub_vector)
        offset = maximum(desired_sub_vector[lastindex(desired_sub_vector)])
    end
    return global_pair_vector
end

"""
    create_projection_vector(local_pair_vector, total_elements)

Compute the projection vector for redundant rotations.
"""
function create_projection_vector(local_pair_vector, total_elements)
    global_pair_vector = get_global_pair(local_pair_vector)
    matrix = ones(total_elements, total_elements)
    for sub_vector in global_pair_vector
        for (i, (x, y)) in enumerate(sub_vector)
            matrix[x,y] = 0
        end
    end
    kout = zeros(total_elements*(total_elements-1)÷2)
    ind = 1
    for i in 1:total_elements
        for j in i+1:total_elements
            kout[ind] = matrix[i,j]
            ind += 1
        end
    end
    return kout
end

"""
    create_projection_matrix(vector)

Compute the projection matrix from a projection vector (outer product).
"""
function create_projection_matrix(vector)
    vector = reshape(vector, (length(vector), 1))
    return vector * transpose(vector)
end

"""
    get_rdm(ints, d1, k, clusters, fspace)

Compute the total RDM of the clusters at rotated orbitals.
"""
function get_rdm(ints, d1, k, clusters, fspace)
    norb = n_orb(ints)
    K = unpack_gradient(k, norb)
    for ci in clusters
        K[ci.orb_list, ci.orb_list] .= 0
    end
    Ui = exp(K)
    rdm1 = orbital_rotation(d1, Ui)
    tmp_ints = orbital_rotation(ints, Ui)
    e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, rdm1,
                                      maxiter_d1 = 100, maxiter_ci = 100,
                                      tol_d1 = 1e-9, tol_ci = 1e-10,
                                      verbose = 0, sequential = true)
    gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
    return gd1, gd2
end
