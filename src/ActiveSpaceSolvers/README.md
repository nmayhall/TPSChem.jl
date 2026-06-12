# TPSChem.ActiveSpaceSolvers

*Formerly the standalone package [ActiveSpaceSolvers.jl](https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl).*

Exact diagonalization (FCI) of an active space, organized around three concepts:

1. **`Ansatz`** — metadata defining a determinant basis. The concrete subtype here is
   `FCIAnsatz(norb, n_elec_a, n_elec_b)`; the design anticipates others (RASCI, ...).
   An `Ansatz` + `InCoreInts` defines the action of H on a trial vector (a `LinearMap`).
2. **`SolverSettings`** — which eigensolver to use and its convergence options.
3. **`Solution{A,T}`** — the resulting eigenstates (energies + vectors), from which
   RDMs and the operator matrices needed by TPSChem's cluster basis are computed.

```
solve(InCoreInts + Ansatz + SolverSettings) --> Solution --> RDMs / operators
```

## Usage

```julia
using TPSChem
using TPSChem.ActiveSpaceSolvers

ints = InCoreInts(h0, h1, h2)
ansatz = FCIAnsatz(6, 3, 3)                     # 6 orbitals, 3α + 3β electrons
solver = SolverSettings(nroots=3, tol=1e-8, maxiter=100)

solution = solve(ints, ansatz, solver)
display(solution)

e = solution.energies
v = solution.vectors

rdm1a, rdm1b = compute_1rdm(solution, root=1)
d1a, d1b, d2aa, d2bb, d2ab = compute_1rdm_2rdm(solution, root=1)
s2 = compute_s2(solution)
```

The nested `FCI` module (`TPSChem.ActiveSpaceSolvers.FCI`) holds the determinant-string
machinery and the `build_H_matrix` / sigma routines behind `solve`. The
`compute_operator_*` family builds the cluster operator tensors (a†, a†a, a†a†a, ...)
that TPSChem's clustered Hamiltonian requires.

## Selected exports

`FCIAnsatz`, `SolverSettings`, `Solution`, `solve`, `compute_1rdm`,
`compute_1rdm_2rdm`, `compute_s2`, `apply_S2_matrix`, `svd_state`,
`compute_operator_c_a`, `compute_operator_ca_aa`, ... (see `ActiveSpaceSolvers.jl`)

## See also

- [`TPSChem.BlockDavidson`](../BlockDavidson/README.md) — one of the available eigensolvers
- [`TPSChem.ClusterMeanField`](../ClusterMeanField/README.md) — solves each cluster with this module
