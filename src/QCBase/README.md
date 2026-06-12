# TPSChem.QCBase

*Formerly the standalone package [QCBase.jl](https://github.com/nmayhall-vt/QCBase.jl).*

Basic quantum chemistry data types shared by the rest of TPSChem: molecules, atoms,
and orbital clusters, plus a handful of generic functions (`compute_energy`,
`orbital_rotation`, ...) that the other submodules extend with their own methods.

## Key types

| Type | Description |
|---|---|
| `Atom` | atomic number, symbol, and xyz coordinates |
| `Molecule` | charge, multiplicity, list of `Atom`s, basis-set string |
| `MOCluster` | an indexed set of molecular-orbital indices defining one cluster |

## Usage

These names are re-exported by `TPSChem`, so `using TPSChem` is usually all you need:

```julia
using TPSChem

atoms = [Atom(1, "H", [0.0, 0.0, 0.0]),
         Atom(2, "H", [0.0, 0.0, 1.0])]
mol = Molecule(0, 1, atoms, "sto-3g")

# partition orbitals 1:8 into two 4-orbital clusters
clusters = [MOCluster(1, [1, 2, 3, 4]),
            MOCluster(2, [5, 6, 7, 8])]
n_orb(clusters[1])   # 4
```

The module can also be used on its own with `using TPSChem.QCBase`.

## Exported names

`Atom`, `Molecule`, `MOCluster`, `n_orb`, `dim_tot`, `write_xyz`,
`compute_energy`, `orbital_rotation`, `possible_focksectors`

## See also

- [`TPSChem.InCoreIntegrals`](../InCoreIntegrals/README.md) — integrals over these orbitals
- [`TPSChem.ClusterMeanField`](../ClusterMeanField/README.md) — CMF over `MOCluster` partitions
