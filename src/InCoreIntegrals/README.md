# TPSChem.InCoreIntegrals

*Formerly the standalone package [InCoreIntegrals.jl](https://github.com/nmayhall-vt/InCoreIntegrals.jl).*

Container for the 1- and 2-electron integrals that appear in quantum chemistry
calculations. Aimed at active-space models (≲ 200 orbitals), so everything is held
in memory.

## The `InCoreInts{T}` type

| Field | Meaning |
|---|---|
| `h0` | scalar constant (nuclear repulsion, frozen-core energy, ...) |
| `h1` | matrix $h_{pq}$ for the 1-body operator $h_{pq}\,\hat a_p^\dagger \hat a_q$ |
| `h2` | 4-index tensor $g_{pqrs}$ for the 2-body operator $g_{pqrs}\,\hat a_p^\dagger \hat a_q^\dagger \hat a_s \hat a_r$ |

## Usage

```julia
using TPSChem
using NPZ

ints = InCoreInts(npzread("h0.npy"), npzread("h1.npy"), npzread("h2.npy"))

# energy for given 1- and 2-RDMs:  E = h0 + Σ h_pq D_pq + ½ Σ g_pqrs Γ_pqrs
E = compute_energy(ints, rdm1, rdm2)

# rotate to a new orbital basis
ints2 = TPSChem.orbital_rotation(ints, U)

# extract the integrals for a subset of orbitals (e.g. one cluster)
ints_i = TPSChem.InCoreIntegrals.subset(ints, cluster.orb_list)
```

`subset` also has methods that embed a cluster in the mean field of the others
(used heavily by [`ClusterMeanField`](../ClusterMeanField/README.md)).

## Exported names

`InCoreInts`, `subset` (plus methods extending `QCBase.compute_energy`,
`QCBase.orbital_rotation`, `QCBase.n_orb`)
