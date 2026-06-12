# TPSChem.RDM

*Formerly the standalone package [RDM.jl](https://github.com/nmayhall-vt/RDM.jl).*

Reduced density matrix types and operations: spin-resolved and spin-summed 1- and
2-RDMs, energy contractions, and orbital-rotation gradients (used by the CMF orbital
optimizer).

## Key types

| Type | Description |
|---|---|
| `RDM1{T}` | 1-RDM with separate α (`.a`) and β (`.b`) blocks |
| `RDM2{T}` | 2-RDM with `.aa`, `.ab`, `.bb` spin blocks |
| `ssRDM1{T}`, `ssRDM2{T}` | spin-summed variants |

## Usage

```julia
using TPSChem

no = n_orb(ints)
d1 = RDM1(no)            # zero-initialized no×no α and β blocks
d1 = RDM1(da, db)        # from explicit α/β matrices
d2 = RDM2(d1)            # mean-field (Slater determinant) 2-RDM from a 1-RDM

# energy of a state given its RDMs
E = compute_energy(ints, d1, d2)

# gradient of the energy w.r.t. orbital rotations (for CMF-OO)
g = TPSChem.RDM.build_orbital_gradient(ints, d1, d2)
```

`RDM1`/`RDM2` are re-exported by `TPSChem`; the less common functions are accessed as
`TPSChem.RDM.<name>` or via `using TPSChem.RDM`.

## Exported names

`RDM1`, `RDM2`, `ssRDM1`, `ssRDM2`, `build_orbital_gradient`

## See also

- [`TPSChem.ClusterMeanField`](../ClusterMeanField/README.md) — consumes/produces these RDMs
- `compute_1rdm`, `compute_2rdm` in the TPSChem core for RDMs of TPSCI states
