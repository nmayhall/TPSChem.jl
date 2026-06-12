# Upstream Package Tests

These tests were copied from the standalone `nmayhall-vt` repositories that are
vendored into `TPSChem.jl`:

- `ActiveSpaceSolvers.jl`
- `BlockDavidson.jl`
- `QCBase.jl`
- `ClusterMeanField.jl`
- `RDM.jl`
- `InCoreIntegrals.jl`

The Julia imports were rewritten from standalone package imports such as
`using QCBase` to the corresponding submodule imports such as
`using TPSChem.QCBase`.

The test wrapper changes into each copied test directory before including its
`runtests.jl`, so upstream fixture paths like `h6_sto3g/h1.npy` continue to work.

`ClusterMeanField.jl` upstream tests need the optional `PyCall` extension and a
Python environment with PySCF. They are copied here but skipped by default. To
run them, set:

```bash
TPSCHEM_TEST_PYSCF=1 julia --project=. test/upstream/runtests.jl
```
