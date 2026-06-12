# NO-CMF: Non-orthogonal CMF state interaction — design document

*Status: implemented in `src/core/nocmf.jl` (tests in `test/test_nocmf.jl`). Entry
points: `nocmf_level0`, `nocmf_routeA`, `nocmf_level1a` (`nocmf_optimize!`),
`nocmf_level1b`, and level-2 rank growth via `nocmf_split_blocks` +
`nocmf_rank_growth!` (`nocmf_optimize_blocks!`, `nocmf_gen_eig`). Union-basis
scaffolding: `build_union_basis` in `src/core/type_ClusterBasis.jl`.*

## 1. Motivation

TPSCI builds one orthonormal cluster basis per cluster and expands in product configurations. The cluster states are computed in a *single* mean-field environment (one FockConfig), so describing charge-transfer or locally-excited FockConfigs well requires many cluster states — the basis is not adapted to each Fock sector's environment.

NO-CMF instead solves a **separate CMF-CI for each FockConfig** in a user-chosen list, so every FockConfig gets cluster states polarized by *its own* environment, and couples a few tensor product states (TPS) per FockConfig in a small CI problem. The aim is a compact, interpretable wavefunction — (number of FockConfigs) × (a few states) rather than a product space — analogous to NOCI over cMF states, but with shared orthonormal orbitals.

## 2. Definitions and ansatz

- Clusters i = 1…N partition a fixed, orthonormal MO space (from localization or a preliminary CMF). **No orbital optimization anywhere in this method** — this is a hard design constraint (see §3).
- A FockConfig f assigns (nα, nβ) to each cluster. The user supplies a list F (neutral, CT, spin-flipped, locally ionized, ...).
- For each f in F: converge CMF-CI at fixed f (cluster FCI in the mean field of the other clusters' RDMs, iterated to self-consistency; no orbital rotation). Keep `m_i^f` eigenstates `|s_i^f⟩` per cluster.

**Level-0 ansatz** (fixed cluster states):

$$
|\Psi\rangle = \sum_{f \in F} \sum_{s \in S_f} c_{f,s} \: |\Phi_{f,s}\rangle ,
\qquad
|\Phi_{f,s}\rangle = \bigotimes_i |s_i^f\rangle
$$

where `S_f` is a small set of selected product states (cMF ground product + selected local excitations).

## 3. Orthogonality structure

Because all FockConfigs share one orthonormal MO basis:

- f ≠ f′: the configs differ in (nα, nβ) on at least one cluster, whose states are then orthogonal by particle number, so `⟨Φ_{f,s}|Φ_{f',s'}⟩ = 0` **exactly**.
- f = f′: the TPS are products of one CMF solution's orthonormal cluster eigenstates, hence orthonormal.

**Therefore S = 1 globally and the level-0/1 CI is a standard (not generalized) eigenproblem.** The "non-orthogonality" lives entirely inside Hamiltonian matrix elements: cluster states with the same sector but different cMF parents overlap nontrivially, so *spectator clusters contribute overlap matrices instead of Kronecker deltas*.

This breaks only when the same FockConfig appears more than once in the expansion (level 2, §6) — then S ≠ 1 within that block and the small eigenproblem becomes generalized. It also breaks if orbitals are ever relaxed per FockConfig (non-orthogonal orbitals require NOCI-grade matrix-element machinery); we exclude that by design.

## 4. Matrix elements via a union working basis

A clustered operator term has the form `T̂ = γ · ô_{i1} ⊗ ô_{i2} ⋯`. Acting between bra block f and ket block f′, its matrix element factors into a contraction over the term's *active* clusters times overlap factors on every *spectator* cluster j:

$$
\langle \Phi_f | \hat T | \Phi_{f'} \rangle
= (\text{active-cluster contraction}) \times \prod_j \sigma_j^{ff'} ,
\qquad
\sigma_j^{ff'} = \langle a_j^f | a_j^{f'} \rangle
$$

The active-cluster factors need transition densities **between two different cluster bases**, e.g. `⟨a_i^f|p̂†|a_i^{f'}⟩`. Rather than computing determinant-level TDMs for every pair of cMF parents, use a shared coordinate system:

1. **Union working basis.** Per cluster i, per Fock sector: stack the kept states from *all* parents, orthonormalize with an **SVD and singular-value threshold τ** (not plain QR — parents that barely repolarize a cluster give near-duplicate states; near-null directions must be dropped, with the discarded norm reported). Result: a working basis of dimension d, at most the total number of stacked states.
2. **Operators once.** Compute standard cluster operator tensors O, with elements `⟨w_m|ô|w_n⟩`, in the working basis with the existing machinery. Done once, shared by all parents.
3. **Factors.** Each parent's states are exact coefficient columns `U_i^f` (a `d × m_i^f` matrix) in the working basis. All inter-basis quantities are sandwiches:

$$
\langle a^f | \hat o | a^{f'} \rangle = U^{f\dagger} O \: U^{f'} ,
\qquad
\sigma_i^{ff'} = U_i^{f\dagger} U_i^{f'}
$$

The spectator overlaps fall out of the same structure (the identity operator transforms to `U^{f†} U^{f'}`). **The working basis is scaffolding only — it never enters the variational space.** Results must be invariant to its construction details (a useful correctness test).

Each TPS block is exactly a Tucker block: core = CI coefficients over kept local states, factors = `U_i^f`. This is the existing `SPTstate` structure with one block per FockConfig, whose factors come from that FockConfig's cMF instead of an HOSVD. Blocks in distinct FockConfigs are mutually orthogonal, so the existing orthogonal SPT solve applies at level 0.

**Can the SPT machinery really be reused with non-orthogonal cross-block factors?** Verified in the source (a natural worry, since Tucker factors are "orthonormal" by construction):

- The H build does **not** assume cross-block factor overlaps are δ. `contract_dense_H_with_state` explicitly forms spectator overlaps `S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]` and applies them to the ket core (`src/core/tucker_contract_dense_H_with_state.jl:89-105`, comment: *"needed when TuckerConfigs aren't the same because each does their own compression and has distinct Tucker factors"*). Active clusters are sandwiched `U' O U` in `build_dense_H_term` (`src/core/tucker_build_dense_H_term.jl:14,34-48`). This is exactly the §4 sandwich structure.
- What **is** assumed, and holds for NO-CMF: (i) orthonormal *columns* within each block's factors — true, each cMF's kept states are orthonormal vectors in the working basis — used by `orth_dot` norms; (ii) spectator clusters must have matching `TuckerConfig` ranges (`check_term`, `src/core/type_ClusteredTerm.jl:100-105`) — satisfied by giving every block the full union-sector range.
- Caution: compression (HOSVD) and PT2 code paths may carry stronger orthonormality assumptions and must be audited before reuse; the level-0 build/solve path (`build_sigma!`, `form_sigma_block!`) does not.

## 5. Level 1: resonating CMF (variational factor optimization)

Adding more fixed states per FockConfig converges slowly: the parent cMF states are optimal for their own diagonal block but know nothing about coupling. The efficient improvement is to **optimize the cluster states defining each TPS in the presence of the others** — the cluster analog of resonating HF (Fukutome) / few-determinant projected-HF chains (Scuseria). Take one TPS per FockConfig, `|Φ_f⟩ = ⊗_i |x_i^f⟩`, and minimize the lowest NOCI root over all factors.

**Multilinearity gives ALS structure.** Every `H_{ff'}` is linear in each factor separately, so optimize one factor at a time (cf. one-site DMRG). For the update of `x = x_j^g` (cluster j, FockConfig g), with all other factors fixed:

- `H_gg = x† A x`, where A is the **embedded cluster Hamiltonian** for cluster j in block g — the same operator CMF-CI already builds from the other clusters' densities;
- `H_fg = w_f† x` for f ≠ g, where the **resonance vector** `w_f` is the half-projected sigma vector: apply the clustered Hamiltonian to block g with cluster j's slot open, contract all other clusters against bra `Φ_f` (transition densities + spectator overlaps);
- normalization contributes `x† x` on the g diagonal only.

Absorbing the CI coefficient into `y = c_g x`, the energy is a ratio of quadratics in y:

$$
E(y) = \frac{ y^\dagger A y + 2 v^\dagger y + \kappa }{ y^\dagger y + \rho }
$$

$$
v = \sum_{f \ne g} c_f w_f ,
\qquad
\kappa = \sum_{f \ne g} \sum_{f' \ne g} c_f c_{f'} H_{ff'} ,
\qquad
\rho = \sum_{f \ne g} c_f^2
$$

Its minimization is the lowest eigenpair of the bordered (d+1)-dimensional symmetric pencil with matrix rows `(A, v)` and `(v†, κ)` and metric `diag(1, ρ)` — each update relaxes the factor and its CI weight simultaneously. Sweep over all (cluster, FockConfig) pairs; an outer loop re-solves the full NOCI vector c; iterate to convergence. This strictly generalizes the existing `cmf_ci` iteration: with a single FockConfig the resonance terms vanish and it *is* CMF-CI.

**Where does x live?** Two flavors:

- **1a — subspace relaxation:** restrict x to the span of the union working basis. Everything stays in cheap sandwiched form; A and `w_f` are small. Limited by the union span.
- **1b — full relaxation:** after converging 1a, compute the unconstrained residual/gradient direction in the cluster's determinant space (needs half-transformed operators `⟨m|ô|a^{f'}⟩` with m a determinant — same TDM machinery, cost comparable to the cluster FCI solves cmf_ci already performs), **augment the union basis with it**, recompute ops in the enlarged basis, repeat. Krylov-like; converges to the unconstrained optimum while keeping operators compact.

**Structural gift:** factors in different FockConfigs stay exactly orthogonal *no matter how they are optimized* (particle number). The classic resonating-HF pathologies — states collapsing onto each other, singular overlap — cannot occur across FockConfigs; S = 1 survives the whole optimization.

**Initial guess:** the level-0 fixed cMF states. Levels 0 and 1 share one code path; level 0 is iteration zero of level 1.

## 6. Level 2: systematic convergence

Even optimized, one TPS per FockConfig carries zero inter-cluster entanglement *within* its block. Two complementary knobs:

1. **Rank growth via repeated FockConfigs (CP-style).** Add a second, independently optimized TPS in the same FockConfig — greedily: optimize the new TPS in the presence of the frozen previous ones (the resonating-chain / FED recipe), then optionally a global re-sweep. Exact in the rank limit. Within a repeated FockConfig S ≠ 1: the (small) NOCI eigenproblem becomes generalized; monitor its condition number and discard rank additions that approach linear dependence.
2. **PT2 diagnostics and correction.** Epstein–Nesbet PT2 over the union product space (the Route-A space of §7) on top of the converged reference: estimates the residual error, and its per-FockConfig / per-sector decomposition tells you *which* FockConfig to add to the list or where to grow rank next. Reuses TPSCI screening/PT2 logic.

**Spin.** Fixed (nα, nβ) per cluster breaks Ŝ². Mitigations, in order of rigor: (i) always include each FockConfig's spin-flip partners in the list (`possible_spin_focksectors` exists) and keep cluster multiplets intact when truncating; (ii) at level 1, tie/state-average factors across spin-partner FockConfigs; (iii) check ⟨Ŝ²⟩ of NOCI roots as a diagnostic. **Excited states:** state-average the level-1 objective over the lowest few NOCI roots.

## 7. Validation strategy (Route A benchmark)

A nearly-free upper-level benchmark using only existing machinery: per-FockConfig `cmf_ci`, then union ClusterBasis (`merge_cluster_bases(augment=true)`, upgraded QR to SVD), then **standard TPSCI over the full product space of the union basis** restricted to the listed FockConfigs.

Route A's variational space contains every NO-CMF state **whose factors lie in the span of the union basis it was built from**. That covers level 0 and level 1a (subspace relaxation), giving the rigorous chain

$$
E(\mathrm{FCI}) \le E(\mathrm{Route\ A}) \le E(\mathrm{level\ 1a}) \le E(\mathrm{level\ 0})
$$

(with matching TPS counts per FockConfig between levels 1a and 0). **Level 1b is not bounded by the initial Route A**: full relaxation moves the factors outside the original union span, so the only guarantees are variational ones, E(FCI) ≤ E(level 1b) ≤ E(level 1a). To restore a contemporaneous benchmark, rebuild the union basis from the *converged* level-1b factors and rerun TPSCI in it — the gap between that re-built Route A and level 1b then doubles as a convergence diagnostic (it measures the intra-block entanglement that rank growth, §6, would capture).

- every level must respect the (corrected) chain above;
- a single-FockConfig list must reproduce the plain CMF-CI energy exactly;
- results must be invariant to the union-basis orthonormalization (re-run with shuffled stacking order / different τ well below the discard scale);
- small test system (e.g., H₆–H₈, 3–4 clusters) compared against FCI.

## 8. Mapping onto TPSChem.jl

Existing pieces (reuse as-is or near):

| Piece | Where |
|---|---|
| Per-FockConfig CMF-CI (fixed orbitals) | `cmf_ci`, `src/ClusterMeanField/incore_cmf.jl:224` |
| Union of cluster bases | `merge_cluster_bases(augment=true)`, `src/core/type_ClusterBasis.jl:90` — **upgrade QR to thresholded SVD** |
| Cluster ops in working basis | `compute_cluster_ops`, `src/core/build_local_quantities.jl` |
| Per-block factor structure, nonorth contractions | `SPTstate` / Tucker machinery, `src/core/type_SPTstate.jl`, `src/core/hosvd.jl` |
| Spin-partner FockConfig generation | `possible_spin_focksectors`, `src/core/type_FockConfig.jl` |
| PT2 / screening patterns | TPSCI code, `src/core/tpsci_*.jl` |

New pieces (all in `src/core/nocmf.jl` unless noted):

| Function | Role |
|---|---|
| `nocmf_cmf_solutions` | per-FockConfig CMF-CI + parent cluster eigenbases + embedding RDMs |
| `build_union_basis` (`type_ClusterBasis.jl`) | SVD-thresholded union working basis + exact parent factors |
| `nocmf_state` | SPTstate with one Tucker block per FockConfig (`nkeep` truncates block rank) |
| `nocmf_level0`, `nocmf_ci_solve`, `build_H_dense` | level-0 dense solve |
| `nocmf_routeA` | union-product-space TPSCI benchmark |
| `nocmf_optimize!`, `nocmf_level1a` | ALS sweeps via the bordered (d+1) pencil |
| `nocmf_level1b` | residual-augmented union cycles (embedded-H residual direction) |
| `nocmf_split_blocks`, `nocmf_optimize_blocks!`, `nocmf_rank_growth!`, `nocmf_gen_eig` | level-2 rank growth, metric-corrected ALS, canonical-orthogonalization generalized solve |

Validated behavior (test/test_nocmf.jl, h8 + h12 fixtures): single-FockConfig
limit reproduces CMF-CI to machine precision; the bound chain FCI ≤ RouteA ≤
lvl1a ≤ lvl0 ≤ CMF holds; level-0 energies are invariant to union stacking
order; ALS is monotone; on h8 (union of 2 states/sector) rank-2 growth
saturates the union product space and lands on Route A exactly.

## 9. Open questions

- Selection of the per-FockConfig TPS set at level 0: lowest-energy products vs. perturbative-coupling screening against the reference.
- Automating the FockConfig list (CT generator? screen by PT2 estimate from a cheap level-0 pass?).
- State-averaging weights and root-flipping protection during ALS.
- Whether level-1b's basis augmentation should be per-cluster-adaptive or global per sweep.
