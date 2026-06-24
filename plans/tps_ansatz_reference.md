# Context-optimized tensor product state ansatz

A reference for the ansatz and its two optimization levels.

## What distinguishes this from standard TPSCI

In standard TPSCI each cluster carries a single basis of local states per Fock
sector, built once (typically from one cMF or a local diagonalization) and
shared across every FockConfig. A spectator cluster in a fixed sector therefore
contributes a Kronecker delta to any matrix element, and the cluster states form
one orthonormal set per sector.

The ansatz here drops the "shared per sector" assumption. The local states of a
cluster are allowed to be context-specific: they depend on the entire
FockConfig, not just on the cluster's own sector. Two FockConfigs that put the
same cluster in the same sector can still give that cluster two different local
CI vectors, because the surrounding clusters differ. Those two vectors share one
determinant basis but are not orthogonal. This cross-FockConfig non-orthogonality
of spectator clusters is the new structural feature, and it is a deliberate
consequence of letting each cluster state be optimal for its context rather than
an artifact to be removed.

## The ansatz

$$
|\Psi\rangle \;=\; \sum_{F\in\mathcal{F}}\;\sum_{\mathbf{p}\in\mathcal{P}(F)}
C_{F\mathbf{p}}\,|F,\mathbf{p}\rangle,
\qquad
|F,\mathbf{p}\rangle \;=\; \bigotimes_{I=1}^{N_c}|\Phi^I_{p_I}(F)\rangle .
$$

### Index sets

- **FockConfig** $F=(f_1,\dots,f_{N_c})$, one local Fock sector
  $f_I=(N^I_\alpha,N^I_\beta)$ per cluster. The retained set is restricted to the
  target global sector,
  $$
  \mathcal{F}\subseteq\Big\{F:\textstyle\sum_I N^I_\alpha=N_\alpha,\ \sum_I N^I_\beta=N_\beta\Big\}.
  $$
- **ClusterConfig** $\mathbf{p}=(p_1,\dots,p_{N_c})$, selecting one retained local
  CI vector per cluster. The per-cluster range depends on $F$,
  $$
  \mathcal{P}(F)=\bigotimes_{I=1}^{N_c}\{1,\dots,m_I(F)\},
  $$
  with $m_I(F)$ the number of local states kept in cluster $I$ given $F$. With
  one state per cluster per sector, $m_I(F)=1$, $\mathbf{p}$ collapses, and
  $|\Psi\rangle=\sum_F C_F|F\rangle$.
- A single flat global index is convenient downstream:
  $$
  \mathcal{B}=\{(F,\mathbf{p}):F\in\mathcal{F},\ \mathbf{p}\in\mathcal{P}(F)\},
  \qquad |\Psi\rangle=\sum_{B\in\mathcal{B}}C_B|B\rangle .
  $$

### Local cluster states

$$
|\Phi^I_p(F)\rangle \;=\; \sum_{i\in\mathcal{H}^I_{f_I}} d^{\,I,F}_{p,i}\,|D^I_i\rangle .
$$

The determinant (or selected CI) space $\mathcal{H}^I_{f_I}$ and its basis
$\{|D^I_i\rangle\}$ depend only on the sector $f_I$. The coefficient tensor
$d^{\,I,F}$ depends on the full FockConfig $F$. That dependence is the entire
source of the context specificity.

### Overlaps

Within a fixed $F$ the local solve delivers orthonormal states:
$$
\langle\Phi^I_p(F)|\Phi^I_q(F)\rangle=\delta_{pq}.
$$

Across FockConfigs in the same sector ($f_I=f'_I$) they are generally not orthogonal:
$$
s^I_{pq}(F,F')\;=\;\langle\Phi^I_p(F)|\Phi^I_q(F')\rangle
\;=\;\sum_i \big(d^{\,I,F}_{p,i}\big)^*\,d^{\,I,F'}_{q,i}
\;\neq\;\delta_{pq}.
$$
When $f_I\neq f'_I$ the overlap vanishes by quantum numbers.

### Orthonormality of the global basis

$$
\langle F,\mathbf{p}\,|\,F',\mathbf{p}'\rangle
=\prod_{I=1}^{N_c}\langle\Phi^I_{p_I}(F)|\Phi^I_{p'_I}(F')\rangle
=\delta_{FF'}\prod_{I=1}^{N_c}\delta_{p_I p'_I}.
$$

Any $F\neq F'$ has at least one cluster whose sector differs, which zeros the
product. The spectator overlaps $s^I(F,F')$ never reach the global metric, so
$$
\langle\Psi|\Psi\rangle=\sum_{B}|C_B|^2,
$$
with no global metric matrix. The non-orthogonality is quarantined inside $\hat H$.

### Where the non-orthogonality enters

Writing $\hat H$ as a sum of terms $t$, each a product of cluster-local operator
strings $\hat O^I_t$ on support $\mathrm{supp}(t)$,
$$
\langle F\mathbf{p}\,|\hat H|\,F'\mathbf{p}'\rangle
=\sum_t h_t
\underbrace{\prod_{I\in\mathrm{supp}(t)}\langle\Phi^I_{p_I}(F)|\hat O^I_t|\Phi^I_{p'_I}(F')\rangle}_{\text{operator / transition blocks}}
\underbrace{\prod_{I\notin\mathrm{supp}(t)} s^I_{p_I p'_I}(F,F')}_{\text{spectator overlaps}}.
$$
The spectator product is the term that is absent in the shared-basis case. A
spectator in the same sector contributes $s^I(F,F')$ rather than $\delta$; a
spectator whose sector differs forces the term to zero.

---

## Method A: independent cMF per FockConfig, then a single coupled build

Solve a self-contained cluster mean field for each $F$, fixing the basis, then
build and diagonalize once.

For each cluster $I$ in sector $f_I$, an embedded local eigenproblem:
$$
\hat H^I_{\text{cMF}}(F)\,|\Phi^I_p(F)\rangle=\varepsilon^I_p(F)\,|\Phi^I_p(F)\rangle,
\qquad
\hat H^I_{\text{cMF}}(F)=\hat H^I+\sum_{J\neq I}\big\langle\hat V^{IJ}\big\rangle_{\rho^J(F)},
$$
with the embedding density $\rho^J(F)$ taken from the cMF reference of the other
clusters within the same $F$. This fixes every $d^{\,I,F}$.

Then build $H_{F\mathbf{p},F'\mathbf{p}'}$ once in the union basis, including the
spectator overlaps $s^I(F,F')$, and solve the plain eigenproblem
$$
\mathbf{H}\mathbf{C}=E\,\mathbf{C}.
$$

**Character.** Each $d^{\,I,F}$ is stationary for its local embedded energy, not
for the global energy. A residual global gradient $\partial E/\partial d^{\,I,F}\neq 0$
is left unaddressed. The local states are tuned to the mean field of their own
FockConfig and are blind to how the FockConfigs mix in $C$.

**Cost and structure.** Every cMF is a small independent solve, parallel over
$F$, with no global nonlinear loop. The $s^I(F,F')$ are computed once, after the
bases are fixed. This is the diagonalize-then-couple level; it is variational in
$C$ only.

---

## Method B: joint minimization over global and local coefficients

Minimize the energy with respect to both $C$ and the local expansions $d$, with
within-$F$ orthonormality carried as a constraint so the global metric stays the
identity:
$$
\mathcal{L}=\langle\Psi|\hat H|\Psi\rangle
-E\Big(\sum_{B}|C_B|^2-1\Big)
-\sum_{I,F}\sum_{pq}\lambda^{I,F}_{pq}\Big(\langle\Phi^I_p(F)|\Phi^I_q(F)\rangle-\delta_{pq}\Big).
$$

**Stationarity in $C$.** The same plain eigenproblem,
$\mathbf{H}\mathbf{C}=E\,\mathbf{C}$.

**Stationarity in $d$.** A generalized Brillouin condition: the kept cluster
states must not couple to the discarded (external) ones through a
$C$-contracted effective operator,
$$
\big\langle\chi^I_a(F)\big|\hat{\mathcal{F}}^I(F)\big|\Phi^I_p(F)\big\rangle=0
\qquad\forall\,a\in\text{external},
$$
where $\hat{\mathcal{F}}^I(F)$ is the environment contraction of $\hat H$ with
cluster $I$'s slot left open: the bare $\hat H^I$ plus the inter-cluster
$\hat V^{IJ}$ contracted against the $C$-weighted transition densities of the
other clusters, summed over every $F'$ connected to $F$, and carrying the
spectator overlaps $s^J(F,F')$. In contrast to A, whose effective operator is the
intra-$F$ mean field, B's effective operator is the inter-$F$, $C$-weighted,
correlation-dressed operator. The local states in B feel the global state they
build.

**Redundancy.** Rotations among the kept cluster states at fixed $F$ can be
absorbed into a compensating rotation of the $C$ block for that $F$; this is pure
gauge, the CI-orbital redundancy of MCSCF lifted to whole cluster states. The
only genuine freedom in $d$ is the kept-external rotation, i.e. which subspace of
$\mathcal{H}^I_{f_I}$ the retained states span. So B is a per-cluster,
per-FockConfig active-external rotation and nothing more.

**Cost and structure.** An MCSCF-class coupled nonlinear optimization: possible
multiple minima, a kept-external gradient to converge, and the spectator
overlaps $s^I(F,F')$ rebuilt every macroiteration as $d$ relaxes, rather than
once at the end. This is the relaxed, variationally optimized level; reserve
"variational TPS" for B, since A is variational in $C$ only.

---

## Relationship between A and B

For matched kept dimensions $\{m_I(F)\}$,
$$
E_{\text{B}}\le E_{\text{A}},
$$
with equality when B's relaxation is redundant. If the full sector space is kept
($m_I(F)$ complete), the external space is empty, the kept-external rotation is
trivial, and A and B coincide. B does something only when the number of cluster
states is truncated: its value is recovering, through relaxation of the basis,
what would otherwise need additional cluster states to capture.

Both A and B keep a separate basis per FockConfig, which is why
$s^I(F,F')\neq\delta$ survives in either case. In A that context is the local
mean field; in B it is the global energy, and the overlaps become a moving target
recomputed each macroiteration. Imposing a single shared orthonormal basis per
sector across all $F$ would make $s^I(F,F')=\delta$ identically, but it discards
the context specificity that makes the local states good. That shared-basis
choice is a third, strictly less flexible ansatz, worth naming as the boundary
case so the non-orthogonality reads as a deliberate consequence of context
optimization.

A practical path is A as reference and initial guess, then B as relaxation on
top, with the kept-external gradient as the convergence monitor.
