# Pair-prior MLIP — companion text

*A plain-language but technically precise companion to the poster. It explains
what the model is, the decisions behind it, how it differs from DeepMD, MACE and
NequIP, and — in detail — why a classical EAM or Born–Mayer potential is not
enough on its own and what exactly our extension changes.*

---

## 1. The model in one line

$$
E_\text{total} \;=\; \underbrace{E_\text{pair}^{\text{EAM}}}_{\text{physical prior}}
\;+\;
\underbrace{\varepsilon_\text{cap}\,\tanh\!\Big(\tfrac{E_\text{corr}}{\varepsilon_\text{cap}}\Big)}_{\text{bounded equivariant correction}}
$$

We do **not** learn the energy from scratch. A physically structured pair prior
carries most of the interaction; a small, bounded $E(3)$-equivariant graph
network adds only the many-body and magnetic detail the prior cannot represent.
Forces and the stress tensor come from a single autograd pass through this total
energy, so the potential is **conservative by construction**.

The cap $\varepsilon_\text{cap}$ (default **0.3 eV/atom**) is a hard `tanh`
limit on the per-atom correction. It is the load-bearing design choice: it
prevents the degenerate solution where the network silently inflates
$E_\text{corr}$ and deflates $E_\text{pair}$ (the $(E_\text{pair}+c,\,
E_\text{corr}-c)$ cancellation drift), which would destroy the interpretability
of the prior. On Mn–Ni–Ga the pair term ends up carrying **~87 %** of the
per-atom signal — the correction is genuinely a correction.

---

## 2. Architecture

Two components feed one energy.

### 2.1 `PairFoundationEAM` — the physical prior

Three tiers, all evaluated on the neighbour graph within a cutoff $r_c$:

1. **Born–Mayer repulsion** (per edge)
   $\;A_{ij}\,e^{-\alpha_{ij} r_{ij}}$ — short-range Pauli repulsion.
2. **Attractive radial tail** (per edge)
   $\;\sum_k w_{ij,k}\,\mathrm{sinc}_k(r_{ij})$ — a flexible learned bonding tail
   in a sinc radial basis.
3. **EAM embedding** (per atom)
   $\;F(\rho_i, Z_i)$, with a **16-channel learned electron density**
   $\rho_i = \sum_j \psi_{Z_j}(r_{ij})\,f_\text{cut}(r_{ij})$.

The crucial twist: every coefficient — $A_{ij}$, $\alpha_{ij}$, the attractive
weights $w_{ij,k}$, and the density readouts $\psi_{Z}$ — is **produced by a
chemistry embedding** $e(Z)$, combined per pair as $p_{ij}=e(Z_i)\odot e(Z_j)$.
Because $e(Z)$ is continuous, the pair physics varies smoothly across
composition and the prior generalises to mixtures it was not explicitly fit to.

### 2.2 `CorrectionGNN_v22` — the bounded correction

A PaiNN-style equivariant message-passing network: each atom carries a scalar
feature $h_i$ and a vector feature $\vec v_i$, updated over $L$ layers. At each
layer a rotation-invariant $\ell=2$ feature
$\big\lVert\sum_j w_j(h)\,Y_2(\hat r_{ij})\big\rVert^2$ is folded back into the
scalar channel, giving the network access to angular/directional information
that a central-force model cannot encode. Its output $E_\text{corr}$ is squashed
through the `tanh` cap before being added to the prior.

---

## 3. Why not EAM or Born–Mayer alone?

These classical forms are the historical backbone of metallic potentials, and
they are cheap, smooth and physically motivated. But each is structurally
limited.

**Born–Mayer** is a pure pairwise repulsion, $V(r)=A\,e^{-r/\rho}$. It has *no*
many-body term at all: the energy is a sum over independent pairs. It cannot
distinguish a close-packed environment from an open one at fixed pair distances,
cannot bind a metal on its own, and has no notion of coordination.

**EAM** repairs the worst of this with the embedding term
$E_i = \tfrac12\sum_j \phi(r_{ij}) + F(\rho_i)$, where
$\rho_i=\sum_j \rho(r_{ij})$. The "glue" $F(\rho_i)$ makes bonds depend on local
density, which is why EAM works so well for simple close-packed metals (Cu, Al,
Ni). **But the density $\rho_i$ is a single, spherically symmetric scalar.**
This is the root limitation:

- **No angular / directional bonding.** $\rho_i$ depends only on neighbour
  *distances*, not on the *arrangement* of neighbours. Two configurations with
  identical radial distributions but different bond angles get the same energy.
  Covalent or partially covalent bonding (Si, directional TM bonding, many
  intermetallics) is invisible.
- **No bond-order / chemical environment beyond density.** A central-force
  model cannot represent the Tersoff/bond-order effects that matter at surfaces,
  defects and in non-close-packed phases.
- **No magnetism, no charge transfer.** Classical EAM has no spin or charge
  degrees of freedom, so it cannot describe the magnetic energetics of a system
  like Mn–Ni–Ga.
- **Poor transferability across composition.** Coefficients are fit per element
  or per binary; there is no principled continuation to ternary/multicomponent
  disorder.

These are exactly the regimes our datasets live in (magnetic ternary, disordered
multicomponent alloys), so EAM/Born–Mayer alone leave a large, *structured*
residual on the table.

---

## 4. What our extension changes — EAM/Born–Mayer vs. our prior

This is the key distinction, point by point. "Typical" = textbook EAM / Born–Mayer.

| Aspect | Typical EAM / Born–Mayer | Our `PairFoundationEAM` |
|---|---|---|
| Pair coefficients | Fixed analytic or spline functions, fit **per species pair** | $A_{ij},\alpha_{ij},w_{ij,k}$ are **outputs of a chemistry embedding** $e(Z_i)\odot e(Z_j)$ → continuous in composition |
| Density $\rho_i$ | **Single scalar**, spherically symmetric | **16-channel learned vector**, species-dependent $\psi_{Z_j}$ |
| Embedding $F$ | Fixed 1-D function $F(\rho)$ | Learned MLP on $(\rho_i \,\Vert\, \text{chem-emb}_i)$ |
| Attractive part | Often absent (Born–Mayer) or a single fitted $\phi(r)$ | Flexible learned sinc-basis tail |
| Generalisation | Re-fit for each new alloy | Smooth across composition by design |
| Many-body angular terms | **None** (central force) | Supplied by the bounded equivariant correction on top |
| Magnetism / charge | **None** | Carried by the correction's auxiliary heads |

So the difference is two-fold. **(1)** Even the *prior itself* is a generalised,
learnable EAM: its coefficients are functions of chemistry rather than frozen
per-pair constants, and its density is a multi-channel vector rather than a
scalar, so the embedding $F$ can express richer many-body physics. **(2)** On top
of that prior we add an equivariant correction that supplies precisely the
angular, directional and magnetic physics that *no* central-force model — however
cleverly fit — can represent, while the `tanh` cap keeps the interpretable
physical prior dominant.

The result: the model degrades gracefully. Where the physics is simple
(close-packed metals like Cu), the prior already does almost all the work and the
correction is nearly idle. Where the physics is hard (magnetic Mn–Ni–Ga,
disordered Al–Mg–Si), the correction earns its keep — but is never allowed to
take over and turn the model into an opaque black box.

---

## 5. How it differs from DeepMD, MACE and NequIP

All three references learn the energy **end-to-end with no physical prior**. They
differ from each other mainly in how they encode the local environment:

- **DeepMD** — a smooth descriptor of the local environment feeds a fitting
  network. Invariant (not equivariant) by construction; effective and fast, but
  the entire energy is learned and the model is a black box with no built-in
  short-range physics.
- **NequIP** — $E(3)$-equivariant message passing built from tensor products of
  spherical harmonics. Very data-efficient and accurate, but again learns
  everything from scratch and is comparatively expensive.
- **MACE** — higher body-order equivariant messages (an ACE-style expansion),
  reaching high accuracy in few layers. Same philosophy: no physical prior, the
  full energy is learned.

**Our model is philosophically different.** Instead of asking a network to
re-discover Pauli repulsion, metallic binding and density-dependent embedding
from data, we *give* it those as a structured, interpretable prior and ask the
equivariant network only for the residual. The practical consequences:

- **Interpretability.** We can read off how much of the energy is "known
  physics" (the pair prior) versus "learned correction" — and on our data the
  prior dominates (~87 % on Mn–Ni–Ga). DeepMD/MACE/NequIP give a single opaque
  number.
- **Built-in short-range behaviour.** The Born–Mayer wall is physically correct
  by construction, so extrapolation to short distances (high pressure, close
  approaches in MD) is far better behaved than a purely learned potential.
- **Data efficiency where it matters.** The network only has to learn a small,
  bounded correction, not the whole potential energy surface.
- **A principled fallback.** Even with zero correction the model is a sensible
  physical potential — useful for stability and for diagnosing what the network
  actually contributes.

### Why would one use the pair-prior model?

Use it when you want MLIP-level accuracy **and** want to keep a physically
meaningful, inspectable potential — especially for chemically disordered and
magnetic metallic alloys, where transferability across composition and
well-behaved short-range physics matter, and where being able to attribute the
energy to "physics" vs. "learned correction" is scientifically valuable.
