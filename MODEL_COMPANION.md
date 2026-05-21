# Pair-prior MLIP — companion text

*Study notes on the model: what it is, why each piece is built the way it is,
where the classical forms break, and exactly what our extension changes. Written
to be read and learned from, not skimmed — every claim is followed by its
reasoning.*

---

## 0. The idea, and why it is worth the trouble

A machine-learned interatomic potential (MLIP) predicts the energy $E$ of an
arrangement of atoms (and, by differentiation, the forces and stress). The
mainstream approach — DeepMD, NequIP, MACE — learns $E$ **end-to-end**: a neural
network takes the local atomic environment and emits an energy, with no physics
assumed.

We take the opposite stance. Decades of condensed-matter physics already tell us
the *shape* of the metallic energy: a stiff short-range repulsion, an attractive
cohesive part, and a density-dependent "embedding" term. So we write the energy
as a **physical prior plus a small learned correction**:

$$
E_\text{total} \;=\; \underbrace{E_\text{pair}^{\text{EAM}}}_{\text{physics we know}}
\;+\;
\underbrace{\varepsilon_\text{cap}\,\tanh\!\Big(\tfrac{E_\text{corr}}{\varepsilon_\text{cap}}\Big)}_{\text{the residual we don't}} .
$$

Why bother, when a big network can fit everything anyway? Three reasons that
recur throughout these notes:

1. **You only have to learn the hard part.** The network is not asked to
   re-derive Pauli repulsion or metallic binding from scratch; it only models the
   residual the prior misses. Less to learn ⇒ less data needed and better
   behaviour where data is thin.
2. **The result is inspectable.** Because the prior is a real physical object,
   you can ask "how much of this energy is known physics vs. learned correction?"
   On Mn–Ni–Ga the prior carries **~87 %** of the per-atom signal (see §6 for why
   that number is physically reasonable). A pure black-box gives you one opaque
   number.
3. **It extrapolates sanely.** A learned potential can do anything outside its
   training data; a Born–Mayer wall is *guaranteed* to repel at short range, so
   the model stays physical under compression or close approaches in MD.

The rest of this document earns each of those claims.

---

## 1. The pair prior, tier by tier

The prior is `PairFoundationEAM`. It is evaluated on the neighbour graph: for
each atom $i$, its neighbours $j$ within a cutoff $r_c$.

### 1.1 Born–Mayer repulsion — $A_{ij}\,e^{-\alpha_{ij} r_{ij}}$

**What it is.** A stiff, exponentially decaying repulsion between every pair.

**Why this form (not, say, $1/r^{12}$).** When two atoms approach, their
closed electron shells overlap. By the Pauli principle electrons in the same
quantum state are pushed up in energy, producing a repulsion. Because atomic
electron densities fall off *exponentially* with distance ($\psi\sim e^{-\kappa
r}$ for bound states), the overlap — and hence the repulsion — is also
exponential. This is the Born–Mayer (and Born–Mayer–Huggins) form, derived from
screened nuclear overlap; it is more physically grounded at short range than the
empirical $r^{-12}$ of Lennard-Jones.

**Why it is not enough by itself.** It is *purely repulsive and monotonic*: it
has no minimum, so it can never bind a solid. A crystal held together by
Born–Mayer alone would simply fly apart. You need an attractive, cohesive
contribution.

### 1.2 Attractive radial tail — $\sum_k w_{ij,k}\,\mathrm{sinc}_k(r_{ij})$

**What it is.** A flexible learned attractive/oscillatory tail expressed in a
sinc radial basis (a set of localized radial bumps); the $w_{ij,k}$ are learned
weights.

**Why a learned basis rather than a fixed $\phi(r)$.** The effective pair
interaction in a metal is not a simple universal curve — it has Friedel-like
oscillations (the conduction electrons screen ions, producing a damped
oscillatory tail) and it depends on the chemistry of the pair. A basis expansion
lets the model represent whatever radial shape the data demand instead of
committing to one analytic guess.

### 1.3 EAM embedding — $F(\rho_i, Z_i)$ with $\rho_i=\sum_j \psi_{Z_j}(r_{ij})\,f_\text{cut}$

**What it is.** Each atom accumulates a local "electron density" $\rho_i$ from
its neighbours, then pays an embedding energy $F(\rho_i, Z_i)$.

**Why this captures cohesion that pairwise terms cannot.** This is the key idea
of the Embedded Atom Method, and it is rooted in density-functional theory: to a
good approximation, the energy to embed an atom into a host is a function of the
local electron density it sits in. Crucially, $F$ is **non-linear** in $\rho$.
That non-linearity is what makes bonding *environment-dependent*: an atom with
many neighbours (high $\rho$) does not gain energy in proportion to the number of
neighbours — the bonds "share" and weaken. This single fact explains why metals
have the cohesive-energy/coordination trends, surface relaxations and vacancy
formation energies that a strictly pairwise sum gets wrong. A pure pair sum is
linear in the number of bonds; the embedding term is the cheapest way to break
that linearity.

So already the prior is physically complete in the EAM sense: repulsion (1.1) +
cohesion (1.2) + density-dependent many-body glue (1.3). §3 explains why this
is still not enough, and §2 explains the one thing that makes our prior more than
a textbook EAM.

---

## 2. Why the coefficients are *learned functions of chemistry* — and why that generalizes

In a textbook EAM/Born–Mayer potential the coefficients $A, \alpha, w$ and the
density function are **fixed numbers/curves fit per element or per species
pair**. To handle an $n$-element alloy you fit a separate table for every pair,
and a composition you did not fit has *no defined coefficients at all*.

In our prior, the coefficients are not free constants. Each element $Z$ is mapped
to a learned **chemistry embedding vector** $e(Z)\in\mathbb{R}^d$, a pair is
formed as $p_{ij}=e(Z_i)\odot e(Z_j)$ (elementwise product), and small readout
MLPs turn $p_{ij}$ into $A_{ij}, \alpha_{ij}, w_{ij,k}$ and the density readouts
$\psi_{Z}$.

**Now the claim "this generalizes across composition" — and the actual
mechanism behind it:**

- **The map $Z\mapsto e(Z)$ is continuous-valued and learned jointly with the
  potential.** Nothing forces two elements to be far apart in embedding space.
  During training, elements that play similar chemical/energetic roles get pushed
  to *nearby* embedding vectors, because that lets the shared readout MLPs reuse
  the same parameters to fit both — the optimizer is rewarded for discovering a
  soft, data-driven "periodic table" geometry in $\mathbb{R}^d$.

- **The readouts are smooth (Lipschitz) functions of $p_{ij}$.** A small change
  in the embedding produces a small change in the coefficients. So if a new
  alloy's atoms have embeddings that lie *between* embeddings seen in training,
  the resulting pair coefficients lie between the trained ones too. The model
  **interpolates in chemistry space**, exactly as a smooth function interpolates
  between sampled points. That is the precise sense in which it "generalizes to
  mixtures it was not explicitly fit to."

- **Contrast with one-hot per-pair tables.** A one-hot/lookup scheme has no
  notion of "between": each pair is an independent free parameter, so an unseen
  pair is simply undefined and a slightly different composition carries *zero*
  information from neighbouring compositions. Continuity is what converts
  "memorize each pair" into "learn a function of chemistry."

- **The honest caveat.** This is **interpolation**, not magic. It transfers well
  to compositions and orderings *inside the convex region the training elements
  and concentrations span*. It does **not** justify extrapolating to chemistry
  far outside that region (a genuinely new element, or a concentration regime with
  new physics) — there the embedding has no information to interpolate from. The
  guarantee is "smooth and sane between what you've seen," not "correct
  everywhere."

This is also why the prior alone is already more transferable than a classical
EAM: the classical model must be re-fit for each new alloy; ours slides
continuously along composition.

---

## 3. Why EAM/Born–Mayer are *structurally* insufficient — the angular blindness

Even the complete EAM prior of §1 has a hard ceiling, and it is worth seeing
*exactly* why, because it is what motivates the equivariant correction.

**The density is rotationally blind.** Look at the density:
$\rho_i=\sum_j \psi(r_{ij})$. It depends only on the **distances** $r_{ij}$, never
on the **directions** $\hat r_{ij}$. Therefore $\rho_i$ — and any energy built
from it — is invariant not just to rotating the whole cluster (which is
correct), but to rotating *each neighbour independently about $i$* at fixed
distance (which is **not** physical). The model literally cannot tell two
environments apart if they share the same set of bond lengths.

**Concrete example.** Take three atoms: a central atom and two neighbours, both
at distance $r$. Whether the trimer is *linear* (180°) or sharply *bent* (60°),
the multiset of distances from the centre is identical $\{r, r\}$, so $\rho_i$ is
identical, so EAM assigns the **same energy**. But the true energies differ
strongly — bond angles cost energy; this is the whole basis of covalent and
directional bonding (Si, many transition-metal compounds, intermetallics,
surfaces and defects). A central-force model is blind to it by construction.

**Multi-channel density does not fix this.** We use a **16-channel** learned
density $\rho_i\in\mathbb{R}^{16}$ rather than a single scalar — i.e. 16 different
radial weightings $\psi_c$. That genuinely helps: the embedding $F$ can read a
richer picture of the *radial* environment (it can distinguish "few close + many
far" from "uniform shell," which a single scalar blurs together). But every
channel is still $\rho_{i,c}=\sum_j \psi_c(r_{ij})$ — a sum over a radial
function. It is still a function of distances only. **More channels add radial
resolution, not angular resolution.** The angular blindness is a property of the
*central-force form itself*, and no amount of radial richness removes it.

**What is also simply absent: magnetism and charge.** Classical EAM has no spin
or charge degrees of freedom. A magnetic ternary like Mn–Ni–Ga has energetics
that depend on local moments and their arrangement; EAM cannot represent that at
all.

So the residual EAM leaves on the table is *structured*: it is precisely the
angular/directional and magnetic physics. That is what the correction must
supply.

---

## 4. The correction: equivariant, and why $Y_2$ specifically

`CorrectionGNN_v22` is a PaiNN-style equivariant message-passing network. Each
atom carries a **scalar** feature $h_i$ and a **vector** feature $\vec v_i$,
updated over $L$ layers by exchanging messages with neighbours.

**Why equivariant (and what that word buys us).** A physical energy is invariant
under rotating the whole system, and forces rotate *with* it. "Equivariant"
features are ones that transform correctly under rotation — scalars stay fixed,
vectors rotate. Building this symmetry into the network means it never has to
waste capacity or data *learning* that physics is rotationally symmetric; the
symmetry is exact by construction. This is the same reason NequIP/MACE are so
data-efficient, and we inherit it for the correction.

**Why the $\ell=2$ invariant $\big\lVert\sum_j w_j(h)\,Y_2(\hat r_{ij})\big\rVert^2$.**
This is the term that directly attacks the angular blindness of §3. The spherical
harmonics $Y_2(\hat r_{ij})$ are the lowest-order functions that encode the
**angular** distribution of neighbours (the $\ell=0$ part is just count/density —
what EAM already has; $\ell=1$ vanishes for centrosymmetric environments; $\ell=2$
is the leading *anisotropy*, the quadrupolar "is this environment stretched or
squashed, and along what axis"). Summing $Y_2$ over neighbours and taking the
squared norm gives a **rotation-invariant** scalar that is nonzero exactly when
the neighbour arrangement is angularly anisotropic — i.e. precisely the
information $\rho_i$ throws away. Folding it back into $h_i$ lets the correction
represent bond-angle energetics that the prior cannot.

**Why bounded forces/stress come "for free" and correctly.** We compute one total
energy $E_\text{total}$ and obtain forces and stress by autograd:
$\vec F_i=-\partial E/\partial \vec r_i$, $\sigma=\tfrac1V\,\partial E/\partial
\varepsilon$. Because they are exact gradients of a single scalar energy, the
forces are **conservative** — $\oint \vec F\cdot d\vec r=0$. This matters in
practice: in molecular dynamics, conservative forces conserve energy, while a
model that predicts forces *directly* (independently of its energy) has a
non-conservative force field that slowly pumps or drains energy and corrupts
long MD trajectories and any thermodynamic average taken from them.

---

## 5. The `tanh` cap — a gauge-fixing device, explained

The cap $\varepsilon_\text{cap}$ (default **0.3 eV/atom**) is the load-bearing
trick. Here is the problem it solves and why this is the right solution.

**The problem: a gauge freedom.** We only ever supervise the *total* energy (and
its derivatives). But the total is a **sum** of two learnable pieces,
$E_\text{pair}+E_\text{corr}$. For any constant (or smooth field) $c$, the pair
$(E_\text{pair}+c,\;E_\text{corr}-c)$ gives the identical total and therefore the
identical loss. The split is **underdetermined** — a gauge freedom. Left alone,
the optimizer happily drifts into a regime where the flexible network
$E_\text{corr}$ absorbs more and more of the energy and the physical prior
$E_\text{pair}$ is pushed into an unphysical, compensating shape. The total still
fits, but the interpretability (claim #2 in §0) is destroyed and the "physical
prior" is physical in name only. We call this the cancellation drift.

**Why a cap fixes it.** Wrapping the correction in
$\varepsilon_\text{cap}\tanh(E_\text{corr}/\varepsilon_\text{cap})$ makes the
correction's contribution **bounded by $\varepsilon_\text{cap}$ per atom**, by
construction. The network simply *cannot* absorb more than 0.3 eV/atom, so it
cannot swallow the prior. The gauge freedom is broken from the correction side:
whatever the prior cannot explain beyond the cap *must* be explained by the prior
itself, forcing it to stay physical.

**Why `tanh` specifically.** Three properties make it ideal:
- it is **smooth** (everywhere differentiable → clean forces);
- near zero, $\tanh(x)\approx x$, so **small corrections pass through almost
  linearly** (gradient ≈ 1) — the cap does not distort the easy regime where the
  correction is small;
- it **saturates** gently at $\pm\varepsilon_\text{cap}$ rather than clipping with
  a kink, so there is no discontinuous gradient to destabilize training or MD.

**Why a cap rather than a loss penalty.** One could instead add a regularizer
penalizing $|E_\text{corr}|$. In practice that is fragile: it trades off against
the fit, needs a hand-tuned weight, and (as we found) symmetric penalties on both
the pair and correction terms can *backfire* and collapse the hierarchy. A hard
architectural cap is a *guarantee*, not a soft pressure — the bound holds
regardless of optimizer, learning rate or data.

---

## 6. Why ~87 % pairwise is the *expected* and *desirable* outcome

It might seem worrying that the learned correction ends up small. It is exactly
right, for a physical reason. In metals the overwhelming part of the cohesive
energy *is* well described by pairwise interactions plus a density-embedding term
— that is precisely why EAM has been a workhorse for fifty years. The
angular/directional and magnetic contributions are real and **decisive for
specific properties** (elastic anisotropy, stacking faults, defect energetics,
magnetic ordering) but they are *energetically small* compared to bulk cohesion.

So a healthy model on metallic data *should* put most of the energy in the
physical prior and reserve a small, bounded correction for the property-critical
residual. The ~87 % figure is the model confirming the physics, not a deficiency.
It is also the quantitative payoff of claim #2: we can actually *measure* the
decomposition because the architecture keeps the two terms separable.

---

## 7. How it differs from DeepMD, MACE and NequIP — with the reasoning

All three references learn the full energy with **no physical prior**. They
differ mainly in how they encode the local environment:

- **DeepMD.** Builds a smooth, *rotation-invariant* descriptor of each local
  environment and feeds it to a fitting network. Invariance is achieved by
  constructing the descriptor from inner products of neighbour vectors. *Why this
  is a limitation:* an invariant-from-the-start representation discards
  directional (vector/tensor) information early, so the model must compensate with
  many descriptor components and depth, and it has no built-in short-range
  physics — the entire PES, repulsion included, is learned.

- **NequIP.** *Equivariant* message passing built from tensor products of
  spherical harmonics, so internal features carry angular ($\ell>0$) character
  throughout. *Why it works well:* keeping equivariant features means the network
  never wastes data learning rotational symmetry, giving excellent data
  efficiency and accuracy — at higher computational cost per evaluation, and
  still learning everything from scratch.

- **MACE.** Adds **high body-order** messages via an Atomic Cluster Expansion, so
  a single layer already sees 3-, 4-body correlations. *Why this matters:* it
  reaches high accuracy in very few message-passing layers (cheaper, less
  over-smoothing than deep GNNs) — but again, no physical prior; the model is the
  potential.

**Our model is a different philosophy, not just a different network.** Where the
others ask a network to *re-discover* repulsion, cohesion and embedding from
data, we *supply* those as a structured, interpretable prior (§1–§2) and ask an
equivariant network only for the angular/magnetic residual (§3–§4), bounded so
the physics stays dominant (§5). The consequences follow directly from §0:
interpretable energy decomposition, physically correct short-range extrapolation,
and data efficiency because only the residual is learned. Even with the
correction switched off, what remains is a sensible physical potential — a
property none of DeepMD/MACE/NequIP have.

### When to reach for the pair-prior model

Use it when you want MLIP-level accuracy *and* a physically meaningful,
inspectable potential — especially for chemically disordered and magnetic
metallic alloys, where (i) transferability across composition matters and the
chemistry-embedding prior interpolates sensibly (§2), (ii) well-behaved
short-range physics matters for MD/high-pressure (§0, §1.1), and (iii) being able
to attribute energy to "known physics" vs. "learned correction" is itself a
scientific result (§2, §6).
