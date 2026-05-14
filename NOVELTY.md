# Honest novelty assessment

What this document is: a candid placement of the `pair-prior` MLIP in the
modern ML-interatomic-potential landscape. It's not advocacy. Where parts
of the design are unoriginal, that's said plainly. Where there's a real
engineering contribution, that's also said plainly. The claim that "pair
priors help most with small datasets" is examined against the literature.

Cited papers and the search notes that produced them are in
`sources/literature_review_notes.md`.

## TL;DR

This is a **production-engineered MLIP that combines well-established
design patterns**, tuned for small-DFT-dataset training on magnetic
transition-metal alloys. The combination is sensible and the implementation
is solid. **It is not a foundational research contribution and would not be
expected to outperform NequIP / MACE / Allegro on standard benchmarks.**

The intuition that "the pair prior works best for small datasets" is
**partly correct but easy to overstate**. The dominant source of data
efficiency in modern MLIPs is equivariance, not pair priors. Pair priors
mostly help with extrapolation robustness (short-range, AL-driven
out-of-distribution) and avoid the "GNN learns nonsense in unseen regions"
failure mode. They don't replace the equivariance multiplier.

## Component-by-component honest assessment

### 0. Specifically: the additive form `E_total = E_pair + E_GNN` — **not novel, but what's distinctive is having the pair parameters be learnable**

This deserves its own block because it's a question worth answering precisely.

**Yes, additive `E_pair + E_NN` has been done many times.** The published precedents,
ordered by closeness to what we do:

| Approach                        | Pair form        | Pair parameters | Pair trained jointly with NN? | Notes |
| ---                             | ---              | ---             | ---                            | ---   |
| **DeePMD-kit pairtab**          | tabulated `E(r)` | **fixed** (user-supplied table) | DP trained, pair frozen at inference | Direct sum or interpolated weighting between DP and pair. ZBL is the typical pair. [^deepmd-pairtab] |
| **NEP + ZBL** (Liu 2021)        | universal ZBL    | **fixed** (Z-only formula) | NEP trained, ZBL frozen | Used for radiation damage in W. [^nep-zbl] |
| **DimezblNet** (2024)           | universal ZBL    | **fixed**       | DimeNet trained, ZBL frozen     | Embeds ZBL into DimeNet at training time, not deploy time. [^dimezbl] |
| **MTP + ZBL, GAP + ZBL**        | universal ZBL    | **fixed**       | Layered (deploy-time hybrid)    | LAMMPS `pair_style hybrid/overlay`. [^practical-guide] |
| **PhysNet** (Unke 2019)         | `q_i q_j / r` long-range | **learnable** (charges from NN) | Yes, joint                      | Different physics (electrostatics, not short-range repulsion). [^physnet] |
| **Yang 2025** (arXiv 2504.15925) | empirical short-range | not specified in abstract; "compatible with most existing MLFF architectures" | Yes, joint   | Tests on LLZO with **25 training configurations**, claims improved robustness AND **training efficiency**. The closest published match in spirit. [^yang-hybrid] |
| **This codebase**               | learnable Born-Mayer + learnable attractive radial | **learnable per-species-pair**, factored through `pair_rank`-d embeddings | Yes, joint | A and α come from `MLP(species_embed[Z_i] · species_embed[Z_j])` |

**What's well-established (not novel):**
- The additive structure itself (`E = E_pair + E_NN`) is the *standard* way to
  combine an empirical baseline with an ML correction. It dates back to the
  Δ-ML literature (Ramakrishnan et al. 2015) and shows up everywhere from
  DeePMD to MACE-MP foundation models.
- Joint training of an ML correction on top of a fixed empirical pair (NEP+ZBL,
  DimezblNet, etc.) is published and routine.

**What's a smaller engineering distinction (still not novel-with-a-paper, but
less common in the surveyed literature):**
- The pair parameters being **learnable per species pair** rather than fixed
  by atomic number (as ZBL is). PhysNet is the most cited prior example of
  "learnable parameters in the additive empirical-style term," but PhysNet's
  learnable terms are charges driving Coulomb electrostatics, not a Born-Mayer
  short-range repulsion.
- The use of **a learned *attractive* radial term** alongside the learned
  repulsion (`Σ w(Z_i, Z_j) · sinc_basis(r)`) — this is essentially "an
  embedded GAP-flavored pair potential," running in parallel to the
  message-passing backbone. The closest precedent would be "Pairwise
  interactions for potential energy surfaces and atomic forces using deep
  neural networks" [^pair-nn-2022], which uses pairwise inputs in a deep NN
  but doesn't separate them into a distinct empirical-style additive term.

**What the most relevant precedent (Yang 2025) actually shows:**
The arXiv 2504.15925 paper from April 2025 directly tests a hybrid empirical
+ MLFF architecture and concludes:

- **Training efficiency improves**: it works "with just 25 training
  configurations" on LLZO.
- **Robustness improves**: "Purely data-driven MLFFs fail to prevent
  unphysical atomistic clustering in extended simulations" — the hybrid
  fixes this.

That paper is the closest existing endorsement of the user's intuition that
"pair priors help most for small datasets." It's a 2025 result, fairly
recent, and directly relevant. Their pair part is described as "an
empirical short-range repulsive potential" but the abstract doesn't clarify
whether parameters are learnable.

**Bottom line:** The additive form is **not** novel. The variant with
fully-learnable per-species-pair pair parameters trained jointly with an
equivariant GNN backbone is **less common but not unique** — especially
once you count GAP-style learnable pair contributions as precedent. Calling
this part of the design "novel" in a paper would be a stretch; calling it
"a sensible refinement of an established pattern" is fair.

---

### 1. Pair-potential prior with learnable repulsion — **not novel**

The structure is:

```
E_pair = Σ_(i,j)  [A(Z_i,Z_j) · exp(−α(Z_i,Z_j) · r)
                   + w(Z_i,Z_j) · sinc_basis(r)]
```

i.e. a learnable Born-Mayer repulsion + a learnable attractive term, with
all parameters factored through small species-pair embeddings. This is a
known design pattern:

- **DimezblNet** (Computers and AI, 2024) embeds the ZBL screened-Coulomb
  repulsion explicitly into DimeNet, with the same motivation we have:
  "compensate for extrapolation biases caused by missing data in extreme
  short-range regions." [^dimezbl]
- **NequIP / MACE / Allegro** all support `pair_zbl` as a hybrid component
  via LAMMPS `pair_style hybrid/overlay`. [^nequip-readme] [^mace-readme]
- The "practical guide to MLIPs" (Berkeley, 2025) explicitly states:
  "A repulsive Ziegler-Biersack-Littmark (ZBL) term can be added to the
  potential as a means to improve accuracy." [^practical-guide]
- A direct title hit: "Prediction of Interatomic Potentials Combining
  Empirical Potential and Graph Neural Networks" describes this strategy
  as a known approach. [^empirical-gnn]

What's mildly distinctive in our version: the prior parameters are
**learnable per species-pair** (rather than the fixed-by-atomic-number ZBL
formula). This is a small generalization — empirically a learned prior can
match the dataset's effective core potential better than the universal ZBL
form. Whether this matters depends on the system.

**Verdict**: Pattern not novel. Specific parameterization (learned
Born-Mayer + learned attractive radial) is a small engineering choice.

### 2. Sinc radial basis with polynomial envelope — **not novel**

`f_n(r) = sin(nπr/c) · (1 − (r/c)²)²`. Variations on this:

- **DimeNet** uses Bessel basis `sin(nπr/c) / r · envelope` — the strict
  orthogonal version on `[0, c]`.
- **PaiNN** uses similar Bessel-flavored radial bases.
- Our version drops the `1/r` (which is what makes Bessels orthogonal),
  trading exact orthogonality for numerical stability near `r = 0`.

This is at most a minor variation on a 5-year-old standard.

### 3. AngularAttention message passing — **not novel**

Multi-head attention on (scalar + angular projection of vector features +
radial basis + species-pair embedding), with degree-normalized aggregation
and a gated residual. The recipe is:

- **PaiNN** family: scalar + vector node features, equivariant updates.
- **Equiformer / SE(3)-Transformer**: attention-based equivariant models.
- **NequIP / MACE**: spherical-harmonic equivariance (more expressive
  but more expensive).

We're a Cartesian (scalar + vector, no higher-order tensors) attention
variant. Cheaper than NequIP/MACE, less expressive. Standard trade-off.

The **degree normalization** (`agg /= sqrt(degree+1)`) is a known fix for
mixed-coordination training data — it shows up in several papers and was
necessary in our case to stop high-CN sites (Heusler L2₁ inner atoms)
from dominating the loss. Engineering necessity, not a contribution.

### 4. Direct virial head with detached features — **the most distinctive piece**

```
σ_atom = VirialHead(h.detach(), v.detach(), Z)
```

i.e. a separate equivariant head that reads the message-passed features
and emits per-atom Voigt virial directly, with the input features
**detached** so the stress loss `L_S = ‖σ_pred − σ_target‖²` cannot flow
gradients back into the message-passing backbone.

Direct virial heads as such are not new:

- **HotPP** [^hotpp] is an E(n)-equivariant tensor MP model with separate
  prediction heads for forces *and* virials, not via autograd of energy.
- **GNNFF** [^gnnff] predicts forces directly (non-conservative), and is
  the precedent for "skip the autograd step, use a head."
- The "non-conservative MLIP" line of work is explicit that some of the
  best-performing models train forces and stress separately from energy.

What I did **not** find in the surveyed literature is the **explicit
detach** that severs gradient flow from stress loss into the energy/force
backbone. The motivation here was concrete: in earlier iterations, a high
stress-loss weight degraded force R² over ~50 epochs. Detaching the head
inputs decouples the trade-off entirely — the head is fit *given* the
backbone's representation, and the backbone is updated only by energy and
force losses. Force quality is preserved; stress quality grows
independently.

This is a small but real engineering contribution — gradient surgery on
the multi-task loss. I'd describe it as "useful design choice that may or
may not be in the literature I missed; if it's published, it's not a
prominent talking point."

If we wanted to publish this as a methods note, the contribution would
be: "decoupling auxiliary-property heads from the backbone by detaching
their inputs reliably prevents auxiliary losses from harming primary-task
quality, at the cost of treating the auxiliary as a fixed-feature
regression problem."

### 5. Discrete 3-class spin label per atom — **not novel**

The mechanism (NM=0 / FM-up=1 / FM-down=2 per-atom, embedded and
concatenated into the scalar feature):

- **Behler 2021** [^behler-spin] adds spin degrees of freedom to
  atom-centered symmetry functions. The per-atom-spin descriptor approach
  is explicit in that paper.
- **SpinGNN** (Phys. Rev. B 2024) [^spingnn] is a full spin-dependent GNN
  with two distinct edge types (Heisenberg + spin-distance).
- **Spin-informed universal GNN** (2025) [^spin-univ] incorporates spin
  as input with symmetry preservation.
- **SpookyNet** [^spookynet] — system-level spin attribute via attention.

Our 3-class discrete label is a *cheaper* version of these. It works
because Heusler training data is collinear (spin-up vs spin-down only) and
because we're not modeling spin dynamics. For non-collinear or thermal
spin behavior, this would need to grow into a continuous-moment input.

### 6. Per-atom virial parameterization — **standard trick**

Total virial scales with `N · V`; per-atom virial is intensive. Mixing
B2 (2-atom) and L2₁ (16-atom) supercells in the same loss without this
normalization gave a 64–100× imbalance in gradient signal. Fix: divide
predictions and targets by atom count.

This is a standard normalization choice; it's only worth flagging because
without it the model literally couldn't fit B2 stress while L2₁ was being
trained.

## On "the pair prior helps most with small data"

This claim is **partly true but easily overstated**, and the literature
is fairly clear about why.

### What's true

- Physics priors **always reduce variance**. Fewer parameters need to be
  fit from data; that's a textbook small-data benefit.
- Repulsion priors specifically **prevent pathological predictions in
  unseen short-range regimes**. When training data has no `d < 1.5 Å`
  pair separations and an active-learning rollout proposes such a
  geometry, a no-prior GNN can return a finite, low energy — and the AL
  loop diverges. The DimezblNet paper makes exactly this argument.
- For active-learning workflows on tight DFT budgets, this is a real
  practical win. Less wall-clock spent retraining out crashed AL runs.

### What's overstated

- The dominant data-efficiency mechanism in modern MLIPs is
  **equivariance**, not pair priors. NequIP claims **3 orders of
  magnitude fewer training data** [^nequip] for accuracy parity with
  prior invariant models. That's the equivariance multiplier.
- Pair priors give percent-level MAE improvements (DimezblNet: 3.7–5.3%
  on MD17) — useful but not revolutionary.
- For a fixed equivariance backbone, the pair prior's main effect is
  **extrapolation robustness**, not in-distribution sample efficiency.
  In-distribution, the GNN can fit the short-range region just fine.

### How to actually test the claim on this codebase

If the small-data hypothesis matters for the project, the experiment is
straightforward:

1. Take a fixed evaluation set from MnNiGa (e.g. the existing val split).
2. Train two configurations on training-set sizes
   `{50, 200, 500, 2000}` frames:
   - **A**: full model (with pair prior).
   - **B**: ablation — `pair_potential.pair_scale` zeroed and frozen, so
     `E_pair = 0` everywhere.
3. Compare validation R²_E, R²_F, R²_S vs training-set size.

Three plausible outcomes:

- A consistently beats B by a large margin at low data → claim confirmed.
- A and B converge as data grows; A is only better at small data → claim
  *narrowly* confirmed (small-data benefit, vanishes with data).
- A and B perform similarly at all data sizes → claim refuted; the prior
  doesn't carry weight in-distribution. (Most likely outcome based on the
  DimezblNet results, where the prior helped 3-5% across all data sizes
  rather than dramatically more at small data.)

The clean experiment hasn't been run on this codebase. Until it is, "the
pair prior helps small data" is a plausible-but-untested hypothesis here,
and the literature suggests the effect would be modest.

## Where this model belongs in the landscape

| Axis                    | Where this sits                                   |
| ---                     | ---                                               |
| Equivariance order      | scalar + vector (PaiNN family). Below NequIP/MACE (full SH tensors). |
| Cutoff range            | short (5 Å), single message-passing depth × 3 ≈ 15 Å receptive field. Same as most short-range MLIPs. |
| Pair prior              | learnable Born-Mayer + learnable attractive radial. Slightly more flexible than fixed ZBL. |
| Stress prediction       | direct equivariant head with detached features. Minor design distinction. |
| Spin handling           | discrete 3-class label per atom. Cheap, restrictive (collinear only). |
| Long-range              | none. No Ewald, no charges-to-electrostatics. |
| Production tooling      | full LAMMPS deploy path with parity-tested C++ pair_style. Mature. |

**Most accurate one-sentence placement**: "a 2025-era short-range
equivariant MLIP in the PaiNN/NequIP family, with a learnable empirical
pair prior and a decoupled stress head, focused on practical training and
deployment for magnetic transition-metal alloys."

**What it is good for**: small-DFT-budget projects on multi-element
metallic alloys, especially when active learning is in the loop and
short-range robustness matters; specific magnetic systems where the
collinear-spin label captures the relevant physics.

**What it is not**: a benchmark-leading model, a foundation model, a
universal MLIP, or a novel architecture. It's a competent, opinionated
production system.

## What would actually be novel here

If you wanted to claim a small but real research contribution, the
candidates ranked by novelty:

1. **The detached-feature stress head** as a multi-task gradient-surgery
   pattern. If the surveyed literature truly doesn't have this, it's
   worth a short methods note. Generalizes beyond stress: any auxiliary
   property can be predicted from the same backbone without competing
   for capacity.
2. The combination of a **learnable Born-Mayer prior + learnable
   attractive radial** (vs fixed ZBL + GNN correction). Not fundamentally
   new but worth a 2-paragraph evaluation if anyone asks "does
   learnability of the prior pay off?".
3. Honestly, nothing else.

Things explicitly **not** worth claiming as novel:
- Pair-potential priors in MLIPs.
- Sinc radial basis.
- Equivariant scalar + vector node features.
- Discrete spin labels per atom.
- Per-atom virial normalization.
- Equivariant attention message passing.
- "Sample efficiency from physics priors."

## References

[^dimezbl]: "Prediction of Interatomic Potentials Combining Empirical
    Potential and Graph Neural Networks." *Computers and Artificial
    Intelligence* (2024). https://journals.zeuspress.org/index.php/CAI/article/view/266

[^nequip]: Batzner et al., "E(3)-equivariant graph neural networks for
    data-efficient and accurate interatomic potentials." *Nature
    Communications* 13, 2453 (2022). https://www.nature.com/articles/s41467-022-29939-5

[^nequip-readme]: NequIP code & docs.
    https://github.com/mir-group/nequip

[^mace-readme]: MACE code & docs.
    https://github.com/ACEsuit/mace

[^practical-guide]: Riebesell et al., "A practical guide to machine
    learning interatomic potentials" (2025).
    https://ceder.berkeley.edu/publications/2025_Ryan_MLP-guide.pdf

[^empirical-gnn]: "Prediction of Interatomic Potentials Combining
    Empirical Potential and Graph Neural Networks." See [^dimezbl].

[^hotpp]: "E(n)-Equivariant cartesian tensor message passing interatomic
    potential." Online at PMC.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11366765/

[^gnnff]: Park et al., "Accurate and scalable graph neural network force
    field and molecular dynamics with direct force architecture." *npj
    Computational Materials* (2021).
    https://www.nature.com/articles/s41524-021-00543-3

[^behler-spin]: Eckhoff and Behler, "High-dimensional neural network
    potentials for magnetic systems using spin-dependent atom-centered
    symmetry functions." *npj Computational Materials* 7, 170 (2021).
    https://www.nature.com/articles/s41524-021-00636-z

[^spingnn]: "Spin-dependent graph neural network potential for magnetic
    materials." *Phys. Rev. B* 109, 144426 (2024).
    https://link.aps.org/doi/10.1103/PhysRevB.109.144426

[^spin-univ]: "Spin-informed universal graph neural networks for
    simulating magnetic ordering" (2025).
    https://pubmed.ncbi.nlm.nih.gov/40591595/

[^spookynet]: Unke et al., "SpookyNet: Learning force fields with
    electronic degrees of freedom and nonlocal effects." *Nature
    Communications* 12, 7273 (2021).
    https://www.nature.com/articles/s41467-021-27504-0

[^deepmd-pairtab]: DeePMD-kit documentation, "Interpolation or combination
    with a pairwise potential."
    https://docs.deepmodeling.com/projects/deepmd/en/r2/model/pairtab.html

[^nep-zbl]: Fan et al., "Neuroevolution machine learning potentials" *Phys.
    Rev. B* 104, 104309 (2021); Liu et al. hybrid NEP-ZBL for radiation
    damage in W. https://link.aps.org/doi/10.1103/PhysRevB.104.104309

[^physnet]: Unke and Meuwly, "PhysNet: A Neural Network for Predicting
    Energies, Forces, Dipole Moments, and Partial Charges." *J. Chem.
    Theory Comput.* 15, 3678 (2019).
    https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181

[^yang-hybrid]: Yang et al., "Improving robustness and training efficiency
    of machine-learned potentials by incorporating short-range empirical
    potentials." arXiv:2504.15925 (April 2025).
    https://arxiv.org/abs/2504.15925

[^pair-nn-2022]: "Pairwise interactions for potential energy surfaces and
    atomic forces using deep neural networks." *Computational Materials
    Science* (2022). arXiv:2111.05603.
    https://arxiv.org/abs/2111.05603
