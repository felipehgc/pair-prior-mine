# Where this model sits in the MLIP landscape

A positioning document for `MLIP_v20` (the `pair-prior` model). It answers
one question: **given a job, when would you reach for this model instead of
MACE, DeepMD-kit, or something else — and when would you not?**

This is the third of three companion docs and deliberately does *not*
repeat them:

- `MODEL.md` — what the architecture *is* (read this first if you want internals).
- `NOVELTY.md` — whether any of it is a *research contribution* (honest answer: mostly no).
- **`LANDSCAPE.md`** (this file) — how it *compares* to the field and when to *use* it.

Where a point here would just paraphrase `NOVELTY.md`, it's cut and cross-referenced instead.

---

## 1. The landscape in one map

Modern MLIPs cluster into a few families. The axes that actually matter for
choosing one are: **how they encode geometry** (invariant descriptors vs.
equivariant features vs. full spherical-harmonic tensors), **how they get
forces and stress** (autograd of a conservative energy vs. direct heads),
**whether they are trained per-project or pretrained universal**, and
**what physics is bolted on** (long-range electrostatics, spin, empirical
priors).

| Family | Geometry encoding | Representative models | Forces/stress | Typical use |
| --- | --- | --- | --- | --- |
| Invariant descriptors | hand-crafted or learned scalar descriptors | Behler–Parrinello SF [^bp2007], GAP [^gap], MTP [^mtp], ACE [^ace], NEP [^nep-zbl], SchNet [^schnet], **DeepMD `se_e2_a`** [^deepmd] | autograd | fast, mature, per-project |
| Equivariant scalar+vector | Cartesian vector node features | PaiNN [^painn], SO3krates, **this model** | autograd (E,F), direct head (S) | mid-cost, per-project |
| Equivariant SH tensors | full spherical-harmonic tensors | NequIP [^nequip], Allegro [^allegro], **MACE** [^mace], Equiformer-v2 | autograd | highest in-distribution accuracy, per-project |
| Direct-force / non-conservative | any backbone, force is a head | GNNFF [^gnnff], HotPP [^hotpp] | direct head | fast inference, no energy conservation guarantee |
| Foundation / universal | equivariant SH tensors, pretrained on large DBs | MACE-MP-0 [^macemp], CHGNet [^chgnet], M3GNet [^m3gnet], SevenNet, ORB, MatterSim, DPA-2 [^dpa2] | autograd | zero-/few-shot across the periodic table |
| Hybrid empirical + ML | ML correction on an empirical pair baseline | ZBL+{NequIP,MACE,NEP}, DimezblNet [^dimezbl], Yang 2025 [^yang], **this model** | autograd | short-range robustness, small data |

This model appears twice on purpose: it is a **PaiNN-family equivariant
scalar+vector potential** *and* a **hybrid with a learnable empirical pair
prior**. That intersection — equivariant GNN backbone + jointly-trained,
per-species-pair-learnable Born–Mayer prior + a decoupled stress head — is
its actual position. None of the three ingredients is novel alone (see
`NOVELTY.md`); the combination, tuned for small-DFT-budget magnetic alloy
work, is the niche.

**One-sentence placement:** a 2025-era short-range equivariant MLIP in the
PaiNN/NequIP lineage, carrying a learnable empirical pair prior and a
detached-feature stress head, built for practical training and LAMMPS
deployment on magnetic transition-metal alloys.

---

## 2. Head-to-head: this model vs. MACE

MACE [^mace] is the closest "serious" comparison — same equivariant GNN
philosophy, same E/F/S targets, also LAMMPS-deployable.

### Architecture

| | This model | MACE |
| --- | --- | --- |
| Equivariance | Cartesian scalar + vector (ℓ ≤ 1 equivalent) | full spherical-harmonic tensors, higher body order via ACE-style products |
| Message passing | 3× attention layers, angular term from `v·û` | 2 layers, high body-order messages per layer |
| Expressivity | lower — cannot represent arbitrary ℓ≥2 angular correlations | higher — that's MACE's whole point |
| Energy baseline | **learnable per-species-pair Born–Mayer + attractive radial prior** | per-element E0 only (optional ZBL via LAMMPS hybrid/overlay) |
| Stress | **direct equivariant head, inputs detached from backbone** | autograd of energy w.r.t. strain |
| Spin | **discrete 3-class per-atom label** (NM / FM↑ / FM↓) | none built in |
| Aux outputs | per-atom magmom + charge heads (training-time signal) | none |

### What each does better

**MACE is more accurate in-distribution.** Full SH tensors capture angular
correlations this model structurally cannot. On a standard benchmark with
adequate data, expect MACE to win on energy/force MAE. `NOVELTY.md` says
this plainly: this model "would not be expected to outperform NequIP / MACE
/ Allegro on standard benchmarks."

**This model has three things MACE lacks out of the box:**

1. **A jointly-trained, learnable short-range prior.** MACE's ZBL option is
   a fixed-by-atomic-number formula layered at deploy time; here the
   repulsion *and* an attractive radial term are learned per species pair,
   jointly with the backbone. This matters most for active-learning loops
   where rollouts propose sub-1.5 Å separations the training set never
   sampled — the prior keeps the energy surface sane there. See
   `NOVELTY.md` §0–1 for the honest "is this novel" verdict (it's a
   refinement, not an invention) and the small-data discussion.
2. **A built-in collinear spin label.** For Heusler-type magnetic alloys
   (MnNiGa) where Mn↑ and Mn↑↓ are the same element with different local
   bonding, MACE needs the magnetic state baked into the data implicitly or
   a custom modification; here it's a first-class input.
3. **Stress training that cannot harm force accuracy.** Because the virial
   head reads `h.detach(), v.detach()`, the stress loss never flows into
   the backbone. With MACE's autograd stress, energy/force/stress share all
   parameters and the loss weights genuinely trade off. This model's
   decoupling is its one mild design distinction (`NOVELTY.md` §4) — but
   note the flip side in §6 below.

### Training cost and data

This model is **cheaper to train** (no SH tensor products, `hidden=64`,
3 layers) and is designed for **small DFT budgets** — the subsample →
compute-transforms → train pipeline assumes a few thousand frames, not
millions. MACE scales to large datasets and benefits from them more.

### The foundation-model fork

This is the biggest practical divergence. **MACE has MACE-MP-0** [^macemp]
— a universal pretrained checkpoint covering most of the periodic table,
usable zero-shot or as a fine-tuning start. This model has **no pretrained
weights**; every system is trained from scratch. If your project can lean
on universal pretraining, that is a strong reason to pick MACE and it has
no answer here.

### Deployment

Both deploy to LAMMPS. This model ships a parity-tested C++ `pair_style`
(`lammps/`, `LAMMPS_DEPLOY.md`) and a `torch.jit.trace` path (trace, not
script — autograd inside a scripted module segfaults under torch 2.7; see
`MODEL.md`). MACE's LAMMPS integration is more widely used and battle-tested
across more groups. Call deployment maturity a wash-to-slight-MACE-edge.

---

## 3. Head-to-head: this model vs. DeepMD-kit

DeepMD-kit [^deepmd] (`se_e2_a` descriptor, the variant used in `comparison/`)
represents the **invariant-descriptor** branch.

### Architecture

| | This model | DeepMD `se_e2_a` |
| --- | --- | --- |
| Geometry encoding | equivariant scalar + vector GNN | smooth-edition invariant descriptor + fitting net |
| Equivariance | yes (vector features rotate) | no — invariance only, achieved by descriptor construction |
| Angular information | `v·û` projection in message passing | embedded in the two-body `se_e2_a` descriptor |
| Stress | direct detached head | autograd virial |
| Spin / magnetism | discrete spin label | none in `se_e2_a` (DeepSPIN is a separate model variant) |
| Data layout | one CSV indexing NPZ frames, any composition mixed | **one "system" folder per fixed composition** (`type.raw` + `set.NNN/`) |

### What each does better

**DeepMD is faster and more mature in production.** `se_e2_a` is a
well-optimized descriptor; DeepMD-kit has years of LAMMPS integration,
a large user base, GPU kernels, and DPA-2 [^dpa2] as a foundation-model
path. For a non-magnetic alloy with a decent dataset and an existing
DeepMD pipeline, there is little reason to switch.

**This model's advantages over `se_e2_a`:**

- **Equivariance.** `se_e2_a` is invariant; equivariant vector features are
  a more data-efficient way to encode directionality (the NequIP result,
  [^nequip]). For small DFT budgets this is the single biggest theoretical
  edge.
- **Spin.** `se_e2_a` has no magnetic degree of freedom. Magnetic Heusler
  work needs DeepSPIN or a different tool; here spin is native.
- **The learnable pair prior**, same argument as vs. MACE.
- **Data ergonomics for heterogeneous compositions.** DeepMD's system
  layout requires one folder per unique composition — the HEA dataset
  (AlFeCoNiCu) explodes into ~750 train + ~235 val system folders, and
  getting the export right is fiddly (it caused the `AssertionError`
  that `comparison/data_export.py::write_deepmd_systems` now fixes). This
  model's CSV-of-frames layout mixes arbitrary compositions in one file
  with no such bookkeeping.

### The DPA-2 caveat

DeepMD is not frozen at `se_e2_a`. **DPA-2** [^dpa2] is DeepMD's attention-
based, multi-task, foundation-model-capable descriptor — much closer in
spirit to the equivariant-GNN frontier and pretrainable across datasets.
A fair "DeepMD vs. this model" in 2026 should acknowledge that the DeepMD
*ecosystem* now spans from cheap `se_e2_a` to DPA-2; the comparison in
`comparison/` uses `se_e2_a` because it's the simplest faithful baseline,
not because it's DeepMD's ceiling.

---

## 4. Briefly: the rest of the field

**NequIP** [^nequip] — the model that established the equivariance →
data-efficiency result (claimed ~3 orders of magnitude fewer training
configs than invariant predecessors). Same family as this model but with
full SH tensors. Strictly more expressive; slower. This model is best read
as "a cheaper, Cartesian, attention-flavored cousin of NequIP with a pair
prior and a spin label stapled on."

**Allegro** [^allegro] — strictly-local (no message passing across the
graph beyond the cutoff), which makes it embarrassingly parallel and
excellent for very large MD. This model *does* pass messages (3 layers ≈
15 Å receptive field), so it captures medium-range order Allegro can't,
but it won't scale to billion-atom MD the way Allegro does.

**GAP / MTP / ACE / NEP** [^gap][^mtp][^ace][^nep-zbl] — the classical
descriptor potentials. Extremely fast, extremely mature, very strong for
single-element and simple-alloy non-magnetic systems. NEP in particular is
GPU-native and trains in minutes. If your system is non-magnetic and you
want the fastest possible train-and-deploy loop, these beat this model on
every practical axis except equivariance-driven data efficiency and spin.

**Foundation models — MACE-MP-0, CHGNet, M3GNet, SevenNet, ORB,
MatterSim, DPA-2** [^macemp][^chgnet][^m3gnet][^dpa2] — pretrained on
large materials databases, usable zero-shot or fine-tuned. This is the
category with **no analogue here**. If "I don't want to run DFT at all" or
"I want a sane starting point across the periodic table" is on the table,
a foundation model is the answer and this model is not competing.

**GNNFF / HotPP** [^gnnff][^hotpp] — direct-force / direct-virial models.
HotPP is the nearest precedent for "predict the virial from a head, not
from autograd." This model is a *partial* member of this club: energy and
forces are conservative (autograd), but the **stress** comes from a direct
head. So it sits between fully-conservative (MACE, NequIP) and
fully-direct (GNNFF) — a deliberate split, see §6.

---

## 5. When to use this model — and when not to

The load-bearing section. Concrete decision guidance.

### Reach for this model when

- **The system is a magnetic transition-metal alloy with collinear order**
  (FM↑/FM↓ sublattices — MnNiGa-type Heuslers). The discrete spin label is
  a native, cheap mechanism most general-purpose MLIPs lack.
- **The DFT budget is small** (hundreds–few-thousand frames) and you want
  equivariance's data efficiency without paying MACE/NequIP's SH-tensor
  cost.
- **Active learning is in the loop.** The learnable short-range prior is
  specifically there to stop AL rollouts from diverging when they propose
  unphysically short bonds — the failure mode `NOVELTY.md` §1 and the
  DimezblNet argument describe.
- **The dataset mixes many compositions** (HEAs, broad alloy sweeps). The
  CSV-of-frames layout handles this with no per-composition bookkeeping;
  DeepMD's system layout actively fights you here.
- **You need a from-scratch, self-contained, LAMMPS-deployable potential**
  for a specific project and don't want a foundation-model dependency.

### Use something else when

- **You want best-in-class in-distribution accuracy and have the data/compute**
  → MACE or NequIP (full SH tensors).
- **You want universal / zero-shot / pretrained coverage** → MACE-MP-0,
  CHGNet, MatterSim, DPA-2. This model has no pretrained weights.
- **Long-range electrostatics matter** (insulators, polar surfaces, ionic
  liquids) → not this model. Charge is predicted as an auxiliary target but
  *never drives an Ewald sum*; there is no long-range term at all
  (`MODEL.md` "What this model does NOT do"). Use a model with explicit
  electrostatics or layer one on.
- **You need non-collinear or thermal spin dynamics** → the 3-class label
  is collinear-only. Use DeepSPIN, SpinGNN [^spingnn], or a continuous-
  moment model.
- **The system is non-magnetic and you want the fastest train/deploy loop**
  → NEP, ACE, or MTP will be faster end-to-end with comparable accuracy.
- **You need billion-atom MD throughput** → Allegro's strict locality wins.
- **You already have a working DeepMD or MACE pipeline** for a non-magnetic
  system → switching costs aren't worth it; the edges above are real but
  incremental, not categorical.

---

## 6. Disadvantages, stated bluntly

No soft-pedalling. These are real and several were observed empirically in
this repo's own runs (`tools/`, `properties/`, `comparison/`).

- **Lower expressivity ceiling.** Cartesian scalar+vector features cannot
  represent arbitrary ℓ≥2 angular correlations. MACE/NequIP can. On a
  level-playing-field benchmark this model is expected to lose on E/F MAE.

- **The stress head is hydrostatic-only on symmetric cells.** Observed
  directly in `tools/properties.py` runs: under uniaxial strain the model
  returns σ_xx ≈ σ_yy ≈ σ_zz, and under shear σ_xy ≈ 0. The detached
  virial head, trained on TM23-style near-equilibrium thermal MD, only
  learned the isotropic (pressure) component well. Consequence: **bulk
  modulus is meaningful, individual cubic elastic constants C11/C12/C44
  are not** from the current checkpoints. The `isotropic_only` diagnostic
  flag in `properties.py` exists to catch this. The stress *magnitude* is
  also ~5× low relative to the EOS-derived bulk modulus.

- **The detach is a double-edged sword.** Decoupling the stress head
  (`NOVELTY.md` §4) guarantees stress training can't *harm* energy/force —
  but it also means stress quality is capped by whatever representation the
  backbone happened to learn from E/F alone. The head cannot ask the
  backbone for better features. The hydrostatic-only limitation above is
  partly this: the backbone never had a reason to encode shear response.

- **No long-range physics.** Covered in §5. Worth repeating because it's a
  hard boundary, not a tuning issue.

- **No foundation-model story.** Every system trained from scratch. No
  pretrained checkpoint, no transfer, no zero-shot.

- **Not yet benchmarked head-to-head.** The empirical comparison against
  MACE and DeepMD on the same four datasets (Cu, Fe, MnNiGa, AlFeCoNiCu)
  is *set up* in `comparison/` but **not run** — the MACE and DeepMD models
  aren't trained yet. Every accuracy claim above is architectural
  reasoning, not measured. When `comparison/mace/<name>/properties/` and
  `comparison/deepmd/<name>/properties/` are populated and compared against
  `properties/<name>/`, this section should be revised with numbers. Until
  then, treat the comparison as qualitative.

- **Elastic-constant evaluation has structural caveats** independent of the
  model: the MnNiGa frame in use is tetragonal martensite (not cubic
  Heusler), and AlFeCoNiCu needs an SQS supercell for proper cubic C_ij.
  See `comparison/TODO.txt`. This is a property-pipeline limitation, but it
  bounds what the comparison can currently say.

---

## 7. Summary table

| Axis | This model | MACE | DeepMD `se_e2_a` | NequIP | Foundation models |
| --- | --- | --- | --- | --- | --- |
| Geometry encoding | equiv. scalar+vector | equiv. SH tensors | invariant descriptor | equiv. SH tensors | equiv. SH tensors |
| In-distribution accuracy | good | best-in-class | good | best-in-class | varies (broad, not tuned) |
| Data efficiency | high (equivariance) | high | moderate | highest | n/a (pretrained) |
| Training cost | low | high | low–moderate | high | n/a / fine-tune |
| Empirical pair prior | **learnable, joint** | fixed ZBL (optional) | fixed pairtab (optional) | fixed ZBL (optional) | none |
| Spin / magnetism | **native collinear label** | none | none (`se_e2_a`) | none | none |
| Stress mechanism | direct detached head | autograd | autograd | autograd | autograd |
| Long-range electrostatics | none | none | none | none | none |
| Pretrained / universal | no | **MACE-MP-0** | **DPA-2** | no | **yes** |
| Heterogeneous-composition data | trivial (CSV of frames) | easy (extxyz) | awkward (per-comp folders) | easy | n/a |
| LAMMPS deploy maturity | working, parity-tested | widely used | most mature | working | varies |

---

## References

[^bp2007]: Behler & Parrinello, "Generalized neural-network representation
    of high-dimensional potential-energy surfaces." *Phys. Rev. Lett.* **98**,
    146401 (2007).

[^schnet]: Schütt et al., "SchNet — a deep learning architecture for
    molecules and materials." *J. Chem. Phys.* **148**, 241722 (2018).

[^painn]: Schütt, Unke & Gastegger, "Equivariant message passing for the
    prediction of tensorial properties and molecular spectra (PaiNN)."
    *ICML* (2021). arXiv:2102.03150.

[^nequip]: Batzner et al., "E(3)-equivariant graph neural networks for
    data-efficient and accurate interatomic potentials." *Nature
    Communications* **13**, 2453 (2022).
    https://www.nature.com/articles/s41467-022-29939-5

[^allegro]: Musaelian et al., "Learning local equivariant representations
    for large-scale atomistic dynamics (Allegro)." *Nature Communications*
    **14**, 579 (2023).

[^mace]: Batatia et al., "MACE: Higher order equivariant message passing
    neural networks for fast and accurate force fields." *NeurIPS* (2022).
    arXiv:2206.07697. https://github.com/ACEsuit/mace

[^macemp]: Batatia et al., "A foundation model for atomistic materials
    chemistry (MACE-MP-0)." arXiv:2401.00096 (2023/2024).

[^deepmd]: Wang, Zhang, Han & E, "DeePMD-kit: A deep learning package for
    many-body potential energy representation and molecular dynamics."
    *Comput. Phys. Commun.* **228**, 178 (2018). `se_e2_a` smooth edition:
    Zhang et al., *NeurIPS* (2018). https://github.com/deepmodeling/deepmd-kit

[^dpa2]: Zhang et al., "DPA-2: a large atomic model as a multi-task
    learner." arXiv:2312.15492 (2023/2024).

[^gap]: Bartók, Payne, Kondor & Csányi, "Gaussian approximation potentials:
    the accuracy of quantum mechanics, without the electrons." *Phys. Rev.
    Lett.* **104**, 136403 (2010).

[^mtp]: Shapeev, "Moment tensor potentials: a class of systematically
    improvable interatomic potentials." *Multiscale Model. Simul.* **14**,
    1153 (2016).

[^ace]: Drautz, "Atomic cluster expansion for accurate and transferable
    interatomic potentials." *Phys. Rev. B* **99**, 014104 (2019).

[^nep-zbl]: Fan et al., "Neuroevolution machine learning potentials (NEP)."
    *Phys. Rev. B* **104**, 104309 (2021).

[^gnnff]: Park et al., "Accurate and scalable graph neural network force
    field and molecular dynamics with direct force architecture." *npj
    Comput. Mater.* **7**, 73 (2021).
    https://www.nature.com/articles/s41524-021-00543-3

[^hotpp]: "HotPP: an E(n)-equivariant Cartesian tensor message passing
    interatomic potential." https://pmc.ncbi.nlm.nih.gov/articles/PMC11366765/

[^chgnet]: Deng et al., "CHGNet as a pretrained universal neural network
    potential for charge-informed atomistic modelling." *Nature Machine
    Intelligence* **5**, 1031 (2023).

[^m3gnet]: Chen & Ong, "A universal graph deep learning interatomic
    potential for the periodic table (M3GNet)." *Nature Computational
    Science* **2**, 718 (2022).

[^dimezbl]: "Prediction of Interatomic Potentials Combining Empirical
    Potential and Graph Neural Networks (DimezblNet)." *Computers and
    Artificial Intelligence* (2024).
    https://journals.zeuspress.org/index.php/CAI/article/view/266

[^yang]: Yang et al., "Improving robustness and training efficiency of
    machine-learned potentials by incorporating short-range empirical
    potentials." arXiv:2504.15925 (2025).

[^spingnn]: "Spin-dependent graph neural network potential for magnetic
    materials." *Phys. Rev. B* **109**, 144426 (2024).
