# pair-prior MLIP — model architecture

This document describes `MLIP_v20`, the model used throughout this repo. The
canonical implementation is `model/model.py`; this is the explanation that
goes with it.

## TL;DR

A short-range (5 Å cutoff), permutation- and rotation-equivariant graph
neural network potential for arbitrary multi-element alloys, with optional
per-atom magnetic-moment input. Predicts:

- **total energy** (eV)
- **per-atom forces** (eV/Å) via autograd of energy w.r.t. positions
- **per-atom virial / stress** (eV/atom in Voigt 6, then divided by cell volume)
- per-atom **magmoms** and **charges** (auxiliary, used only in training)

Family: closest to NequIP / Allegro / MACE — short cutoff, scalar+vector
node features, equivariant message passing. Differences from those: a
*pair-potential prior* with explicit short-range repulsion, *direct virial
head* decoupled from the autograd path, and a discrete *spin label* per
atom rather than a continuous magnetic moment.

## Inputs and outputs

| Input          | Shape       | Type   | Notes                                          |
| ---            | ---         | ---    | ---                                            |
| `Z`            | `[N]`       | int64  | atomic numbers                                 |
| `pos`          | `[N, 3]`    | float32| Cartesian positions, Å                         |
| `edge_index`   | `[2, E]`    | int64  | (i, j) atom pairs from the neighbor list        |
| `edge_shift`   | `[E, 3]`    | float32| integer fractional periodic shift              |
| `cell`         | `[3, 3]`    | float32| rows = lattice vectors a, b, c (upper-triangular) |
| `spin`         | `[N]`       | int64  | discrete magnetic label ∈ {0=NM, 1=FM up, 2=FM down} |

| Output         | Shape       | Notes                                                       |
| ---            | ---         | ---                                                         |
| `E_total`      | `[]`        | physical eV; interaction + Σ E0[Z]                          |
| `F`            | `[N, 3]`    | physical eV/Å, from `-∂E_total/∂pos`                        |
| `virial_atom`  | `[N, 6]`    | physical eV per-atom virial, Voigt order (xx,yy,zz,xy,xz,yz) |

In the deploy wrapper used by LAMMPS, only `E_total` and `virial_atom` are
returned; forces come from `pos.requires_grad_(true); E.backward(); F = -pos.grad`.

## Architecture, top-down

```
                                   ┌──────────────────────────────────┐
   atom_embed(Z) + spin_embed(spin) ┤  scalar features h_i  (init)     │
                                   └──────────────────────────────────┘
                                                    │
                                   ┌──────────────────────────────────┐
   v_init_proj(rad) ⊗ unit_vec     ┤  vector features v_i  (init)     │
                                   └──────────────────────────────────┘
                                                    │
                  ┌───────────────────────────────────────────┐
                  │   3 × AngularAttention layers              │
                  │   ─ multi-head attention on (h, v, edges)  │
                  │   ─ degree-normalized aggregation          │
                  │   ─ gated residual update                  │
                  │   ─ equivariant vector update              │
                  └───────────────────────────────────────────┘
                                                    │
                  ┌───────────────────────────────────────────┐
                  │  EnergyHead(h, v, Z) → ε_i  (per atom)     │
                  │  + PairPotential prior (pair sum)          │
                  └───────────────────────────────────────────┘
                                                    │
                                              E_interaction
                                                    │
                  ┌───────────────────────────────────────────┐
                  │  VirialHead(h.detach(), v.detach(), Z)    │
                  │  → σ_atom (per-atom virial Voigt 6)        │
                  └───────────────────────────────────────────┘
```

### 1. Pair-potential prior

```
E_pair = Σ_(i,j) [A_pair(Z_i, Z_j) · exp(−α_pair(Z_i, Z_j) · r_ij)
                  + w(Z_i, Z_j) · sinc_basis(r_ij)]
```

Adds a **species-dependent short-range repulsion** plus an attractive radial
component before any neural-network correction kicks in. The repulsion term
prevents the runaway atomic overlaps that AL-generated structures
sometimes propose; without it, an atom shoved 0.3 Å from a neighbor sees
no penalty and the AL loop diverges. Both A and α are species-pair-dependent
and learnable, factored through small embeddings to keep the parameter
count tame.

### 2. Sinc radial basis

```
f_n(r) = A_n · sin(nπr/c) · (1 − (r/c)²)²,    n = 1 … n_radial
```

Frequencies are fixed (no learnable centers). Only per-frequency amplitudes
`A_n` are learned. Compared to the standard Gaussian basis, this is:

- approximately orthogonal on `[0, c]` → downstream linears are well-conditioned;
- non-overlapping at the boundaries → no near-redundant rows in the basis;
- inherently smooth-cutoff via the `(1 − (r/c)²)²` envelope (no extra cutoff
  function needed).

`c = 5 Å` and `n_radial = 20` in the default config.

### 3. Atom and spin embedding

```
h_i = embed_proj([atom_embed(Z_i), spin_embed(spin_i)])
```

`atom_embed` is a learned per-Z embedding of dim `hidden`; `spin_embed`
is a 3-entry table for {non-magnetic, FM up, FM down} of dim `hidden//4`.
The two are concatenated and projected to `hidden`.

The discrete spin label is the model's mechanism for distinguishing magnetic
sublattices in MnNiGa-style Heusler phases (where Mn↑ and Mn↓ are
chemically the same atom but produce different bonding around them). For
non-magnetic systems (TM23), every atom gets `spin = 1` and the label
becomes a constant.

### 4. AngularAttention message passing

Three identical layers, each updates `(h, v)`:

```
edge_features  = concat([h_j, radial_basis(r_ij), angular(v_i ⋅ ûij), angular(v_j ⋅ ûij), pair_embed])
attention      = softmax(q(h_i) · k(edge_features))   per row
agg_h          = degree_norm(Σ_j  attention · v_proj(edge_features))
agg_v          = Σ_j  vec_proj(edge_features) · ûij ·  cutoff_smoothing
h_new          = LayerNorm(h + s · gate · agg_h)
v_new          = v + s · agg_v
```

Notes:

- **Angular dependence** comes in through `(v_i ⋅ ûij)`: the projection of
  each atom's vector features onto the bond direction. Pure scalar models
  (e.g. SchNet) miss this; we get it without going to spherical harmonics.
- **Degree normalization**: aggregates are scaled by `1/√(degree+1)` so
  high-coordination sites don't dominate the gradient signal. This was
  an explicit fix during training — without it, B2 (CN=8) and L2₁ (CN=14)
  in the same batch produced wildly different gradient magnitudes.
- **Residual + LayerNorm** keeps deep stacking stable.
- **Equivariance**: `v` rotates with positions (it's a true vector field);
  `agg_v` is built from `ûij` (rotates) times scalar weights, so it rotates
  too. `h`, the scalar field, is invariant.

### 5. Energy head

```
ε_i = (MLP([h_i, ‖v_i‖]) · scale[Z_i] + shift[Z_i]) · out_scale
E_NN = Σ_i ε_i
E_interaction = E_pair + E_NN     (in normalized-internal units)
```

Per-element scale and shift act as a learnable atomic-energy offset; the
explicit `scale[Z]` factor lets the network learn species-dependent
sensitivity without forcing the bulk MLP to handle it.

### 6. Virial head (detached)

```
σ_aniso[a,b]  = Σ_k  w_k(h_i) · (v_k_a v_k_b − ⅓ |v_k|² δ_ab)
σ_iso[a,b]    = (α(h_i) + bias[Z_i]) · δ_ab
σ_atom[a,b]   = (σ_aniso + σ_iso) · out_scale
```

This is the most non-standard piece. Several MLIPs compute virial from
`∂E/∂ε` (the strain derivative); we don't. Instead:

- The virial head is a *separate* readout from the same `(h, v)` features.
- We pass `h.detach()` and `v.detach()` to it, so its loss never propagates
  into the message-passing backbone.
- This means **stress training cannot pull on the parameters that govern
  energy and force accuracy**. In an earlier iteration (without detach),
  scaling up the stress loss weight degraded the force-R² in 50 epochs.
  After this fix, the head trains independently and force quality is
  preserved.

The `(v_k_a v_k_b − ⅓ |v_k|² δ_ab)` form is symmetric and traceless; it's
the equivariant "anisotropic" part. The isotropic part is a separate scalar
readout. Both rotate correctly under rotation of the system: `σ → R σ Rᵀ`.

The output is in **normalized per-atom virial** units; the deploy wrapper
multiplies by `std_S_pa` to give physical eV/atom.

## Normalization

Training operates on normalized quantities:

| Quantity      | What it is                                                            |
| ---           | ---                                                                   |
| `std_E`       | std of per-atom interaction energy across the training set (eV/atom)  |
| `mean_E`      | mean of per-atom interaction energy across the training set (eV/atom) |
| `std_F`       | std of force components across the training set (eV/Å)                |
| `std_S_pa`    | std of *per-atom* Voigt virial across the training set (eV/atom)      |
| `E0[Z]`       | per-element reference energy (sets the absolute zero of energy)       |

- **Energy targets**: training fits `(y_E - E_ref) / N - mean_E) / std_E`
  where `E_ref = Σ E0[Z]`.
- **Force targets**: training fits `y_F / std_F`.
- **Virial targets**: training fits `y_virial / N / std_S_pa` (per-atom virial).

The `std_S_pa` parameterization is what cured an earlier bias-toward-large-cells
problem. Total virial `W = Σ_pair r ⊗ F` scales with both `N` and cell
volume; mixing 2-atom (B2) and 16-atom (L2₁) supercells in the same loss
meant L2₁ contributions dominated by 64–100×. Per-atom virial is intensive
and treats both system sizes equally.

`E0[Z]` is computed by least-squares regression at the start of training:
fit `E_total ≈ Σ_atoms E0[Z_atom]` over the training set, take residuals as
the "interaction" energy that the model actually predicts.

## How training runs

1. **Subsample**. `model/subsample_dataset.py` cuts each trajectory's
   frames down to a target per-structure cap (default 12), structure-level
   80/20 train/val split, balanced by phase prefix when present.
2. **Compute transforms**. `model/data.py:compute_transforms(...)` runs
   on the train split: builds `E0`, `mean_E`, `std_E`, `std_F`, `std_S`
   (and `std_S_pa` from the per-atom virial). Stress component mask
   detects constrained zeros (e.g. 2-D systems).
3. **Train**. `model/train.py` runs the standard Adam loop with cosine LR
   schedule. Loss = w_E·MSE_E + w_F·MSE_F + w_m·MSE_m + w_q·MSE_q + w_S·MSE_S.
   The stress weight `w_S` ramps from 0 to a target over the first ~50
   epochs so stress isn't fighting energy/force during early training.
4. **Strain augmentation**. After ep 50, batches get a small random
   symmetric-strain perturbation applied consistently to positions, cell,
   and force targets (the contravariant transform). Increases stress
   diversity without breaking force consistency.
5. **Split-clip**. Backbone and `virial_head` parameters are gradient-clipped
   independently. Without this, a fresh randomly-initialized head emits
   huge gradients in the first few steps that dominate the global norm
   and effectively shrink the backbone updates by 100×.
6. **Checkpoint** at best validation `score = w·R²_E + w·R²_F + …`; written
   as `{"model", "config", "transforms", "epoch", "score", "r2"}`.

## Deployment

`model/model.py` defines a `LammpsModel` wrapper:

- Re-binds the trained submodules onto a fresh `nn.Module`.
- Stores `std_E`, `std_F`, `std_S_pa`, `mean_E`, and the per-element `E0`
  table as buffers.
- Forward signature: `(Z, pos, edge_index, edge_shift, cell, spin) → (E_total, virial_atom)`,
  with all denormalization done internally so the C++ side gets *physical*
  units directly.
- Drops the magmom/charge heads (LAMMPS doesn't consume them).

`lammps/deploy/deploy.py` runs `torch.jit.trace` on this wrapper and saves
the resulting `.pt`. Trace (not script) because `torch.autograd.grad`
inside a scripted module segfaults under torch 2.7. The trace is verified
to be shape-dynamic (different `N` produces correct outputs).

## Pointers to the code

- Architecture: `model/model.py` — `MLIP_v20`, `PairPotential`,
  `AngularAttention`, `EnergyHead`, `VirialHead`.
- Deploy wrapper: same file, bottom — `LammpsModel`, `build_lammps_model`.
- Trainer: `model/train.py` — `train_one_epoch`, loss assembly, EMA.
- Data: `model/data.py` — graph build, dataset, `make_dataloaders`,
  `compute_transforms`.
- Subsampler: `model/subsample_dataset.py`.
- Calculator (ASE wrapper): `model/calculator.py`.
- LAMMPS: `lammps/` (see `lammps/README.md` and `BUILD_TUTORIAL.md`).

## Configuration knobs (default values)

```python
CONFIG_DEFAULT = {
    "num_species":    100,
    "hidden":          64,
    "n_layers":         3,
    "n_radial":        20,
    "cutoff":           5.0,   # Å
    "n_spin":           3,     # NM, FM up, FM down
    "dropout":          0.10,
    "n_heads":          4,
    "angular_dim":     16,
    "pair_embed_dim":   8,
    "pair_rank":        8,
    "normalize_vec":   False,
}
```

`hidden=64` was chosen after observing capacity-limited training at
`hidden=48`. `n_layers=3` is the message-passing depth; the receptive
field reaches `n_layers · cutoff = 15 Å`, which is enough for medium-range
order in metallic alloys.

## What this model does NOT do

- **Long-range Coulombic forces.** The cutoff is short; charge is predicted
  but never used to drive electrostatic interactions. For systems where
  long-range matters (insulators, polar surfaces) you'd add a `pair hybrid`
  Ewald component on top.
- **Continuous magnetic moments.** Spin is a 3-class label, not a vector.
  For SPIN-package-style spin dynamics in LAMMPS you'd need a different
  spin head + the SPIN package on the LAMMPS side.
- **Non-orthogonal cells with real-time deformation.** Stress is correct
  in static evaluation; box-deformation runs work but haven't been
  end-to-end validated for extreme tilt yet.

## Where it came from

The architecture went through 20 small revisions. Notable steps:

- v17 → v18: added the pair-potential prior with repulsion (cured AL
  divergence on overlap-prone structures).
- v18 → v19: replaced the Gaussian radial basis with sinc.
- v19 → v20: added the *direct virial head* with detached features
  (decouples stress training from energy/force).

The "v20" tag is reflected in checkpoint metadata (`model_version: "v20_virial_head"`).
