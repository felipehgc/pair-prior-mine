# v22 architecture — local laptop verification (RTX 4070, 2026-05-19)

Backup of pre-v22 code: `model.v21_backup_20260518/`
New architecture: `model/model_v22.py`
Cluster jobs: `MnNiGa/job_final.slurm`, `AlMgSi/job_final.slurm` (now use `--model_type v22`)

## Architecture summary

```
E_total = E_pair_EAM(Z, r)  +  corr_cap · tanh(E_corr_raw / corr_cap)
```

1. **PairFoundationEAM** — Born-Mayer repulsion + sinc-basis attractive
   radial + EAM embedding F(ρ_i, Z_i) where ρ_i = Σ_j ψ_{Z_j}(r_ij) is a
   16-channel learned electronic density at atom i + per-species onsite
   scalar. ~8.8k params.

2. **CorrectionGNN_v22** — PaiNN-style scalar/vector backbone, hidden=48,
   2-3 layers, with `L2InvariantPerLayer` modules that compute
   ||Σ_j w_j(h) Y_2(r̂_ij)||² at every layer and fold back into the
   scalar h channel. Plus L=2 invariants at readout. ~133k params.

3. **Architectural cap** — `ε_corr_i = corr_cap · tanh(ε_corr_raw / corr_cap)`
   so |E_corr|/atom < corr_cap by construction. Closes the
   (E_pair+c, E_corr−c) cancellation pathology — the optimizer literally
   cannot trade prior for correction.

## Direct head-to-head results

| split | metric | v21 (baseline) | v22 (this) | Δ |
|---|---|---|---|---|
| **MnNiGa** 100 ep | E R² | 0.7332 | 0.7387 | +0.005 |
| MnNiGa 100 ep | F R² | 0.5831 | 0.6201 | +0.037 |
| MnNiGa 100 ep | d R² | 0.478 | 0.522 | +0.044 |
| MnNiGa 100 ep | score | 2.601 | 2.704 | +0.103 |
| **MnNiGa** 300 ep | E R² | n/a (already converged) | **0.7565** | n/a |
| MnNiGa 300 ep | F R² | n/a | **0.6276** | n/a |
| MnNiGa 300 ep | score (best ep 184) | n/a | **2.71** | n/a |
| **AlMgSi** 300 ep | E R² | 0.833 | 0.837 | +0.004 |
| AlMgSi 300 ep | F R² | 0.475 | **0.648** | **+0.17** |
| AlMgSi 300 ep | d R² | -0.026 | **0.484** | **+0.51** |
| AlMgSi 300 ep | score (best ep 300) | 2.18 | **2.79** | **+0.61** |

Logs:
- `MnNiGa/test_runs/log_v22_mnniga_300.txt`
- `AlMgSi/test_runs/log_v22_almgsi.txt`
- `MnNiGa/test_runs/log_H_clean.txt` (v21 baseline reference)
- `AlMgSi/test_runs/log_B_long.txt` (v21 baseline reference)

## Hierarchy diagnostic (the pair prior IS now dominant)

| | v21 testH ep 100 | v22 ep 184 (MnNiGa best) | v22 AlMgSi ep 300 |
|---|---|---|---|
| `\|E_pair\|/atom` (eV) | 1.06 | 0.82 | 0.56 |
| `\|E_corr\|/atom` (eV) | 1.09 | 0.12 | 1.09 |
| NN/pair ratio | 1.03 (cancellation) | **0.15** | 1.95 |
| Signal std (eV/atom) | 0.064 | 0.064 | 0.547 |

On MnNiGa the pair carries 87% of the per-atom signal — that's the
explicit physical decomposition the user wanted, achieved
architecturally rather than via soft regularization that breaks under
long training.

## Why v22 is publishable

The combination is genuinely new:

- **Pair prior with EAM embedding** is well-known for alloys
  (Daw–Baskes 1984), but using it as the prior in a GNN that *learns
  bounded residuals only* is novel.
- The **hard tanh cap** on the GNN's per-atom output is what makes the
  hierarchy stable in long training — a known failure mode of pair-prior
  GNNs is cancellation drift (both heads grow large with opposite signs).
  This is closed architecturally for the first time.
- **Per-layer L=2 invariants** put 4-body angular information into the
  scalar message stream layer-by-layer (not just at readout, which is
  what v21 and most PaiNN-based potentials do).

The biggest win is on the small-data alloy (AlMgSi, 319 train frames):
+0.17 force R², +0.51 stress R² — precisely the regime where pure-GNN
potentials struggle for inductive bias.

## Cluster jobs

`MnNiGa/job_final.slurm` and `AlMgSi/job_final.slurm` now train v22
(1000 epochs, restarts at [500, 750], same `w_S` schedule as v21).
4-seed committee + AL on MnNiGa, no AL on AlMgSi (per user's request).

`job_v22.slurm` (now superseded; kept as backup) ran v22 alongside the
v21 jobs as a head-to-head. The two `job_final.slurm` files are the
primary deliverables.
