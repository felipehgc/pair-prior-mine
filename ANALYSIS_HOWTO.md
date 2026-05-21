# Analysis & figures — where everything is and how to run it

Everything below lives at the **repo root** unless noted. Run all commands from
the repo root (`/.../pair-prior`).

---

## 1. The scripts

| Script | Produces | Needs |
|---|---|---|
| `run_all_parity.sh` | **one-shot:** all dumps (§3) + all figures | the checkpoints / frozen models in place |
| `parity_pairprior_deepmd.py` | `parity_figs/<ds>_{energy,force,stress}.png` + `MnNiGa_aux_mq.png` | the `.out`/`.npz` prediction dumps (§3) |
| `ternary_magmom_MnNiGa.py` | `parity_figs/MnNiGa_ternary_magmom.png` | `properties/MnNiGa/parity_all.npz` |
| `tools/parity.py` | pair-prior prediction dumps (`parity_*.npz`) | a checkpoint + dataset CSV |
| `dp test` (DeepMD CLI) | DeepMD prediction dumps (`*_par.*.out`) | a frozen `.pb` model + deepmd-format data |

Companion writeup: `MODEL_COMPANION.md`.

### Run the figures (once the dumps in §3 exist)

```bash
python parity_pairprior_deepmd.py     # all parity panels -> parity_figs/
python ternary_magmom_MnNiGa.py       # Mn-Ni-Ga ternary  -> parity_figs/
```

Both write PNGs into `parity_figs/`. Colours: **train = dodgerblue, val =
tomato**. Each parity figure is two panels — **left = DeepMD, right =
pair-prior** — with the y=x line and per-split R² in the legend.

---

## 2. Where the inputs live

| Dataset | Pair-prior checkpoint | Pair-prior CSV (`--csv`) | `--data_dir` | DeepMD frozen model | DeepMD data (`-s`) |
|---|---|---|---|---|---|
| Fe | `properties/Fe/checkpoint.pt` | `benchmark/TM23/Fe/data/dataset.csv` | `benchmark/TM23/Fe` | `comparison/deepmd/Fe/frozen_model.pb` | `comparison/data/Fe/deepmd/{train,val}` |
| Cu | `properties/Cu/checkpoint.pt` | `benchmark/TM23/Cu/data/dataset.csv` | `benchmark/TM23/Cu` | `comparison/deepmd/Cu/frozen_model.pb` | `comparison/data/Cu/deepmd/{train,val}` |
| AlMgSi | `properties/AlMgSi/checkpoint.pt` | `AlMgSi/data/dataset_clean.csv` | `AlMgSi` | `comparison/deepmd/AlMgSi/frozen_model.pb` | `comparison/data/AlMgSi/deepmd/{train,val}` |
| MnNiGa | `properties/MnNiGa/checkpoint.pt` | `MnNiGa/data/dataset_subsampled.csv` | `MnNiGa` | `comparison/deepmd/MnNiGa/frozen_model.pb` | `comparison/data/MnNiGa/deepmd/{train,val}` |

> **MnNiGa uses `dataset_subsampled.csv`** (582 train / 80 val) — that is the CSV
> the checkpoint was trained on (`properties/MnNiGa/config.yaml`). `dataset_clean.csv`
> (1221/169) is a *different, larger* split and would not match the checkpoint.
> For the ternary's "entire dataset" view you may still run `--split all` on
> `dataset_clean.csv` to maximise composition coverage (predictions are valid on
> unseen frames); the parity *plots*, however, must use `dataset_subsampled.csv`.

The dataset CSV has a `split` column (`train`/`val`); `tools/parity.py
--split` selects rows from it. The DeepMD data dirs are already split into
`train/` and `val/` subfolders (AlMgSi is multi-system: `sys_000 … sys_052`,
which `dp test -s <dir>` discovers automatically).

---

## 3. Tutorial — regenerating train & val predictions from the checkpoints

This is the step that fills `properties/<ds>/parity_{train,val}.npz` (pair-prior)
and `comparison/deepmd/<ds>/{train,val}_par.*.out` (DeepMD). The figure scripts
read those. Do it once per dataset; **train** and **val** are separate runs.

### 3a. Pair-prior — `tools/parity.py`

`tools/parity.py` loads a checkpoint (`state_dict {model, config, transforms}`),
runs it on the chosen split, and writes physical-unit predictions + ground truth
to an `.npz` (plus a `.csv` summary with MAE/RMSE/R²).

```bash
# --- Fe, training split ---
python tools/parity.py \
    --ckpt     properties/Fe/checkpoint.pt \
    --csv      benchmark/TM23/Fe/data/dataset.csv \
    --data_dir benchmark/TM23/Fe \
    --split    train \
    --out      properties/Fe/parity_train.npz \
    --device   cpu --batch_size 16

# --- Fe, validation split ---  (same, but --split val / --out parity_val.npz)
python tools/parity.py \
    --ckpt properties/Fe/checkpoint.pt --csv benchmark/TM23/Fe/data/dataset.csv \
    --data_dir benchmark/TM23/Fe --split val \
    --out properties/Fe/parity_val.npz --device cpu --batch_size 16
```

Swap the four paths from the table in §2 for Cu / AlMgSi / MnNiGa. Use
`--device cuda` if you have a GPU (much faster — the CPU autograd pass over the
big TM23 cells is the slow part). For the **ternary** you also need the full set:

```bash
python tools/parity.py --ckpt properties/MnNiGa/checkpoint.pt \
    --csv MnNiGa/data/dataset_clean.csv --data_dir MnNiGa \
    --split all --out properties/MnNiGa/parity_all.npz --device cpu
```

Useful flags: `--limit N` (only first N frames, for a quick smoke test);
`--num_workers 0` (keep at 0 for the bundled `.npz` datasets).

Each run prints a table like:

```
Quantity         n         MAE        RMSE        R²    unit
E per atom    2700    0.0079      0.0108     0.9938   eV/atom
F           143100    0.0500      0.0900     0.99..   eV/Å
...
```

### 3b. DeepMD — `dp test`

DeepMD ships its own tester. Point it at the frozen `.pb` and a deepmd-format
data dir; `-d <prefix>` sets the output filename prefix; `-n` caps the number of
frames (set it large to test all).

```bash
cd comparison/deepmd/Fe

# training split  -> train_par.{e,e_peratom,f,v,v_peratom}.out
dp test -m frozen_model.pb \
        -s /ABS/PATH/pair-prior/comparison/data/Fe/deepmd/train \
        -n 100000 -d train_par

# validation split -> val_par.*.out
dp test -m frozen_model.pb \
        -s /ABS/PATH/pair-prior/comparison/data/Fe/deepmd/val \
        -n 100000 -d val_par
cd -
```

Output column layout (what the figure script reads):

| file | columns |
|---|---|
| `*_par.e_peratom.out` | `true_E/atom  pred_E/atom` |
| `*_par.f.out` | `Fx Fy Fz` (true) ‖ `Fx Fy Fz` (pred) |
| `*_par.v.out` | 9 virial comps (true) ‖ 9 (pred), units **eV** |

> Note: the DeepMD stress panel is plotted as the raw **virial** (eV); the
> pair-prior stress is converted to virial as `−S·V` to match. Different models
> use their own train/val splits, so the two panels are independent point clouds
> (each shows that model's own fit quality), not a paired one-to-one comparison.

### 3c. Then make the figures

```bash
python parity_pairprior_deepmd.py
python ternary_magmom_MnNiGa.py
```
