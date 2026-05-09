# LAMMPS integration for pair-prior MLIP

Status: deploy pipeline working end-to-end. C++ pair_style still to write.

## Layout

```
lammps/
├── README.md                  ← this file
├── deploy/
│   ├── deploy.py              ← export a trained checkpoint to a traced .pt
│   └── built/                 ← outputs (.pt files; gitignored)
├── tests/
│   └── compare_with_eager.py  ← assert traced .pt agrees with the eager
│                                MLIP_v20 reference on real cached frames
├── pair_pairprior/
│   ├── pair_pairprior.h        ← LAMMPS pair_style header
│   ├── pair_pairprior.cpp      ← settings/coeff/init/compute (compute is a stub)
│   ├── pairpriorplugin.cpp     ← lammpsplugin_init registration
│   ├── CMakeLists.txt          ← builds the plugin .so against bundled LAMMPS + libtorch
│   └── build/                  ← gitignored; cmake build output
├── external/
│   ├── fetch_lammps.sh        ← one-shot script to (re)download the pinned LAMMPS source
│   ├── .gitignore             ← keeps the source tree out of git
│   └── lammps/                ← LAMMPS stable_29Aug2024_update3 (435 MB, gitignored)
└── examples/
    └── (next: minimal in.* input scripts — a 2-atom static call, then real MD)
```

## Phasing

| Phase | What                                                         | Status |
| ---   | ---                                                          | ---    |
| 1     | Folder structure                                             | ✅      |
| 2     | TorchScript wrapper + `deploy.py`                            | ✅      |
| 3     | `tests/compare_with_eager.py` — traced vs. eager parity      | ✅      |
| 4     | Pin and download LAMMPS source into `external/lammps/`       | ✅      |
| 5.1   | C++ pair_style scaffold compiles (stub `compute()`)          | ✅      |
| 5.2   | LAMMPS built with PLUGIN; plugin loads; `in.smoke` runs      | ✅      |
| 5.3   | Real `compute()` against the traced .pt — E/F/σ match Python | ✅      |
| 5.4   | Multi-frame parity harness (`compare_with_lammps.py`) passes | ✅      |

## Pinned LAMMPS version

`stable_29Aug2024_update3` (released by Sandia, summer 2024). To (re)fetch
on a new machine:

```bash
cd lammps/external && ./fetch_lammps.sh
```

The script downloads ~140 MB tarball, unpacks to `lammps/external/lammps/`
(~435 MB), and prints the resolved version line from `src/version.h` so you
can confirm what you got.

Why this tag: it's what `pair_allegro` and `pair_mace` currently target, so
their build instructions (which we'll fork in Phase 5) translate directly.

## Building the pair_pairprior plugin (Phase 5.1 deliverable)

```bash
cd lammps/pair_pairprior
mkdir -p build && cd build
cmake .. -DTORCH_INSTALL_PREFIX=$(python -c \
    'import torch, os; print(os.path.dirname(torch.__file__))')
cmake --build . -j 4
# → lammps_pairprior.so  (~640 KB)
```

The CMakeLists deliberately does *not* call `find_package(Torch)`. The torch
config calls `enable_language(CUDA)` unconditionally for GPU torch builds,
which fails on any system where `nvcc` doesn't have a clean line to a host
compiler (most clusters). Linking against `libtorch.so` / `libtorch_cpu.so`
/ `libc10.so` directly sidesteps the problem and works cluster-side too.

If MPI isn't installed system-wide, CMake falls back to LAMMPS's bundled
`STUBS/mpi.h` (header-only) so the plugin still compiles for development.

### CXX11 ABI

Torch wheels from PyPI are built with `_GLIBCXX_USE_CXX11_ABI=0`. Source
builds (e.g. conda-forge) use `=1`. Our CMakeLists defaults to 0; override
with `-DTORCH_GLIBCXX_ABI=1` if your torch was built that way. Wrong choice
typically links cleanly but crashes on first `torch::jit::load` call.

## Phase 5.2 deliverable — LAMMPS + plugin smoke test

```bash
# Build LAMMPS (serial) with PLUGIN package (one-shot, ~5 min):
cd lammps/external/lammps
mkdir -p build && cd build
env PATH="/usr/bin:/bin" CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    cmake ../cmake -DCMAKE_BUILD_TYPE=Release \
                   -DBUILD_MPI=off -DBUILD_SHARED_LIBS=on \
                   -DPKG_PLUGIN=on
env PATH="/usr/bin:/bin" cmake --build . -j 4

# Run the smoke test:
env PATH="/usr/bin:/bin" ./lmp -in ../../../examples/in.smoke
```

Confirmed working output:

```
pair_pairprior: loaded TorchScript model 'vhclean.pt' (cutoff = 5 Å)
   Step    Temp    TotEng    PotEng    KinEng    Press
      0    300     0.0387    0         0.0387    647.18
      1    300     0.0387    0         0.0387    647.18
```

`PotEng = 0` is the milestone-1 stub behavior — the plugin loaded, the
TorchScript model deserialized via `torch::jit::load`, and LAMMPS ran a
timestep without crashing. `compute()` actually invoking the model is
Phase 5.3.

## Phase 5.3 deliverable — full E/F/σ parity vs Python

The 2-atom MnGa smoke test now produces:

```
   Step          Temp     TotEng     PotEng     KinEng        Press        Pxx       Pyy       Pzz
       0          0    0.36500561 0.36500561        0    -1085.1977 -1085.198 -1085.198 -1085.198
```

vs the Python reference (LammpsModel called eager with the same edges):

```
E = 0.365006 eV          (LAMMPS Δ = 7e-7 eV)
σ_diag = -6.78e-4 eV/Å³  (LAMMPS = -6.78e-4 eV/Å³ = -1085 bar; isotropic)
|F|max = 1e-6 eV/Å       (zero by symmetry; LAMMPS reports same to noise)
```

Three subtle gotchas that cost time during 5.3 (worth flagging for any future
ML-pair_style work):

1. **Pass `nlocal` atoms with fractional `edge_shift`, NOT `nall` atoms with
   `edge_shift = 0`.** The training graph format uses local indices in both
   columns of `edge_index` and a Voigt-3 fractional shift. Passing ghosts as
   independent atoms makes the model's `ea_nn.sum()` double-count them.
2. **Disable `vflag_fdotr` via `no_virial_fdotr_compute = 1` in the
   constructor.** LAMMPS's default virial shortcut `Σ x_i·f_i` is only
   correct for pair-decomposable energies; for a many-body model it
   produces a wrong (and asymmetric) virial even when forces are right.
3. **`pair_style pairprior` requires `atom_modify map yes`** so we can map
   ghost tags back to their owning local index. The init-time check
   surfaces this with a clear error message.

## Phase 5.4 deliverable — full multi-frame parity

```bash
cd lammps/tests
python compare_with_lammps.py \
    --ckpt   ../../benchmark/stress_diagnostic/outputs/finetuned_VHCLEAN.pt \
    --plugin ../pair_pairprior/build/lammps_pairprior.so \
    --lmp    ../external/lammps/build/lmp \
    --model  ../deploy/built/vhclean.pt \
    --frames ../../MnNiGa/data/cached_clean/system_0_step1.npz \
             ../../MnNiGa/data/cached_clean/system_0_step34.npz \
             ../../MnNiGa/data/cached_clean/system_0_step50.npz \
             ../../MnNiGa/data/cached_clean/system_57_step1.npz \
             ../../MnNiGa/data/cached_clean/system_209_step10.npz
```

Each frame: write a LAMMPS data file, generate a minimal input that loads
the plugin and runs `run 0`, run `lmp`, parse the log + force dump, and
compare to the eager Python reference. On 5 MnNiGa frames (8 and 10
atoms, non-cubic cells, 3 species):

```
                                                E_eager      |dE|     |dF|max   |dS|max
system_0_step1                                   0.1832    4.7e-07   5.7e-05   2.5e-03 bar
system_0_step34                                 -0.7173    2.7e-07   4.8e-06   2.3e-02 bar
system_0_step50                                 -0.6001    9.0e-08   2.0e-06   1.8e-02 bar
system_57_step1                                  1.4460    1.3e-07   2.0e-05   3.1e-03 bar
system_209_step10                                1.5229    9.2e-09   4.9e-06   5.2e-03 bar
```

**Important quirk caught by 5.4**: the cached `edge_index`/`edge_shift` in
training npz files do *not* exactly match a fresh ASE build at the same
cutoff — they include some edges with `d > cutoff` (filtered by the model
internally) and miss a handful of edges right at `d ≈ cutoff`. The parity
harness rebuilds the neighbor list freshly via ASE for the eager reference
so both paths see the same graph LAMMPS would.

### Build gotchas worth flagging (caught during 5.2)

- **Toolchain consistency.** Conda activates a different gcc/libstdc++
  than `/usr/bin/g++` is likely linked against. Mixing them produces
  errors like `'memcpy' was not declared in this scope` from inside
  the bundled `fmt/format.h` (gcc 14 is stricter than gcc 11).
  Fix: pass `CC=/usr/bin/gcc CXX=/usr/bin/g++` explicitly and prepend
  `/usr/bin` to PATH for the build.
- **MPI must match.** A plugin built against conda's MPI headers loaded
  into a serial LAMMPS produces `MPI_Comm`-layout mismatches. Either
  build LAMMPS with system MPI (`apt install libopenmpi-dev`) and pass
  `-DUSE_REAL_MPI=on` to the plugin, or build both with `BUILD_MPI=off`
  + STUBS (the default the plugin's CMakeLists uses).
- **CXX11 ABI.** Modern (2024+) torch wheels are CXX11 ABI=1. Older
  wheels were 0. Plugin default is 1; symptom of a wrong choice is
  `undefined symbol: ...torchInternalAssertFail...RKSs` at runtime.

## Architecture decision: model lives in one place

The canonical location for the model is `model/model.py` at repo root.
`LammpsModel` lives at the bottom of that file, alongside `MLIP_v20`.
The previously-separate copy at `benchmark/stress_diagnostic/model_frozen/`
has been retired; `04_virial_head_finetune.py` now imports from `model/`
directly. Three small TorchScript-compatibility fixes were made there:

  * `MLIP_v20.__init__` no longer takes `**kwargs`
  * `safe_norm` got explicit type annotations
  * `AngularAttention.forward` uses explicit `edge_index[0]/[1]` instead of
    `row, col = edge_index` (Tensor unpacking errors under jit)

These are behaviorally identical to the prior code; training is unaffected.
Both `deploy.py` and the trainer import `LammpsModel` / `MLIP_v20` from the
same `model/model.py`, so retraining auto-flows into the deploy pipeline.
No copies to keep in sync.

`model/calculator.py` (the production ASE calculator) was already wired for
the v20 per-atom-virial output format; it just needed the v20 architecture
to be on its import path. As of this sync, it loads virial-head checkpoints
cleanly.

## How the deploy pipeline works

```
checkpoint.pt  ─►  build_lammps_model()  ─►  LammpsModel  ─►  torch.jit.trace  ─►  scripted.pt
                                              │
                  std_E, std_F, std_S_pa, mean_E, E0[Z]  ─┘  baked as buffers
```

The exported `.pt` returns *physical-unit* outputs:

  * `E_total`   — total energy in eV (interaction + Σ E0[Z])
  * `virial_atom`  — per-atom Voigt virial in eV (xx, yy, zz, xy, xz, yz)

Forces come from `pos.requires_grad_(True); E.backward(); F = -pos.grad`, the
same pattern `pair_allegro` and `pair_mace` use. Trace (not script) is the
deploy method because `torch.autograd.grad` inside a scripted module
segfaults under torch 2.7 — `pair_allegro/pair_mace` use trace for the same
reason.

## Quick sanity check (Phase 2 deliverable)

```bash
cd lammps/deploy
python deploy.py \
    --ckpt ../../benchmark/stress_diagnostic/outputs/finetuned_VHCLEAN.pt \
    --out  built/vhclean.pt
```

The script auto-runs two parity checks:

  1. eager vs. traced on the trace-time input — must agree to 1e-5.
  2. eager vs. traced on a *different N* — catches `int(N)`-as-constant bugs
     that would silently produce wrong energies for any other system size.

On `finetuned_VHCLEAN.pt`: both checks pass with `|dE|=0`, `|dF|max≈1e-7`,
`|dV|max=0`. Output `.pt` is ≈600 KB.

## Real-frame validation (Phase 3 deliverable)

```bash
cd lammps/tests
python compare_with_eager.py \
    --ckpt   ../../benchmark/stress_diagnostic/outputs/finetuned_VHCLEAN.pt \
    --traced ../deploy/built/vhclean.pt \
    --frames ../../MnNiGa/data/cached_clean/system_0_step1.npz \
             ../../MnNiGa/data/cached_clean/system_0_step50.npz \
             ../../MnNiGa/data/cached_clean/system_209_step10.npz \
             ../../MnNiGa/data/cached_clean/system_153_step1.npz \
             ../../MnNiGa/data/cached_clean/system_57_step1.npz
```

Reference is the eager `MLIP_v20.forward(batch, ...)` path the trainer
itself uses (NOT `model/calculator.py`, which is for the older v19-era
architecture and can't load a virial-head checkpoint). On 5 real MnNiGa
cached frames (N=8 and N=10): `|dE| ≤ 2e-7`, `|dF|max ≤ 1e-6`, `|dS|max ≈ 1e-9`
— all within float32 noise.
