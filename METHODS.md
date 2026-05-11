# Methods

What every script in `tools/`, `properties/`, and `comparison/` actually
computes, with literature pointers. Units throughout: eV, Å, GPa,
fs / K / Kelvin.

---

## 1. Parity evaluation (`tools/parity.py`)

For each frame *i* in a dataset split, the model is evaluated and the
**physical-unit** prediction is compared against the cached SIESTA
truth for every quantity the dataset stores: per-atom energy, total
energy, atom-wise forces, virial-stress tensor (Voigt), per-atom
magnetic moment, per-atom Mulliken charge. Per-element baselines
(`E0[Z]`) are added back before comparison so that the energy parity
is in absolute eV, not the model's internal "interaction" channel.

Summary statistics reported per quantity:

- **MAE**:  ⟨|ŷ − y|⟩
- **RMSE**: √⟨(ŷ − y)²⟩
- **R²**:    1 − Σ(ŷ−y)² / Σ(y−⟨y⟩)² — the
  fraction of variance in the truth that the model explains.

R² is preferred over raw RMSE for cross-dataset comparison because
RMSE depends on the dynamic range of the target; R² is dimensionless.

Per-element references are obtained by least squares on the training
set, as in the standard cohesive-energy-subtraction trick used in
modern MLIPs (Behler & Parrinello 2007 [^bp2007], Schütt et al. 2018
SchNet [^schnet], and downstream — see `model/data.py::compute_element_references`).

### Stress sign convention

ASE uses σ = +∂E/∂ε / V (tensile-positive, compressive-negative).
SIESTA reports the same. The model output matches; the calculator
converts per-atom virial × N → total virial, divided by cell volume,
to recover σ in eV/Å³.

[^bp2007]: Behler & Parrinello, *Phys. Rev. Lett.* **98**, 146401 (2007).
[^schnet]: Schütt, Sauceda, Kindermans, Tkatchenko, Müller, *J. Chem. Phys.* **148**, 241722 (2018).

---

## 2. Geometry relaxation (`model/optimize.py`)

Two routines, both using Polak–Ribiere conjugate gradient
[^polak1969] with the **PR⁺** restart rule (β = max(0, β_PR)) and
Armijo backtracking line search [^armijo1966]. Restarts every
N steps (default 20) to break out of curvature drift in long runs.
This is the standard atomistic-relaxation flavor of CG, c.f.
the `ASE` BFGS / SciPy CG implementations [^nocedal_wright].

### `cg_positions` — fixed cell

3*N* DoFs (atom positions). Gradient *g = −F*. Converges to
|F|_max < `fmax`.

### `cg_cellstress` — variable cell

3N + 6 DoFs (positions plus the symmetric **log-strain** tensor).
The cell is parameterized as

  *cell(t) = exp(ε) · cell₀*

with ε a symmetric 3 × 3 strain. The log-strain choice (sometimes
called "Hencky strain") gives a numerically stable, multiplicative
parameterization that avoids the cell-collapse pathologies of additive
strain at large deformations. The gradient on the strain block is
*+σ · V* (in Voigt 6 form), following from the Nielsen-Martin stress
theorem [^nielsen1985]: σ = (1/V) ∂E/∂ε.

Convergence: |F|_max < `fmax` **AND** |σ|_max < `smax`.

### `cg_pressure` — finite external pressure ("CG barostat")

Same algorithm, but minimizes the enthalpy *H = E + P·V*, so the
gradient on ε has an extra *P · 1 · V* on the diagonal. Used for
EOS scans at non-zero pressure, and for relaxing into a target-pressure
state without running MD. Pressure is given in GPa; conversion
1 GPa = 1/160.21766 eV/Å³.

[^polak1969]: Polak & Ribière, *Rev. Fr. Inform. Rech. Op.* **16-R1**, 35 (1969).
[^armijo1966]: Armijo, *Pacific J. Math.* **16**, 1 (1966).
[^nocedal_wright]: Nocedal & Wright, *Numerical Optimization*, Springer (2006), Chap. 5.
[^nielsen1985]: Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985).

---

## 3. Equation of state and bulk modulus (`tools/properties.py::eos`)

After equilibrium relaxation, the cell is rescaled isotropically by a
grid of scale factors *s* (e.g. 0.92 – 1.08), atoms move with the cell
(scaled coordinates fixed), and **positions are CG-relaxed at each fixed
volume**. The resulting *E(V)* curve is fit to the third-order
**Birch–Murnaghan** equation of state [^birch1947] [^murnaghan1944]:

  *E(V) = E₀ + (9 V₀ B₀ / 16) · [ (η−1)³ B′₀ + (η−1)²(6 − 4η) ]*,
  *η = (V₀/V)^(2/3)*

with parameters (*E₀, V₀, B₀, B′₀*) fit by `scipy.optimize.curve_fit`
seeded from a parabolic guess. Returns:

- **V₀**:  equilibrium volume (Å³ per cell)
- **B₀**:  isothermal bulk modulus (GPa)
- **B′₀**: pressure derivative of B₀

This is *not* the so-called "Vinet" or "BMK" form; we use the BM3
isotherm because it remains numerically well-behaved for the ±8 %
volume range typical of stable metals near 0 K. For ranges past
~15 %, the Vinet EOS [^vinet1989] is more physically motivated and
should replace BM3 in `_birch_murnaghan`.

[^birch1947]: Birch, *Phys. Rev.* **71**, 809 (1947).
[^murnaghan1944]: Murnaghan, *Proc. Natl. Acad. Sci. USA* **30**, 244 (1944).
[^vinet1989]: Vinet, Rose, Ferrante, Smith, *J. Phys.: Condens. Matter* **1**, 1941 (1989).

---

## 4. Elastic constants (`tools/properties.py::elastic_cubic`)

The stress-strain method: small symmetric strains are applied to the
equilibrium cell, atoms move with the cell, and the stress tensor is
read out at each ε. C_ij is then the linear regression of σ_α on ε_β
about ε = 0 [^le_page2002] [^golesorkhtabar2013]:

  *σ_α = C_αβ ε_β   (Voigt)*

For cubic crystals, three measurements suffice:

| mode               | strain                  | response             | C_ij                  |
|--------------------|-------------------------|----------------------|-----------------------|
| uniaxial along x   | ε_xx                    | σ_xx, σ_yy           | C₁₁ = dσ_xx/dε_xx; C₁₂ = dσ_yy/dε_xx |
| pure shear in xy   | ε_xy = ε_yx             | σ_xy                 | C₄₄ = ½ dσ_xy/dε_xy   |

Strain magnitudes: ε ∈ {±0.01, ±0.02}. Linear regression through four
points keeps the fit insensitive to the residual stress at the
input geometry (absorbed into the intercept) — important when the
model has a nonzero stress offset (see §8.2 below).

The function reports the bulk modulus cross-check *B ≈ (C₁₁ + 2 C₁₂)/3*;
compare to the EOS B₀ as a self-consistency check on the stress head.

**Diagnostic:** if σ_xx ≈ σ_yy under uniaxial strain (std < 1e-4 eV/Å³),
the model is returning hydrostatic-only stress and the elastic block is
flagged with `isotropic_only: true`. In that case only B is meaningful.

[^le_page2002]: Le Page & Saxe, *Phys. Rev. B* **65**, 104104 (2002).
[^golesorkhtabar2013]: Golesorkhtabar et al., *Comput. Phys. Commun.* **184**, 1861 (2013) — ElaStic toolkit, same approach.

---

## 5. Vacancy formation energy (`tools/properties.py::vacancy_formation`)

For each species *Z* present:

  *E_vac(Z) = E_def(N−1) − ((N−1)/N) · E_bulk(N)*

where *E_bulk* is the energy of a perfect supercell (default 3×3×3)
and *E_def* is the energy of the same supercell with one *Z* atom
removed and **positions** (cell fixed) relaxed [^vineyard1957]
[^mishin1999]. The (N−1)/N scaling makes the cohesive contribution
of N−1 atoms cancel — what remains is the defect's interaction with
its periodic images.

The cell is **not** relaxed in the defect calculation: relaxing the cell
lets it shrink to absorb the missing volume, which underestimates the
energy cost. The "fixed cell, relaxed positions" convention is the
standard one (Mishin reviews [^mishin2017]).

For finite-supercell finite-size error, a 3×3×3 fcc cubic cell (108
atoms) typically gives E_vac converged to ~50 meV vs the periodic-image
extrapolation [^freysoldt2014]. Larger supercells improve this at
super-linear cost.

[^vineyard1957]: Vineyard, *J. Phys. Chem. Solids* **3**, 121 (1957).
[^mishin1999]: Mishin, Farkas, Mehl, Papaconstantopoulos, *Phys. Rev. B* **59**, 3393 (1999).
[^mishin2017]: Mishin, *Acta Mater.* **134**, 117 (2017).
[^freysoldt2014]: Freysoldt et al., *Rev. Mod. Phys.* **86**, 253 (2014).

---

## 6. NPT molecular dynamics (`tools/barostat.py::md_npt`)

Berendsen weak-coupling thermostat + barostat [^berendsen1984]
through `ase.md.nptberendsen.NPTBerendsen`. The temperature and
pressure are dragged toward target values with time constants
τ_T and τ_P (defaults 100 fs and 1 ps); the cell rescales
isotropically according to the instantaneous trace of the
stress tensor.

Caveats (well-known):

- The Berendsen barostat is **not** strictly NPT-ensemble — it
  does not preserve the correct fluctuations of *V* and *P*.
  It's fine for equilibration and for steady-state averages of
  intensive properties, but for free-energy calculations or
  fluctuation-derived response functions (e.g. C_p, κ_T from
  *⟨V²⟩*) you want the **Parrinello–Rahman / Nosé–Hoover**
  combination instead [^parrinello1981] [^martyna1996] — `ase.md.npt.NPT`.
- The `compressibility_au` argument needs to be roughly right
  (within a factor of 5) for the barostat coupling to behave;
  100 GPa is a sensible default for metals.

[^berendsen1984]: Berendsen, Postma, van Gunsteren, DiNola, Haak, *J. Chem. Phys.* **81**, 3684 (1984).
[^parrinello1981]: Parrinello & Rahman, *J. Appl. Phys.* **52**, 7182 (1981).
[^martyna1996]: Martyna, Tuckerman, Tobias, Klein, *Mol. Phys.* **87**, 1117 (1996).

---

## 7. Comparison models

### 7.1 MACE — Higher-order equivariant message-passing

MACE [^mace2022] [^mace2023] is a *(*E(3)*-equivariant)* message-passing
network built on the **Atomic Cluster Expansion (ACE)** basis
[^drautz2019] [^musil2021]. Each layer constructs a body-ordered
many-body expansion of the local environment by symmetrized products
of spherical-harmonic edge messages, truncated at angular momentum
ℓ_max and body order ν. The resulting site energies are summed for
the total energy; forces and stress fall out of autograd.

Default `comparison/mace/<dataset>/config.yaml`:
`hidden_irreps: "64x0e + 64x1o"`, `num_interactions: 2`,
`max_ell: 2`, `r_max: 5.0 Å`, energy / force / stress loss weights
(1, 50, 1), 200 epochs, SWA on. These are intentionally lean — bump
to `128x0e + 128x1o` and `num_interactions: 3` for production. We
deliberately train with the **same E/F/S loss weighting our model
uses** so the comparison is on architecture, not loss tuning.

### 7.2 DeepMD-kit — DP-SE2 descriptors + MLP fitting

DeepMD-kit [^deepmd2018] [^deepmd2023] uses smooth pair-distance
descriptors (`se_e2_a`) — a per-environment learnable two-body
embedding with a smooth cutoff — feeding a per-element MLP fitting
network that predicts the atomic energy contribution. Forces are
analytic gradients; the virial comes from the descriptor's strain
derivative. It's **not** strictly equivariant in the MACE sense:
the descriptor is rotation-invariant (scalar) by construction,
which makes the network smaller per parameter and faster than MACE
on long simulations.

Default `comparison/deepmd/<dataset>/input.json`:
`descriptor.neuron: [25,50,100]`, `fitting_net.neuron: [100,100,100]`,
`rcut: 5.0`, 200 000 steps, energy/force loss balanced as in DeepMD's
standard "smooth" template.

### 7.3 Why both?

MACE and DeepMD anchor the cost/accuracy spectrum: MACE is
near-state-of-the-art on small datasets and benchmarks (rMD17,
3BPA, MPTrj) but slow per step; DeepMD is fast per step and has
been used at million-atom scales (Linear Scaling, Wang group), but
its scalar descriptor is the weak link for systems with strong
angular dependence (transition metals at off-equilibrium, magnetic
phases).

Both are wrapped as ASE Calculators
(`mace.calculators.MACECalculator`, `deepmd.calculator.DP`), so
`tools/properties.py::run_all` runs unchanged on either — only the
calculator attached to the Atoms object changes.

[^mace2022]: Batatia, Kovács, Simm, Ortner, Csányi, *NeurIPS* (2022).
[^mace2023]: Batatia et al., *arXiv* 2401.00096 (2024) — MACE-MP-0 (foundation model paper, methods section is the cleanest reference for the architecture).
[^drautz2019]: Drautz, *Phys. Rev. B* **99**, 014104 (2019) — ACE.
[^musil2021]: Musil et al., *Chem. Rev.* **121**, 9759 (2021) — ML interatomic potentials review.
[^deepmd2018]: Wang, Zhang, Han, E, *Comput. Phys. Commun.* **228**, 178 (2018).
[^deepmd2023]: Zeng et al., *J. Chem. Phys.* **159**, 054801 (2023) — DeePMD-kit v2.

---

## 8. Cross-method caveats

### 8.1 Stress vs virial — sign and units

| code         | sign convention            | units of "stress"   | what `data_export.py` does     |
|--------------|----------------------------|---------------------|--------------------------------|
| ASE / our model | σ = +∂E/∂ε / V (tensile > 0) | eV/Å³            | exposed as-is                   |
| extxyz / MACE | same as ASE                | eV/Å³               | written into `stress="..."`     |
| DeepMD       | virial *W = +σ · V* (tensile > 0, **eV**) | eV       | converted: *W = +(−σ) · V* ?    |

`comparison/data_export.py` writes DeepMD virial as `-σ · V`. This
follows the VASP / DeepMD convention where compressive *W > 0*, but it's
worth a sanity check on your first MACE/DeepMD training run: predict
a frame at zero strain, compare predicted virial to the
training-time value. If signs disagree, flip `virial = -σ * V` to
`virial = +σ * V` in `data_export.py::write_deepmd_set`.

### 8.2 The v20 stress head is hydrostatic-only on symmetric cells

Observed empirically on fcc Cu and bcc Fe: for cubic cells under any
single-component strain, the model outputs σ_xx ≈ σ_yy ≈ σ_zz and
σ_xy ≈ 0 (see `properties/Cu/properties.json`,
`properties/Fe/properties.json`, where `elastic.isotropic_only: true`).

The likely cause is that **TM23 training data only samples thermal MD
trajectories near equilibrium**, where the instantaneous stress in a
cubic cell is dominated by hydrostatic fluctuations and the directional
σ_xy etc. components average to zero. The detached-feature virial head
(see `MODEL.md`) learns that statistical distribution and reproduces
only its dominant mode.

This is one of the things the MACE / DeepMD comparison should reveal:
both have full directional stress trained from the same data; if their
elastic block also collapses to isotropic, the limitation is dataset
sampling, not architecture. If only ours collapses, it's a
weakness of the detached-feature stress head.

### 8.3 Magnetism

Our model uses **categorical spin labels** (0 = nonmagnetic, 1 = up,
2 = down) per atom, embedded into the model's hidden state. MACE and
DeepMD in their stock builds do **not** carry a spin channel; they're
treated as non-magnetic.

For Fe and TM23/Fe this is moot — TM23 was trained non-magnetically.
For MnNiGa, the head-to-head comparison is therefore not strictly fair:
our model has access to magnetic information neither comparison code
sees. Document this clearly when reporting.

---

## 9. References — primary index

A condensed index for citation in any write-up:

1. **CG / line search** — Polak & Ribière 1969, Nocedal & Wright 2006, Armijo 1966.
2. **Stress theorem** — Nielsen & Martin 1985.
3. **EOS** — Birch 1947, Murnaghan 1944, Vinet et al. 1989.
4. **Elastic constants from stress-strain** — Le Page & Saxe 2002, Golesorkhtabar et al. 2013 (ElaStic).
5. **Vacancy formation** — Vineyard 1957, Mishin 1999/2017, Freysoldt 2014.
6. **Barostat (Berendsen vs Parrinello-Rahman)** — Berendsen 1984, Parrinello & Rahman 1981, Martyna 1996.
7. **ASE** — Larsen et al., *J. Phys.: Condens. Matter* **29**, 273002 (2017).
8. **MLIP foundations** — Behler & Parrinello 2007 (atom-centered NNP), Bartók et al. 2010 (GAP), Drautz 2019 (ACE), Musil et al. 2021 (review).
9. **Equivariant message-passing MLIPs** — Schütt et al. 2021 (PaiNN), Batzner et al. 2022 (NequIP), Batatia et al. 2022/2023 (MACE).
10. **DeepMD-kit** — Wang et al. 2018, Zeng et al. 2023.
