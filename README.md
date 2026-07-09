# diffuser

A Python toolkit for machine-learning-potential-driven molecular dynamics and post-processing analysis of ion diffusion and lattice dynamics in solid-state materials.

## Overview

This package combines MACE machine-learning interatomic potentials (MLIPs) with trajectory analysis tools to study:

- **Ion diffusion** in solid-state electrolytes via essential dynamics (PCA of atomic displacements)
- **Lattice dynamics** via velocity autocorrelation functions (VACF) and phonon density of states (PDOS)
- **Vibrational thermodynamics** (Helmholtz free energy, entropy, internal energy) within the harmonic approximation

The methodology follows: J. Yang et al., *"Accounting for the vibrational contribution to the configurational entropy in disordered solids with machine learned forcefields: a case study of garnet electrolyte Li₇La₃Zr₂O₁₂"*, Phys. Chem. Chem. Phys. **27**, 2025.

## Repository Structure

```
diffuser/
├── calculators/
│   └── mace_calculators.py   # MACE-powered structure relaxation and MD
├── mace_model/
│   └── MACE_FT_LiPSCL_compiled.model # the fine-tuned MACE model for studying the dynamics of Li6PS5Cl fast ion conductor 
├── PhononAnalyzer.py          # VACF, PDOS, and vibrational thermodynamics
├── essential_dynamics.ipynb   # PCA of atomic displacements (NVE trajectories)
├── essential_dynamics_langevin.ipynb  # Same analysis for Langevin/NPT trajectories
└── PDOS_plotter.ipynb         # PDOS computation and visualisation
```

## Modules

### `calculators/mace_calculators.py` — `MACECalculator`

Wraps the [MACE](https://github.com/ACEsuit/mace) MLIP via [ASE](https://wiki.fysik.dtu.dk/ase/) to perform:

| Method | Description |
|---|---|
| `relax_structure()` | BFGS geometry optimisation |
| `npt_md()` | NPT equilibration (Berendsen thermostat/barostat) |
| `nve_md()` | NVE production run (Velocity Verlet) |

**Usage example:**

```python
from calculators.mace_calculators import MACECalculator

calc = MACECalculator(
    model_path="/path/to/models/",
    model_name="mace-omat-0-medium.model",
    supercell_size=(2, 2, 2),
)
calc.structure = "POSCAR"
calc.output_dir = "output/"

calc.relax_structure(fmax=0.001)
calc.npt_md(temperature=600.0, duration_ps=20.0)
calc.nve_md(temperature=600.0, duration_ps=100.0)
```

Output files are written in VASP-compatible formats (trajectory `.traj`, log, `CONTCAR`).

---

### `PhononAnalyzer.py` — `PhononAnalyzer`

Computes vibrational properties from an MD trajectory using the VACF method.

**Algorithm:**
1. Extract atomic velocities from trajectory frames.
2. Compute the normalised VACF $C(\tau) = \langle \mathbf{v}(0) \cdot \mathbf{v}(\tau) \rangle / \langle \mathbf{v}(0)^2 \rangle$ averaged over all time origins and selected atoms.
3. FFT the VACF to obtain the PDOS $g(\omega)$.
4. Integrate $g(\omega)$ within the quantum harmonic approximation to yield $F_\text{vib}(T)$, $S_\text{vib}(T)$, and $U_\text{vib}(T)$.

**Usage example:**

```python
from ase.io import read
from PhononAnalyzer import PhononAnalyzer

traj = read("nve_md.traj", index=":")
pa = PhononAnalyzer(
    traj,
    temperature=600,
    time_step=1.0,       # fs between frames
    diffuse_atom="Li",   # None = all atoms
    corr_length=5000,
)

vacf = pa.compute_vacf()
pdos = pa.compute_pdos()
thermo = pa.compute_vibrational_thermodynamics(temperatures=[300, 600, 900, 1200])

print(thermo.fvib_eV_per_atom)   # F_vib [eV/atom]
print(thermo.svib_eV_per_K_per_atom)  # S_vib [eV/K/atom]
```

**Key result dataclasses:**

| Class | Contents |
|---|---|
| `VACFResult` | `time_fs`, `vacf` |
| `PDOSResult` | `freq_THz`, `pdos` |
| `VibrationalThermoResult` | `fvib_eV`, `svib_eV_per_K`, `uvib_eV` (total and per-atom) |

---

### Notebooks

#### `essential_dynamics.ipynb` / `essential_dynamics_langevin.ipynb`

Perform **essential dynamics** (mass-weighted PCA) on MD trajectories to identify the dominant collective atomic motions:

1. Compute the time-averaged structure and per-frame mass-weighted displacements $\sqrt{m_i}\,\Delta\mathbf{r}_i(t)$.
2. Build the $3N \times 3N$ covariance matrix $\mathbf{C}$.
3. Diagonalise $\mathbf{C}$ to obtain eigenvalues $\lambda_k$ and eigenvectors $\mathbf{v}_k$.
4. Analyse the eigenvalue spectrum and cumulative variance to distinguish diffusive modes from vibrational modes.
5. Optionally restrict the analysis to a specific sublattice (e.g., Li ions only).

#### `PDOS_plotter.ipynb`

Loads pre-computed or fresh MD trajectories, runs `PhononAnalyzer` at multiple temperatures, and produces publication-quality PDOS plots with Gaussian broadening.

## Dependencies

| Package | Purpose |
|---|---|
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Atoms object, trajectory I/O, MD integrators |
| [MACE](https://github.com/ACEsuit/mace) | Machine-learning interatomic potential |
| NumPy | Numerical arrays and FFT |
| Matplotlib | Plotting |
| SciPy | Gaussian filtering for PDOS smoothing |
| tqdm *(optional)* | Progress bars |

Install dependencies:

```bash
pip install ase mace-torch numpy matplotlib scipy tqdm
```


