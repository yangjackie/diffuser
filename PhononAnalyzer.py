"""
PhononAnalyzer.py
-----------------
Vibrational (phonon) analysis from molecular-dynamics trajectories.

Algorithm (following Jack Yang, PCCP 2025):
  1. Compute the velocity autocorrelation function (VACF) from atomic velocities.
  2. Fourier-transform the VACF to obtain the phonon density of states (PDOS).
  3. Integrate the PDOS within the harmonic approximation to yield the
     vibrational Helmholtz free energy F_vib(T).

Reference:
  J. Yang et al., "Accounting for the vibrational contribution to the
  configurational entropy in disordered solids with machine learned
  forcefields: a case study of garnet electrolyte Li7La3Zr2O12",
  Phys. Chem. Chem. Phys. 27, 2025.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Physical constants ───────────────────────────────────────────────────────
KB_EV_PER_K = 8.617333262145e-5  # Boltzmann constant  [eV / K]
HBAR_EV_FS = 6.582119569e-4  # ħ  [eV · fs]        (= ħ in eV·s / 1e15)
THZ_PER_FS = 1.0  # frequency axis in THz when dt is in fs
# 1 THz = 1e12 Hz;  ħ [eV·fs] × ω [rad/fs] = ħω [eV]
# ω [rad/fs] = 2π × f [THz·fs⁻¹]  (since 1 THz = 10¹² s⁻¹ = 10⁻³ fs⁻¹)
FS_PER_THz_RADS = 2.0 * np.pi * 1e-3  # convert f[THz] → ω[rad/fs]


# ── Data containers ─────────────────────────────────────────────────────────


@dataclass
class VACFResult:
    """Stores the (normalised) velocity autocorrelation function.

    Attributes:
        time_fs:  Time axis [fs], length N_corr.
        vacf:     Normalised VACF C(t) = <v(0)·v(t)> / <v(0)·v(0)>,
                  averaged over all selected atoms and all time origins.
                  Shape (N_corr,).
        species:  Chemical symbol(s) used to compute this VACF, or 'all'.
    """

    time_fs: np.ndarray
    vacf: np.ndarray
    species: str = "all"


@dataclass
class PDOSResult:
    """Stores the phonon density of states derived from the VACF FFT.

    Attributes:
        freq_THz:  Frequency axis [THz], positive half only.
        pdos:      PDOS g(ω), normalised so that ∫g(ω)dω = 1 (per unit-cell
                   degree of freedom). Shape matches freq_THz.
        species:   Chemical symbol(s), same as the parent VACFResult.
    """

    freq_THz: np.ndarray
    pdos: np.ndarray
    species: str = "all"
    normalisation_constant: float = field(default=1.0, repr=False)  # ∫pdos dω before normalisation


@dataclass
class FvibResult:
    """Stores the vibrational Helmholtz free energy at multiple temperatures.

    Attributes:
        temperatures_K:   Array of temperatures [K] at which F_vib was evaluated.
        fvib_eV:          F_vib(T) per formula unit [eV].
        fvib_eV_per_atom: F_vib(T) per atom [eV / atom].
        n_atoms:          Number of atoms used in the calculation.
        species:          Chemical symbol(s) included in the PDOS.
    """

    temperatures_K: np.ndarray
    fvib_eV: np.ndarray
    fvib_eV_per_atom: np.ndarray
    n_atoms: int = 0
    species: str = "all"


@dataclass
class VibrationalThermoResult:
    """Stores vibrational thermodynamics derived from a PDOS.

    Attributes:
        temperatures_K:      Array of temperatures [K].
        fvib_eV:             Total vibrational Helmholtz free energy [eV].
        fvib_eV_per_atom:    Vibrational Helmholtz free energy per atom [eV/atom].
        svib_eV_per_K:       Total vibrational entropy [eV/K].
        svib_eV_per_K_per_atom:
                             Vibrational entropy per atom [eV/K/atom].
        uvib_eV:             Total vibrational internal energy [eV].
        uvib_eV_per_atom:    Vibrational internal energy per atom [eV/atom].
        n_atoms:             Number of atoms represented by the PDOS.
        species:             Chemical symbol(s) included in the PDOS.
    """

    temperatures_K: np.ndarray
    fvib_eV: np.ndarray
    fvib_eV_per_atom: np.ndarray
    svib_eV_per_K: np.ndarray
    svib_eV_per_K_per_atom: np.ndarray
    uvib_eV: np.ndarray
    uvib_eV_per_atom: np.ndarray
    n_atoms: int = 0
    species: str = "all"


# ── Helper ───────────────────────────────────────────────────────────────────


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration, compatible with all NumPy versions.

    Parameters:
        y: Integrand values, shape (N,).
        x: Abscissa values,  shape (N,).

    Returns:
        Scalar integral.
    """
    return float(np.sum((np.asarray(y[:-1]) + np.asarray(y[1:])) / 2.0 * np.diff(x)))


def _progress(iterable, **kwargs):
    """Wrap an iterable with tqdm if available, else return as-is."""
    if _TQDM_AVAILABLE:
        return _tqdm(iterable, **kwargs)
    return iterable


def compute_vibrational_thermodynamics_from_pdos(
    freq_THz: np.ndarray,
    pdos: np.ndarray,
    temperatures: Sequence[float] | None = None,
    n_atoms: int = 1,
    species: str = "all",
) -> VibrationalThermoResult:
    """Compute F_vib, S_vib, and U_vib from a normalised PDOS.

    Parameters:
        freq_THz:      Frequency axis [THz], positive frequencies only.
        pdos:          Normalised PDOS on ``freq_THz``.
        temperatures:  Temperatures [K]. Default [300, 600, 900, 1200, 1500].
        n_atoms:       Number of atoms represented by the PDOS.
        species:       Chemical label propagated to the result.

    Returns:
        VibrationalThermoResult.
    """
    freq_THz = np.asarray(freq_THz, dtype=np.float64)
    pdos = np.asarray(pdos, dtype=np.float64)
    if freq_THz.ndim != 1 or pdos.ndim != 1 or freq_THz.shape != pdos.shape:
        raise ValueError("freq_THz and pdos must be one-dimensional arrays of equal length.")
    if freq_THz.size < 2:
        raise ValueError("freq_THz and pdos must contain at least two points.")
    if np.any(freq_THz <= 0.0):
        raise ValueError("freq_THz must contain strictly positive frequencies only.")
    if np.any(pdos < 0.0):
        raise ValueError("pdos must be non-negative.")
    if temperatures is None:
        temperatures = [300.0, 600.0, 900.0, 1200.0, 1500.0]
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be strictly positive.")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")

    omega_rad_fs = 2.0 * np.pi * freq_THz * 1e-3
    hbar_omega = HBAR_EV_FS * omega_rad_fs

    fvib_eV = np.zeros_like(temperatures)
    fvib_eV_per_atom = np.zeros_like(temperatures)
    svib_eV_per_K = np.zeros_like(temperatures)
    svib_eV_per_K_per_atom = np.zeros_like(temperatures)
    uvib_eV = np.zeros_like(temperatures)
    uvib_eV_per_atom = np.zeros_like(temperatures)

    for i, T in enumerate(temperatures):
        kT = KB_EV_PER_K * T
        x = np.clip(hbar_omega / kT, 1e-12, None)
        n_be = 1.0 / np.expm1(x)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s_kernel = KB_EV_PER_K * (x * n_be - np.log1p(-np.exp(-x)))
            u_kernel = hbar_omega * (0.5 + n_be)
            f_kernel = u_kernel - T * s_kernel

        dof_factor = 3.0 * n_atoms
        fvib_eV[i] = dof_factor * _trapz(pdos * f_kernel, freq_THz)
        svib_eV_per_K[i] = dof_factor * _trapz(pdos * s_kernel, freq_THz)
        uvib_eV[i] = dof_factor * _trapz(pdos * u_kernel, freq_THz)

        fvib_eV_per_atom[i] = fvib_eV[i] / n_atoms
        svib_eV_per_K_per_atom[i] = svib_eV_per_K[i] / n_atoms
        uvib_eV_per_atom[i] = uvib_eV[i] / n_atoms

    return VibrationalThermoResult(
        temperatures_K=temperatures,
        fvib_eV=fvib_eV,
        fvib_eV_per_atom=fvib_eV_per_atom,
        svib_eV_per_K=svib_eV_per_K,
        svib_eV_per_K_per_atom=svib_eV_per_K_per_atom,
        uvib_eV=uvib_eV,
        uvib_eV_per_atom=uvib_eV_per_atom,
        n_atoms=n_atoms,
        species=species,
    )


# ── Main class ────────────────────────────────────────────────────────────────


class PhononAnalyzer:
    """Compute VACF, PDOS and vibrational free energy from an MD trajectory.

    Typical usage::

        from ase.io import read
        from Analyser.PhononAnalyzer import PhononAnalyzer

        traj = read("nvt_prod.traj", index=":")
        pa = PhononAnalyzer(traj, temperature=900, time_step=2.0)
        pa.compute_vacf()
        pa.compute_pdos()
        fvib = pa.compute_fvib(temperatures=[300, 600, 900])

    Parameters:
        traj:           List of ASE Atoms objects (production run only,
                        equilibration frames should be stripped beforehand).
        temperature:    MD temperature [K]. If ``None``, inferred from
                        ``traj[0].get_temperature()``.
        time_step:      Time between consecutive frames [fs]. Default 2.0 fs.
        diffuse_atom:   Element symbol (str) or list of atom indices to include
                        when computing the VACF/PDOS. ``None`` uses all atoms.
        corr_length:    Number of frames used as the VACF correlation window.
                        Default: half the trajectory length.
        zero_padding:   Zero-padding factor applied before FFT (improves
                        frequency resolution). Default 4.
    """

    def __init__(
        self,
        traj,
        temperature: float | None = None,
        time_step: float = 2.0,
        diffuse_atom: str | list[int] | None = None,
        corr_length: int | None = None,
        zero_padding: int = 4,
    ):
        self.traj = traj
        self.time_step = time_step  # [fs]
        self.zero_padding = zero_padding

        # ── temperature ──────────────────────────────────────────────────────
        if temperature is not None:
            self.temperature = float(temperature)
        else:
            try:
                self.temperature = float(traj[0].get_temperature())
            except AttributeError:
                raise ValueError("Temperature not provided and cannot be inferred from the " "trajectory. Pass temperature= explicitly.")

        # ── atom selection ───────────────────────────────────────────────────
        all_symbols = [atom.symbol for atom in traj[0]]
        if diffuse_atom is None:
            self.indices = list(range(len(traj[0])))
            self.species_label = "all"
        elif isinstance(diffuse_atom, str):
            self.indices = [i for i, s in enumerate(all_symbols) if s == diffuse_atom]
            self.species_label = diffuse_atom
            if not self.indices:
                raise ValueError(f"No atoms with symbol '{diffuse_atom}' found in trajectory.")
        else:
            self.indices = list(diffuse_atom)
            self.species_label = "custom"

        self.n_selected = len(self.indices)
        self.n_frames = len(traj)

        # ── correlation length ───────────────────────────────────────────────
        self.corr_length = corr_length if corr_length is not None else self.n_frames // 2

        # ── storage for results ──────────────────────────────────────────────
        self.vacf_result: VACFResult | None = None
        self.pdos_result: PDOSResult | None = None

        logger.info(
            "PhononAnalyzer: %d frames, dt=%.2f fs, %d atoms (%s), " "corr_length=%d",
            self.n_frames,
            self.time_step,
            self.n_selected,
            self.species_label,
            self.corr_length,
        )

    # ── VACF ──────────────────────────────────────────────────────────────────

    def compute_vacf(self, normalisation=True) -> VACFResult:
        """Compute the normalised velocity autocorrelation function.

        The VACF is averaged over:
          * all selected atoms,
          * all valid time origins (sliding-window, stride 1).

        Parameters:
            normalisation: For computing the elemental contributions to PDOS, that are further summed to get the total PDOS, it is more consistent to not normalise the VACF so that the relative magnitudes of the elemental contributions are preserved. For computing the total PDOS, normalisation is recommended to ensure that the PDOS integral is properly normalised. Default True. If False, the raw VACF is returned without normalisation (i.e. C(0) = <v(0)·v(0)> is not scaled to 1).

        The result is normalised by C(0) so that VACF[0] = 1.

        Returns:
            VACFResult with fields ``time_fs`` and ``vacf``.
        """
        n_origins = self.n_frames - self.corr_length
        if n_origins <= 0:
            raise ValueError(f"corr_length ({self.corr_length}) >= n_frames ({self.n_frames}). " "Reduce corr_length.")

        logger.info("Extracting velocities for %d frames …", self.n_frames)

        # Shape: (n_frames, n_selected, 3)
        velocities = np.array(
            [frame.get_velocities()[self.indices] for frame in _progress(self.traj, desc="Loading velocities", leave=False)],
            dtype=np.float64,
        )

        # compute the centre-of-mass velocity and subtract it from the atomic velocities to get the relative velocities
        # atoms = self.traj[0]
        # masses = atoms.get_masses()
        # com_velocity = np.sum(velocities * masses[None, :, None], axis=1) / np.sum(masses)
        # velocities -= com_velocity[:, None, :]
        # velocities = velocities[:, self.indices, :]

        logger.info("Computing VACF (corr_length=%d, n_origins=%d) …", self.corr_length, n_origins)

        # Accumulate C(τ) = Σ_{t0} Σ_i v_i(t0) · v_i(t0+τ)
        # Shape of vacf_sum: (corr_length,)
        vacf_sum = np.zeros(self.corr_length, dtype=np.float64)

        for t0 in _progress(range(n_origins), desc="VACF origins", leave=False):
            v0 = velocities[t0]  # (n_sel, 3)
            for tau in range(self.corr_length):
                vt = velocities[t0 + tau]  # (n_sel, 3)
                # dot product per atom, then average over atoms
                vacf_sum[tau] += np.mean(np.sum(v0 * vt, axis=1))

        vacf_raw = vacf_sum / n_origins

        # Accumulate C(τ) = Σ_{t0} Σ_i v_i(t0) · v_i(t0+τ) using a strided
        # lag view so the contraction runs in NumPy rather than Python loops.
        # v0 = velocities[:n_origins]  # (n_origins, n_sel, 3)
        # lagged = np.lib.stride_tricks.sliding_window_view(velocities, window_shape=n_origins, axis=0)
        # lagged = np.moveaxis(lagged[: self.corr_length], -1, 1)  # (corr_length, n_origins, n_sel, 3)
        # vacf_sum = np.einsum("oac,toac->t", v0, lagged, optimize=True) / self.n_selected

        # vacf_raw = vacf_sum / n_origins

        # Normalise so VACF[0] = 1
        if normalisation:
            c0 = vacf_raw[0]
            if c0 == 0.0:
                raise ValueError("VACF[0] = 0; check that velocities are non-zero.")
            vacf_norm = vacf_raw / c0
        else:
            vacf_norm = vacf_raw

        time_fs = np.arange(self.corr_length) * self.time_step

        self.vacf_result = VACFResult(
            time_fs=time_fs,
            vacf=vacf_norm,
            species=self.species_label,
        )
        logger.info("VACF done.")
        return self.vacf_result

    # ── PDOS ──────────────────────────────────────────────────────────────────

    def compute_pdos(
        self,
        freq_max_THz: float | None = None,
        apply_hanning: bool = True,
    ) -> PDOSResult:
        """Compute the phonon density of states via FFT of the VACF.

        The VACF is windowed (Hanning by default) and zero-padded before the
        FFT to improve frequency resolution.  Only positive frequencies are
        returned. The PDOS is normalised so that its integral equals 1.

        Parameters:
            freq_max_THz:   Truncate PDOS above this frequency [THz].
                            Default: keep all positive frequencies.
            apply_hanning:  Apply a Hanning window to the VACF before FFT
                            to reduce spectral leakage. Default True.

        Returns:
            PDOSResult with fields ``freq_THz`` and ``pdos``.

        Raises:
            RuntimeError: if :meth:`compute_vacf` has not been called yet.
        """
        if self.vacf_result is None:
            raise RuntimeError("Call compute_vacf() before compute_pdos().")

        vacf = self.vacf_result.vacf.copy()
        N = len(vacf)

        # Optional Hanning window
        if apply_hanning:
            vacf *= np.hanning(N)

        # Zero-padding
        N_padded = N * self.zero_padding

        # Real FFT (VACF is real-valued)
        ft = np.fft.rfft(vacf, n=N_padded)
        pdos_raw = np.abs(ft.real)  # take real part (VACF is even)

        # Frequency axis [THz]:
        #   Δt [fs] → f_max = 1/(2·Δt) [fs⁻¹] = 1/(2·Δt)·1000 THz
        dt_fs = self.time_step
        freqs = np.fft.rfftfreq(N_padded, d=dt_fs) * 1e3  # [THz]

        # Keep only positive frequencies (rfft already does this, but
        # exclude the DC component at index 0 to avoid singularities later)
        pos_mask = freqs > 0.0
        freqs = freqs[pos_mask]
        pdos_raw = pdos_raw[pos_mask]

        # Optional frequency cutoff
        if freq_max_THz is not None:
            mask = freqs <= freq_max_THz
            freqs = freqs[mask]
            pdos_raw = pdos_raw[mask]

        # Normalise: ∫g(ω) dω = 1  (using THz axis directly)
        norm = _trapz(pdos_raw, freqs)
        if norm == 0.0:
            raise ValueError("PDOS integral is zero; check the VACF.")
        pdos_norm = pdos_raw / norm

        self.pdos_result = PDOSResult(
            freq_THz=freqs,
            pdos=pdos_raw,  # store the unnormalised PDOS so the normalisation can be applied depends on what's needed (e.g. summing over species or not)
            species=self.vacf_result.species,
            normalisation_constant=norm,  # normalisation constant for reference, not applied to pdos field
        )
        logger.info(
            "PDOS done. Frequency range: %.2f – %.2f THz",
            freqs[0],
            freqs[-1],
        )
        return self.pdos_result

    # ── F_vib ─────────────────────────────────────────────────────────────────

    def compute_fvib(
        self,
        temperatures: Sequence[float] | None = None,
    ) -> FvibResult:
        """Compute the vibrational Helmholtz free energy F_vib(T).

        Uses the harmonic (quasi-harmonic) approximation:

        .. math::

            F_{\\mathrm{vib}}(T) = 3 N k_B T \\int_0^\\infty g(\\omega)
            \\ln\\!\\left[2 \\sinh\\!\\left(\\frac{\\hbar\\omega}{2 k_B T}\\right)
            \\right] \\mathrm{d}\\omega

        where g(ω) is the normalised PDOS, N is the number of selected atoms,
        and the factor 3N converts from "per degree of freedom" to the total
        vibrational contribution.

        Parameters:
            temperatures:   List or array of temperatures [K] at which to
                            evaluate F_vib.  Default: [300, 600, 900, 1200,
                            1500] K.

        Returns:
            FvibResult containing ``fvib_eV`` and ``fvib_eV_per_atom``.

        Raises:
            RuntimeError: if :meth:`compute_pdos` has not been called yet.
        """
        if self.pdos_result is None:
            raise RuntimeError("Call compute_pdos() before compute_fvib().")

        thermo = compute_vibrational_thermodynamics_from_pdos(
            freq_THz=self.pdos_result.freq_THz,
            pdos=self.pdos_result.pdos,
            temperatures=temperatures,
            n_atoms=self.n_selected,
            species=self.pdos_result.species,
        )

        result = FvibResult(
            temperatures_K=thermo.temperatures_K,
            fvib_eV=thermo.fvib_eV,
            fvib_eV_per_atom=thermo.fvib_eV_per_atom,
            n_atoms=self.n_selected,
            species=self.pdos_result.species,
        )
        logger.info("F_vib computed at %d temperatures.", len(result.temperatures_K))
        return result

    def compute_thermo(
        self,
        temperatures: Sequence[float] | None = None,
    ) -> VibrationalThermoResult:
        """Compute F_vib, S_vib, and U_vib from the current PDOS."""
        if self.pdos_result is None:
            raise RuntimeError("Call compute_pdos() before compute_thermo().")
        return compute_vibrational_thermodynamics_from_pdos(
            freq_THz=self.pdos_result.freq_THz,
            pdos=self.pdos_result.pdos,
            temperatures=temperatures,
            n_atoms=self.n_selected,
            species=self.pdos_result.species,
        )

    # ── Full pipeline helper ──────────────────────────────────────────────────

    def analyze(
        self,
        temperatures: Sequence[float] | None = None,
        freq_max_THz: float | None = None,
        apply_hanning: bool = True,
    ) -> FvibResult:
        """Run the full VACF → PDOS → F_vib pipeline in one call.

        Parameters:
            temperatures:   Temperatures [K] for F_vib. Default: [300 … 1500].
            freq_max_THz:   Frequency cutoff for PDOS [THz]. Default: no cutoff.
            apply_hanning:  Apply Hanning window before FFT. Default True.

        Returns:
            FvibResult.
        """
        self.compute_vacf()
        self.compute_pdos(freq_max_THz=freq_max_THz, apply_hanning=apply_hanning)
        return self.compute_fvib(temperatures=temperatures)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_vacf(
        self,
        save_name: str = "vacf.png",
        fig_dir: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot the normalised VACF.

        Parameters:
            save_name: Output file name (PNG). Default ``"vacf.png"``.
            fig_dir:   Directory for the output file. Default: current directory.
            show:      If True, also call ``plt.show()``.

        Raises:
            RuntimeError: if :meth:`compute_vacf` has not been called yet.
            ImportError:  if matplotlib is not installed.
        """
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for plotting.")
        if self.vacf_result is None:
            raise RuntimeError("Call compute_vacf() first.")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.vacf_result.time_fs, self.vacf_result.vacf, color="steelblue", linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Time (fs)", fontsize=12)
        ax.set_ylabel("VACF  C(t) / C(0)", fontsize=12)
        ax.set_title(
            f"Velocity Autocorrelation Function  " f"[{self.vacf_result.species}]  T = {self.temperature:.0f} K",
            fontsize=11,
        )
        ax.tick_params(direction="in")
        fig.tight_layout()

        out = Path(fig_dir or ".") / save_name
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        logger.info("VACF plot saved to %s", out)
        if show:
            plt.show()
        plt.close(fig)

    def plot_pdos(
        self,
        save_name: str = "pdos.png",
        fig_dir: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot the phonon density of states (PDOS).

        Parameters:
            save_name: Output file name (PNG). Default ``"pdos.png"``.
            fig_dir:   Directory for the output file. Default: current directory.
            show:      If True, also call ``plt.show()``.

        Raises:
            RuntimeError: if :meth:`compute_pdos` has not been called yet.
            ImportError:  if matplotlib is not installed.
        """
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for plotting.")
        if self.pdos_result is None:
            raise RuntimeError("Call compute_pdos() first.")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(self.pdos_result.freq_THz, self.pdos_result.pdos, alpha=0.35, color="coral")
        ax.plot(self.pdos_result.freq_THz, self.pdos_result.pdos, color="firebrick", linewidth=1.2)
        ax.set_xlabel("Frequency (THz)", fontsize=12)
        ax.set_ylabel("PDOS  g(ω)  (normalised)", fontsize=12)
        ax.set_title(
            f"Phonon Density of States  " f"[{self.pdos_result.species}]  T = {self.temperature:.0f} K",
            fontsize=11,
        )
        ax.tick_params(direction="in")
        fig.tight_layout()

        out = Path(fig_dir or ".") / save_name
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        logger.info("PDOS plot saved to %s", out)
        if show:
            plt.show()
        plt.close(fig)

    def plot_fvib(
        self,
        fvib_result: FvibResult,
        save_name: str = "fvib.png",
        fig_dir: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot F_vib(T) vs temperature.

        Parameters:
            fvib_result: Output from :meth:`compute_fvib`.
            save_name:   Output file name (PNG). Default ``"fvib.png"``.
            fig_dir:     Directory for the output file. Default: cwd.
            show:        If True, also call ``plt.show()``.

        Raises:
            ImportError: if matplotlib is not installed.
        """
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for plotting.")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fvib_result.temperatures_K, fvib_result.fvib_eV, "o-", color="steelblue", linewidth=1.4, markersize=5, label="F_vib (total)")
        ax.set_xlabel("Temperature (K)", fontsize=12)
        ax.set_ylabel("F_vib  (eV)", fontsize=12)
        ax.set_title(
            f"Vibrational Free Energy  [{fvib_result.species}]",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.tick_params(direction="in")
        fig.tight_layout()

        out = Path(fig_dir or ".") / save_name
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        logger.info("F_vib plot saved to %s", out)
        if show:
            plt.show()
        plt.close(fig)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def save_results(
        self,
        fvib_result: FvibResult,
        out_dir: str = ".",
        prefix: str = "phonon",
    ) -> None:
        """Save VACF, PDOS and F_vib data as NumPy .npz archives and a CSV.

        Outputs (all in ``out_dir``):
          * ``{prefix}_vacf.npz``  — ``time_fs``, ``vacf``
          * ``{prefix}_pdos.npz``  — ``freq_THz``, ``pdos``
          * ``{prefix}_fvib.csv``  — temperature [K], F_vib [eV],
                                     F_vib/atom [eV/atom]

        Parameters:
            fvib_result: Output from :meth:`compute_fvib`.
            out_dir:     Output directory. Created if it does not exist.
            prefix:      File-name prefix. Default ``"phonon"``.
        """
        import csv

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # VACF
        if self.vacf_result is not None:
            np.savez_compressed(
                out_path / f"{prefix}_vacf.npz",
                time_fs=self.vacf_result.time_fs,
                vacf=self.vacf_result.vacf,
            )

        # PDOS
        if self.pdos_result is not None:
            np.savez_compressed(
                out_path / f"{prefix}_pdos.npz",
                freq_THz=self.pdos_result.freq_THz,
                pdos=self.pdos_result.pdos,
            )

        # F_vib CSV
        csv_path = out_path / f"{prefix}_fvib.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["temperature_K", "fvib_eV", "fvib_eV_per_atom"])
            for T, F, Fa in zip(
                fvib_result.temperatures_K,
                fvib_result.fvib_eV,
                fvib_result.fvib_eV_per_atom,
            ):
                writer.writerow([f"{T:.1f}", f"{F:.6f}", f"{Fa:.6f}"])

        logger.info("Results saved to %s (prefix=%s)", out_path, prefix)
