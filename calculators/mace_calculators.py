import os
import argparse
from mace.calculators import mace_mp
from ase.io import read
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.build import make_supercell
from ase.optimize import BFGS
from ase.io import write
import numpy as np

atm = 0.0000061208295469345  # eV/A^3 per atm
B = 979027.8806 * atm
ps = np.power(10, 3) * units.fs  # convert ps to fs for ASE MD timestep


class MACECalculator:

    def __init__(
        self,
        model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
        model_name: str = "mace-omat-0-medium.model",
        supercell_size: tuple = (1, 1, 1),
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.supercell_size = supercell_size

    @property
    def calculator(self):
        from mace.calculators import MACECalculator

        return MACECalculator(model_path=os.path.join(self.model_path, self.model_name), device="cuda", default_dtype="float64")  # ,enable_cueq=True)
        # if not os.path.exists(os.path.join(self.model_path, self.model_name)):
        #    raise FileNotFoundError(f"Model file not found at {os.path.join(self.model_path, self.model_name)}")
        # return mace_mp(model=os.path.join(self.model_path, self.model_name),device='cuda',default_dtype="float64",enable_cueq=True)  # , device="mps", default_dtype="float32")

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure: str):
        """
        Load an input structure, build a supercell, and attach the MACE calculator.

        This setter expects a VASP-style POSCAR path. When a POSCAR file is provided,
        the structure is read with ASE, expanded according to ``self.supercell_size``,
        and stored internally as ``self._structure``. The configured MACE calculator
        is then attached to the resulting atoms object.

        Args:
            structure (str): Path to the structure file (expected to contain "POSCAR").

        Returns:
            ase.Atoms: The processed structure with supercell applied and calculator set.
        """
        if "CAR" in structure:
            self._structure = read(structure, format="vasp")
            print(f"Loaded structure from {structure} with {len(self._structure)} atoms.")

        print(f"Creating supercell of size {self.supercell_size} for the structure.")
        sc = make_supercell(self._structure, np.diag(self.supercell_size))
        self._structure = sc

        print(f"total number of atoms after supercell creation: {len(self._structure)}")
        # self._structure.set_calculator(self.calculator)
        self._structure.calc = self.calculator
        return self._structure

    @property
    def output_dir(self):
        """
        Get the output directory for storing results. If not set, defaults to the current working directory,
        which is where the script is run from."""
        if not hasattr(self, "_output_dir"):
            self._output_dir = os.getcwd()
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str):
        self._output_dir = output_dir
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def relax_structure(self, fmax: float = 0.0005, max_steps: int = 1000) -> None:
        """
        Relax the atomic structure using the BFGS optimizer with the MACE calculator.

        Performs geometry optimization to find the minimum energy configuration of the structure.
        The relaxation trajectory and log file are saved to the output directory, and the final
        relaxed structure is written to a CONTCAR file.

        Args:
            fmax (float): Maximum force convergence criterion in eV/Å. Default is 0.0005.
            max_steps (int): Maximum number of optimization steps. Default is 1000.

        Returns:
            None

        Raises:
            FileNotFoundError: If the MACE model file is not found.

        Output files:
            - relaxation.traj: ASE trajectory file containing the relaxation path
            - relaxation.log: Log file with optimization details
            - CONTCAR: Final relaxed structure in VASP format
        """

        self.output_dir = os.path.join(self.output_dir, "mace_relaxation")
        trajectory_path = os.path.join(self.output_dir, "relaxation.traj")
        log_path = os.path.join(self.output_dir, "relaxation.log")

        # run the structure relaxation using the BFGS optimizer
        print(f"Running structure relaxation with MACE model: {self.model_name}")
        optimizer = BFGS(atoms=self.structure, trajectory=trajectory_path, logfile=log_path)
        optimizer.run(fmax=fmax, steps=max_steps)

        # save the relaxed structure to a CONTCAR file in the output directory, mimicking VASP output
        poscar_path = os.path.join(self.output_dir, "CONTCAR")
        print(f"Saving relaxed structure to {poscar_path}")
        write(poscar_path, self.structure, format="vasp")

    def npt_md(self, temperature: float = 300.0, duration_ps: float = 10.0, initialise_velocities: bool = True) -> None:
        """
        Run an NPT molecular dynamics equilibration with the MACE calculator to slowly equilibrate the system to the
        cell volume at the target temperature at 1atm (taking into account the effect of thermal expansion).

        The structure is propagated using ASE's ``NPTBerendsen`` integrator at the
        requested temperature and pressure. If requested, velocities are
        initialized from a Maxwell-Boltzmann distribution before the simulation.
        Output files are written into an ``npt_equilibration`` directory under
        ``self.output_dir``.

        Args:
            temperature (float): Target temperature in K. Default is 300.0.
            duration_ps (float): Simulation duration in picoseconds. Default is 10.0.
            initialise_velocities (bool): Whether to initialize atomic velocities
                before running the MD simulation. Default is True.

        Returns:
            None

        Output files:
            - npt_equilibration.traj: Trajectory file for the MD run.
            - npt_equilibration.log: Log file for the MD run.
            - CONTCAR: Final structure written in VASP format.
        """

        npt_path = os.path.join(self.output_dir, "npt_equilibration", str(int(temperature)) + "_K")
        if not os.path.exists(npt_path):
            os.makedirs(npt_path)
        trajectory_path = os.path.join(npt_path, "npt_equilibration.traj")
        log_path = os.path.join(npt_path, "npt_equilibration.log")

        if initialise_velocities:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

            MaxwellBoltzmannDistribution(self.structure, temperature_K=temperature)

        from ase.md.nptberendsen import NPTBerendsen

        trajectory = Trajectory(trajectory_path, "w", self.structure)
        npt = NPTBerendsen(
            self.structure,
            pressure_au=atm,
            timestep=1 * units.fs,
            temperature_K=float(temperature),
            taut=0.05 * 1e3,
            taup=0.25 * 1e3,
            compressibility_au=1 / B,
            # trajectory=str(trajectory_path),
            logfile=str(log_path),
        )
        npt.attach(trajectory.write, interval=10)
        npt.run(int(duration_ps * 1e3))

        poscar_path = os.path.join(npt_path, "CONTCAR")
        print(f"Saving final structure to {poscar_path}")
        write(poscar_path, self.structure, format="vasp")

    def nve_md(self, temperature: float = 300.0, duration_ps: float = 10.0, initialise_velocities: bool = False) -> None:
        """
        Run an NVE molecular dynamics simulation with the MACE calculator.

        The structure is propagated using ASE's ``VelocityVerlet`` integrator in the microcanonical ensemble (NVE).
        If requested, velocities are initialized from a Maxwell-Boltzmann distribution before the simulation.
        Output files are written into an ``nve_md`` directory under ``self.output_dir``.

        Args:
            temperature (float): Target temperature in K for velocity initialization. Default is 300.0.
            duration_ps (float): Simulation duration in picoseconds. Default is 10.0.
            initialise_velocities (bool): Whether to initialize atomic velocities
                before running the MD simulation. Default is True.

        Returns:
            None

        Output files:
            - nve_md.traj: Trajectory file for the MD run.
            - nve_md.log: Log file for the MD run.
            - CONTCAR: Final structure written in VASP format.
        """

        nve_path = os.path.join(self.output_dir, "nve_md", str(int(temperature)) + "_K")
        if not os.path.exists(nve_path):
            os.makedirs(nve_path)
        trajectory_path = os.path.join(nve_path, "nve_md.traj")
        log_path = os.path.join(nve_path, "nve_md.log")

        if initialise_velocities:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

            MaxwellBoltzmannDistribution(self.structure, temperature_K=temperature)

        from ase.md.verlet import VelocityVerlet

        trajectory = Trajectory(trajectory_path, "w", self.structure)
        nve = VelocityVerlet(
            self.structure,
            timestep=1 * units.fs,
            logfile=str(log_path),
        )
        nve.attach(trajectory.write, interval=10)
        nve.run(int(duration_ps * 1e3))

        poscar_path = os.path.join(nve_path, "CONTCAR")
        print(f"Saving final structure to {poscar_path}")
        write(poscar_path, self.structure, format="vasp")

    def langevin_md(self, temperature: float = 300.0, duration_ps: float = 10.0, gamma: float = 0.01, initialise_velocities: bool = False) -> None:
        """
        Run a Langevin molecular dynamics simulation with the MACE calculator.

        The structure is propagated using ASE's ``Langevin`` integrator, which includes stochastic and frictional forces to mimic a heat bath at the specified temperature. If requested, velocities are initialized from a Maxwell-Boltzmann distribution before the simulation. Output files are written into a ``langevin_md`` directory under ``self.output_dir``.

        Args:
            temperature (float): Target temperature in K for velocity initialization. Default is 300.0.
            duration_ps (float): Simulation duration in picoseconds. Default is 10.0.
            gamma (float): Friction coefficient for the Langevin thermostat in fs^-1. Default is 0.01.
            initialise_velocities (bool): Whether to initialize atomic velocities
                before running the MD simulation. Default is True.

        Returns:
            None

        Output files:
            - langevin_md.traj: Trajectory file for the MD run.
            - langevin_md.log: Log file for the MD run.
            - CONTCAR: Final structure written in VASP format.
        """

        langevin_path = os.path.join(self.output_dir, "langevin_md", str(int(temperature)) + "_K_gamma_" + str(gamma).replace(".", "_"))
        if not os.path.exists(langevin_path):
            os.makedirs(langevin_path)
        trajectory_path = os.path.join(langevin_path, "langevin_md.traj")
        log_path = os.path.join(langevin_path, "langevin_md.log")

        if initialise_velocities:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

            MaxwellBoltzmannDistribution(self.structure, temperature_K=temperature)

        from ase.md.langevin import Langevin

        trajectory = Trajectory(trajectory_path, "w", self.structure)

        # find all the non-Li-elements and set a constratinst to only allow the Li atoms to move during the MD simulation
        input_atom_order = self.structure.get_chemical_symbols()
        input_atom_oder_dict = {i: symbol for i, symbol in enumerate(input_atom_order)}
        gamma_list = [gamma if symbol != "Li" else 0.0 for i, symbol in input_atom_oder_dict.items()]

        langevin = Langevin(
            self.structure,
            timestep=1 * units.fs,
            temperature_K=float(temperature),
            friction=np.array([g / units.fs for g in gamma_list]),
            logfile=str(log_path),
            fixcm=False,
        )
        langevin.attach(trajectory.write, interval=10)
        langevin.run(int(duration_ps * 1e3))

        poscar_path = os.path.join(langevin_path, "CONTCAR")
        print(f"Saving final structure to {poscar_path}")
        write(poscar_path, self.structure, format="vasp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlling calculators using the MACE foundation model.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the MACE model.",
        default="/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
    )
    parser.add_argument("--model_name", type=str, help="Name of the MACE model to use.", default="mace-omat-0-medium.model")
    parser.add_argument("--structure", type=str, help="Path to the structure file (e.g., POSCAR).", default="POSCAR")
    parser.add_argument("--dim", type=int, nargs=3, help="Supercell dimensions (e.g., --dim 2 2 2).", default=(1, 1, 1))

    # calculator control arguments
    parser.add_argument("-opt", "--optimise", action="store_true", help="Whether to perform structure relaxation.")
    parser.add_argument("--fmax", type=float, default=0.0005, help="Maximum force for structure relaxation.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps for structure relaxation.")

    # md control arguments
    parser.add_argument("--npt", action="store_true", help="Whether to perform NPT MD equilibration.")
    parser.add_argument("--nve", action="store_true", help="Whether to perform NVE MD simulation.")
    parser.add_argument("--langevin", action="store_true", help="Whether to perform Langevin MD simulation.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature for NPT MD equilibration in K.")
    parser.add_argument("--duration_ps", type=float, default=10.0, help="Duration of NPT MD equilibration in picoseconds.")
    parser.add_argument("--gamma", type=float, default=0.001, help="Friction coefficient for Langevin MD simulation.")
    args = parser.parse_args()

    mace_calculator = MACECalculator(model_path=args.model_path, model_name=args.model_name, supercell_size=args.dim)
    mace_calculator.structure = args.structure

    if args.optimise:
        print("Starting structure relaxation...")
        mace_calculator.relax_structure(fmax=args.fmax, max_steps=args.max_steps)
        print("Structure relaxation completed.")
    elif args.npt:
        print("Starting NPT MD equilibration...")
        mace_calculator.npt_md(temperature=args.temperature, duration_ps=args.duration_ps)
        print("NPT MD equilibration completed.")
    elif args.nve:
        print("Starting NVE MD simulation...")
        mace_calculator.nve_md(temperature=args.temperature, duration_ps=args.duration_ps)
        print("NVE MD simulation completed.")
    elif args.langevin:
        print("Starting Langevin MD simulation...")
        mace_calculator.langevin_md(temperature=args.temperature, duration_ps=args.duration_ps, gamma=args.gamma)
        print("Langevin MD simulation completed.")
