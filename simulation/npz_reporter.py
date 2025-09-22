import openmm.unit as u
import numpy as np


class Spacing:
    """A policy that determines when to record trajectory data."""

    def stepsUntilNextReport(self, currentStep):
        raise NotImplementedError("Derived classes need to implement stepsUntilNextReport method.")


class RegularSpacing(Spacing):
    """Regular spacing, every `reportInterval` steps."""

    def __init__(self, reportInterval):
        """Create a regular spacing.

        Parameters
        ----------
        reportInterval : int
            The interval (in time steps) at which to write frames.
        """
        super(RegularSpacing, self).__init__()
        self._reportInterval = reportInterval

    def stepsUntilNextReport(self, currentStep):
        """Return the number of steps until the next reporting step."""
        steps = self._reportInterval - currentStep % self._reportInterval
        return steps


class NPZReporter(object):
    """NPZReporter outputs positions, velocities, and forces for each frame.

    To use, create a NPZReporter, then add it to the Simulation's list of
    reporters.

    The created NPZ file will contain the following arrays:
      * 'time': (T,) array, simulation time in picoseconds.
      * 'energies': (T,2) array, each row containing [potential, kinetic]
        energies in kJ/mol.
      * 'positions': (T,num_atoms,3) array, positions in nm.
      * 'velocities': (T,num_atoms,3) array, velocities in nm/ps.
      * 'forces': (T,num_atoms,3) array, forces in kJ/(mol nm).
    """

    def __init__(self, filename, spacing, atom_indices=None):
        """Create a NPZReporter.

        Parameters
        ----------
        filename : string
            The filename to write to, should end with '.npz'.
        spacing : Spacing
            The report spacing at which to write frames.
        atom_indices : Range or List or None
            The list of atoms to record in that order in the NPZ file.
            If None, all atom coordinates are saved.
        """
        self._filename = filename
        self._spacing = spacing
        self._atom_indices = atom_indices
        self._nextModel = 0
        self._positions = []
        self._velocities = []
        self._forces = []
        self._energies = []
        self._time = []
        self._step = []

    def describeNextReport(self, simulation):
        steps = self._spacing.stepsUntilNextReport(simulation.currentStep)
        return steps, True, True, True, True, None  # PVFE

    def filter_atoms(self, data):
        if self._atom_indices:
            data = data[self._atom_indices, :]
        return data

    def report(self, simulation, state):
        self._time.append(state.getTime().value_in_unit(u.picoseconds))
        self._step.append(simulation.currentStep)
        self._energies.append(
            [
                state.getPotentialEnergy().value_in_unit(u.kilojoules_per_mole),
                state.getKineticEnergy().value_in_unit(u.kilojoules_per_mole),
            ]
        )

        # Positions
        positions = state.getPositions(asNumpy=True)
        positions = positions.value_in_unit(u.nanometer)
        positions = positions.astype(np.float32)
        positions = self.filter_atoms(positions)
        self._positions.append(positions)

        # Velocities
        velocities = state.getVelocities(asNumpy=True)
        velocities = velocities.value_in_unit(u.nanometer / u.picosecond)
        velocities = velocities.astype(np.float32)
        velocities = self.filter_atoms(velocities)
        self._velocities.append(velocities)

        # Forces
        forces = state.getForces(asNumpy=True)
        forces = forces.value_in_unit(u.kilojoules / (u.mole * u.nanometer))
        forces = forces.astype(np.float32)
        forces = self.filter_atoms(forces)
        self._forces.append(forces)

    def __del__(self):
        # Save all trajectory data to the NPZ file
        step = np.array(self._step)
        time = np.array(self._time)
        energies = np.array(self._energies)

        positions = np.stack(self._positions)
        velocities = np.stack(self._velocities)
        forces = np.stack(self._forces)

        np.savez_compressed(
            self._filename,
            step=step,
            time=time,
            energies=energies,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )