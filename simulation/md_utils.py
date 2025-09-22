import openmm as mm
import openmm.unit as u
from openmm import app
import numpy as np
import sys
# from openmmforcefields.generators import GAFFTemplateGenerator


def get_default_parameters():
    default_parameters = {
        "force-field": "amber14-implicit",
        "integrator": "LangevinMiddleIntegrator",
        "waterbox-pad": 1.0,
        "temperature": 300,
        "timestep": 1.0,
        "friction": 0.5,
        "sampling": 100_000_000,
        "spacing": 1_000,
        "min-tol": 2.0,
        "gpu": -1
    }
    return default_parameters


def get_simulation_environment_integrator(parameters):
    """Obtain integrator from parameters.

    Arguments
    ---------
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    integrator : openmm.Integrator
    """
    temperature = parameters["temperature"]
    friction = parameters["friction"]
    timestep = parameters["timestep"]
    if parameters["integrator"] == "LangevinIntegrator":
        integrator = mm.LangevinIntegrator(
            temperature * u.kelvin,
            friction / u.picosecond,
            timestep * u.femtosecond
        )
    elif parameters["integrator"] == "LangevinMiddleIntegrator":
        # assert version.parse(mm.__version__) >= version.parse("7.5")
        integrator = mm.LangevinMiddleIntegrator(
            temperature * u.kelvin,
            friction / u.picosecond,
            timestep * u.femtosecond
        )
    else:
        raise NotImplementedError(f'Integrator type {parameters["integrator"]} not implemented.')

    return integrator


def get_simulation_environment_from_model(model, parameters=None, ligand=None):
    """Obtain simulation environment suitable for energy computation.

    Arguments
    ---------
    model : openmm.app.modeller.Modeller
        Fully instantiated OpenMM model.
    parameters : dict or str
        Parameter dictionary or preset name.
    ligand: openff.toolkit.Molecule
        extension for small molecules, using GAFF

    Returns
    -------
    simulation : openmm.Simulation
        Simulation (topology, forcefield and computation parameters).  This
        object can be passed to the compute_forces_and_energy method.
    """
    if not parameters:
        parameters = get_default_parameters()
    system = get_system(model, parameters, ligand)
    integrator = get_simulation_environment_integrator(parameters)
    if parameters["gpu"] == -1:
        simulation = mm.app.Simulation(model.topology, system, integrator)
    else:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': f'{parameters["gpu"]}'}
        simulation = mm.app.Simulation(model.topology, system, integrator, platform, properties)

    return simulation


def get_simulation_environment_from_pdb(pdb, parameters, ligand=None):
    model = get_openmm_model(pdb)
    return get_simulation_environment_from_model(model, parameters, ligand)


def get_system(model, parameters, ligand=None):
    """Obtain system to generate e.g. a simulation environment.

    Arguments
    ---------
    model : openmm.app.modeller.Modeller
        Fully instantiated OpenMM model.
    parameters : dict or str
        Parameter dictionary or preset name.
    ligand: openff.toolkit.Molecule
        Ligand Topology, using Generalized Amber Force Field

    Returns
    -------
    system : openmm.system
        System (topology, forcefield).  This
        is required for a simulation object.
    """
    # TODO: use openmmforcefields package to support GAFF2
    # TODO: support CHARMM36 with implicit water
    
    if ligand:
        gaff = GAFFTemplateGenerator(molecules=ligand, forcefield="gaff-2.11")

    # amber99-implicit and amber14-implicit
    if parameters["force-field"].endswith("-implicit"):
        if parameters["force-field"] == "amber99-implicit":
            forcefield = mm.app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        elif parameters["force-field"] == "amber14-implicit":
            # (Onufriev, Bashford, Case, "Exploring Protein Native States and
            # Large-Scale Conformational Changes with a modified Generalized
            # Born Model", PROTEINS 2004) using the GB-OBC I parameters
            # (corresponds to `igb=2` in AMBER)
            # assert version.parse(mm.__version__) >= version.parse("7.7")
            forcefield = mm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        else:
            raise ValueError("Invalid forcefield parameter '%s'" % parameters["force-field"])
        if ligand:
            forcefield.registerTemplateGenerator(gaff.generator)
        model.addExtraParticles(forcefield)

        # Peter Eastman recommends a large cutoff value for implicit solvent
        # models, around 20 Angstrom (= 2nm), see
        # https://github.com/openmm/openmm/issues/3104
        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * u.nanometer,  # == 20 Angstrom
            constraints=mm.app.HBonds,
        )
    elif parameters["force-field"] == "amber14-explicit":
        forcefield = mm.app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        if ligand:
            forcefield.registerTemplateGenerator(gaff.generator)
        model.addExtraParticles(forcefield)
        model.addSolvent(forcefield, padding=parameters["waterbox-pad"])

        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.PME,  # .NoCutoff, .PME for particle mesh Ewald
            constraints=mm.app.HBonds,  # .HBonds   # constrain H-bonds (fastest vibrations)
        )
    elif parameters["force-field"] == "amber14-only":
        forcefield = mm.app.ForceField("amber14-all.xml")
        if ligand:
            forcefield.registerTemplateGenerator(gaff.generator)
        model.addExtraParticles(forcefield)

        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.NoCutoff,
            constraints=mm.app.HBonds
        )
    else:
        raise ValueError("Invalid forcefield parameter '%s'" % parameters["force-field"])

    return system


def get_openmm_model(state0pdbpath):
    """Create openmm model from pdf file.

    Arguments
    ---------
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.

    Returns
    -------
    model : openmm.app.modeller.Modeller
        Modeller provides tools for editing molecular models, such as adding water or missing hydrogens.
        This object can also be used to create simulation environments.
    """
    pdb_file = mm.app.pdbfile.PDBFile(state0pdbpath)
    positions = pdb_file.getPositions()
    topology = pdb_file.getTopology()
    model = mm.app.modeller.Modeller(topology, positions)
    return model


def get_potential(simulation, positions):
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(
            u.kilojoule / u.mole)

    return potential


def get_force(simulation, positions):
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        u.kilojoules / (u.mole * u.nanometer)).astype(np.float32)

    return forces


# class CustomMinimizationReporter(mm.openmm.MinimizationReporter):
#     def __init__(self, fp, *args):
#         super().__init__(*args)
#         # self.step_count = 0
#         self.fp = fp

#     def report(self, iteration, x, grad, args):
#         res = super().report(iteration, x, grad)
#         self.fp.write(f"{iteration} ")
#         return res[0]


def spring_constraint_energy_minimization(simulation, positions, opt_stats_fp=None):
    # reporter = CustomMinimizationReporter(opt_stats_fp)
    # reporter = mm.openmm.MinimizationReporter()
    spring_constant = 10.0 * u.kilocalories_per_mole / u.angstroms ** 2
    restraint = mm.CustomExternalForce('0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)')
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    restraint.addGlobalParameter('k', spring_constant)

    for atom in simulation.topology.atoms():
        if atom.element.symbol != 'H':
            index = atom.index
            position = positions[index]
            restraint.addParticle(index, [position[0], position[1], position[2]])

    simulation.system.addForce(restraint)
    simulation.context.setPositions(positions)
    tolerance = (2.39 * u.kilocalories_per_mole / u.angstroms ** 2) \
        .value_in_unit(u.kilojoules_per_mole / u.nanometers ** 2)
    simulation.minimizeEnergy(tolerance=tolerance)

    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True) \
        .value_in_unit(u.nanometer) \
        .astype(np.float32)
    return positions
