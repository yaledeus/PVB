from argparse import ArgumentParser
import openmm as mm
from tqdm import tqdm
import os
import json
import numpy as np
import yaml
import sys
import shutil
from pdbfixer import PDBFixer
from openff.toolkit import Molecule

sys.path.append('..')
from utils.bio_utils import remove_hydrogens
from simulation.npz_reporter import NPZReporter, RegularSpacing
from simulation.md_utils import *


def load_file(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(s) for s in lines]
    return items


def simulate_trajectory(protein_path, ligand_path, save_path, parameters):
    print(f"Simulation parameters: {parameters}")

    shutil.copy(ligand_path, os.path.join(save_path, 'ligand_state0.sdf'))

    # first remove all hydrogens of proteins, in case the residue did not match templates
    remove_hydrogens(protein_path, tmp_pdb_path := os.path.join(os.path.split(protein_path)[0], "tmp.pdb"))
    receptor = PDBFixer(tmp_pdb_path)
    ligand = Molecule.from_file(ligand_path)

    # fix PDB files
    receptor.findMissingResidues()
    receptor.findMissingAtoms()
    receptor.addMissingAtoms()
    receptor.addMissingHydrogens(7.0)
    receptor.removeHeterogens()

    holo_modeller = mm.app.Modeller(receptor.topology, receptor.positions)
    holo_modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    holo_modeller.addHydrogens()
    holo_modeller.deleteWater()
    holo_atoms = len(holo_modeller.positions)

    protein_modeller = mm.app.Modeller(receptor.topology, receptor.positions)
    protein_modeller.addHydrogens()
    protein_modeller.deleteWater()
    protein_atoms = len(protein_modeller.positions)

    ligand_modeller = mm.app.Modeller(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    ligand_atoms = len(ligand_modeller.positions)

    assert holo_atoms == protein_atoms + ligand_atoms, f'Atom numbers did not match'
    print(f"Pre-processed Atoms of complex / protein / ligand: {holo_atoms} / {protein_atoms} / {ligand_atoms}.")

    state0_holo_pos = holo_modeller.positions
    mm.app.PDBFile.writeFile(holo_modeller.topology, state0_holo_pos,
                             open(os.path.join(save_path, 'holo_state0.pdb'), 'w'))
    mm.app.PDBFile.writeFile(protein_modeller.topology, state0_holo_pos[:protein_atoms],
                             open(os.path.join(save_path, 'protein_state0.pdb'), 'w'))
    mm.app.PDBFile.writeFile(ligand_modeller.topology, state0_holo_pos[protein_atoms:],
                             open(os.path.join(save_path, 'ligand_state0.pdb'), 'w'))

    holo_simulation = get_simulation_environment_from_model(holo_modeller, parameters, ligand=ligand)
    holo_simulation.context.setPositions(holo_modeller.positions)

    tolerance = float(parameters["min-tol"])
    print("Performing ENERGY MINIMIZATION to tolerance %2.2f kJ/mol" % tolerance)
    holo_simulation.minimizeEnergy(tolerance=tolerance)
    print("Completed ENERGY MINIMIZATION")


    temperature = parameters["temperature"]
    print("Initializing VELOCITIES to %s" % temperature)
    holo_simulation.context.setVelocitiesToTemperature(temperature)

    # frame spacing=1ps5
    # simfile = os.path.join(save_path, f'{pdb_name}-sim.pdb')
    # simulation.reporters.append(PDBReporter(simfile, spacing))
    spacing = parameters["spacing"]
    # save NPZ file (energies, positions, velocities, forces)
    trajnpzfile = os.path.join(save_path, 'holo-traj-arrays.npz')
    holo_simulation.reporters.append(
        NPZReporter(trajnpzfile, RegularSpacing(spacing), atom_indices=range(holo_atoms))
    )
    with open(os.path.join(save_path, "simulation_env.yaml"), 'w') as yaml_file:
        yaml.dump(parameters, yaml_file, default_flow_style=False)

    sampling = parameters["sampling"]
    print(f"Begin SAMPLING for {sampling} steps.")
    holo_simulation.step(sampling)
    print("Completed SAMPLING")

    os.remove(tmp_pdb_path)

    """
    # protein relaxation
    protein_modeller = mm.app.Modeller(receptor.topology, receptor.positions)
    protein_modeller.addHydrogens()
    protein_modeller.deleteWater()

    protein_atoms = len(protein_modeller.positions)
    print("Pre-processed protein has %d atoms." % protein_atoms)

    protein_simulation = get_simulation_environment_from_model(protein_modeller, parameters, ligand=ligand)
    protein_simulation.context.setPositions(protein_modeller.positions)

    print("Performing ENERGY MINIMIZATION to tolerance %2.2f kJ/mol" % tolerance)
    protein_simulation.minimizeEnergy(tolerance=tolerance)
    print("Completed ENERGY MINIMIZATION")

    mm.app.PDBFile.writeFile(protein_simulation.topology, protein_simulation.context.getState(getPositions=True).getPositions(),
                             open(os.path.join(save_path, 'protein_state0.pdb'), 'w'))
    
    # ligand relaxation
    ligand_modeller = mm.app.Modeller(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())

    ligand_atoms = len(ligand_modeller.positions)
    print("Pre-processed ligand has %d atoms." % ligand_atoms)

    ligand_simulation = get_simulation_environment_from_model(ligand_modeller, parameters, ligand=ligand)
    ligand_simulation.context.setPositions(ligand_modeller.positions)

    print("Performing ENERGY MINIMIZATION to tolerance %2.2f kJ/mol" % tolerance)
    ligand_simulation.minimizeEnergy(tolerance=tolerance)
    print("Completed ENERGY MINIMIZATION")

    mm.app.PDBFile.writeFile(ligand_simulation.topology, ligand_simulation.context.getState(getPositions=True).getPositions(),
                             open(os.path.join(save_path, 'ligand_state0.pdb'), 'w'))

    print("Initializing VELOCITIES to %s" % temperature)
    pocket_simulation.context.setVelocitiesToTemperature(temperature)

    # save NPZ file (energies, positions, velocities, forces)
    trajnpzfile = os.path.join(save_path, f'pocket-traj-arrays.npz')
    pocket_simulation.reporters.append(
        NPZReporter(trajnpzfile, RegularSpacing(spacing), atom_indices=range(pocket_atoms))
    )

    print(f"Begin SAMPLING for {sampling} steps.")
    pocket_simulation.step(sampling)
    print("Completed SAMPLING")
    """

    """
    print("Initializing VELOCITIES to %s" % temperature)
    ligand_simulation.context.setVelocitiesToTemperature(temperature)

    # save NPZ file (energies, positions, velocities, forces)
    trajnpzfile = os.path.join(save_path, f'ligand-traj-arrays.npz')
    ligand_simulation.reporters.append(
        NPZReporter(trajnpzfile, RegularSpacing(spacing), atom_indices=range(ligand_atoms))
    )

    print(f"Begin SAMPLING for {sampling} steps.")
    ligand_simulation.step(sampling)
    print("Completed SAMPLING")
    """


def parse():
    parser = ArgumentParser(description='simulation')
    parser.add_argument('--summary', type=str, required=True, help='Path to summary file')
    parser.add_argument('--force-field', type=str, default="amber14-implicit",
                        choices=["amber99-implicit", "amber14-implicit", "amber14-explicit", "amber14-only"],
                        help='(preset) Force field, "amber99-implicit", "amber14-implicit", '
                             'or "amber14-explicit". [default: amber14-implicit]')
    parser.add_argument('--integrator', type=str, default="LangevinMiddleIntegrator",
                        choices=["LangevinMiddleIntegrator", "LangevinIntegrator"])
    parser.add_argument('--waterbox-pad', type=float, default=1.0, help='Waterbox padding width in nm [default: 1.0]')
    parser.add_argument('--temperature', type=int, default=300, help='simulation temperature [default: 300K]')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help='Integration time step in femtoseconds [default: 1.0]')
    parser.add_argument('--friction', type=float, default=0.5, help='Langevin friction in 1.0/ps [default: 0.5]')
    parser.add_argument('--sampling', type=int, default=10_000,
                        help='Number of total integration steps [default: 10_000].')
    parser.add_argument('--spacing', type=int, default=1, help='frame spacing in femtoseconds [default: 1]')
    parser.add_argument('--min-tol', type=float, default=2.0,
                        help='Energy minimization tolerance in kJ/mol [default: 2.0].')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='whether to use CUDA to accelerate simulation, -1 for cpu and {>0} for GPU index')
    return parser.parse_args()


def main(args):
    param_keys = ["force-field", "integrator", "waterbox-pad", "temperature", "timestep", "friction",
                  "sampling", "spacing", "min-tol", "gpu"]
    parameters = {key: getattr(args, key.replace('-', '_')) for key in param_keys}
    save_dir = os.path.join(os.path.split(args.summary)[0], 'sim')
    os.makedirs(save_dir, exist_ok=True)
    items = load_file(args.summary)
    for item in tqdm(items):
        pdb = item["pdb"]
        print(f"[+] Start MD simulations on complex: {pdb}.")
        item_dir = os.path.join(save_dir, pdb)
        if os.path.exists(os.path.join(item_dir, 'holo-traj-arrays.npz')):
            continue
        os.makedirs(item_dir, exist_ok=True)
        try:
            simulate_trajectory(item['protein_path'], item['ligand_path'], item_dir, parameters=parameters)
        except Exception as e:
            print(f"Error encountered {e}, skip.")
            continue


if __name__ == "__main__":
    main(parse())
