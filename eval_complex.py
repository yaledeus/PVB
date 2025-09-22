import torch
import numpy as np
import mdtraj as md
import pandas as pd
from argparse import ArgumentParser
import os
import pickle
import glob
from statsmodels.tsa.stattools import acovf, acf
from scipy.stats import wasserstein_distance
from utils.tica_utils import *
from utils.backbone_utils import *
from plots import *


def load_traj(trajfile, top):
    _, ext = os.path.splitext(trajfile)
    if ext in ['.pdb']:
        traj = md.load(trajfile)
    elif ext in ['.xtc']:
        traj = md.load(trajfile, top=top)
    elif ext in ['.npz']:
        positions = np.load(trajfile)["positions"]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    elif ext in ['.npy']:
        positions = np.load(trajfile)
        if positions.ndim == 4:
            positions = positions[0]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    else:
        raise NotImplementedError
    return traj


def traj_analysis(traj_model_path, traj_ref_path, top=None, lagtime=10, use_distances=True, plot=False):
    out = {}

    traj_model = load_traj(traj_model_path, top=top)
    traj_ref = load_traj(traj_ref_path, top=top)

    ref = md.load(top)
    rtop = ref.topology
    rpos = ref.xyz[0] * 10.0
    pocket_residues = []
    prot_idx = rtop.select('protein')
    lig_idx = rtop.select('resname MOL')
    lc = rpos[lig_idx].mean(axis=0)
    for residue in rtop.residues:
        if residue.name == 'MOL':
            continue
        res_atoms = [a.index for a in residue.atoms]
        res_coords = rpos[res_atoms]
        min_distance = np.min(np.linalg.norm(res_coords - lc, axis=1))
        # threshold = 20.0 Angstrom
        if min_distance <= 20.0:
            pocket_residues.append(residue)
    pocket_idx = np.concatenate([[a.index for a in res.atoms] for res in pocket_residues], dtype=np.compat.long)
    pocket_bb_idx = np.concatenate([[a.index for a in res.atoms if (a.name == 'CA' or a.name == 'C' or a.name == 'N')] for res in pocket_residues], dtype=np.compat.long)
    
    ### ====== Ligand pose RMSD ======
    traj_ref_aligned = traj_ref.superpose(ref, atom_indices=pocket_bb_idx, ref_atom_indices=pocket_bb_idx)
    traj_model_aligned = traj_model.superpose(ref, atom_indices=pocket_bb_idx, ref_atom_indices=pocket_bb_idx)
    lig_ref_pos = traj_ref_aligned.xyz[0][lig_idx]
    rmsd_ref = [compute_crmsd(pos[lig_idx], lig_ref_pos, aligned=True) for pos in traj_ref_aligned.xyz]
    rmsd_gen = [compute_crmsd(pos[lig_idx], lig_ref_pos, aligned=True) for pos in traj_model_aligned.xyz]
    lig_pose_emd = wasserstein_distance(rmsd_ref, rmsd_gen)
    print(f"Ligand RMSD EMD = {lig_pose_emd:.4f}")
    out['ligand_pos_emd'] = lig_pose_emd


    ### ====== contact occupancy ======
    cutoff = 0.45

    def contact_map(traj):
        pairs = np.array([[i, j] for i in lig_idx for j in pocket_idx])
        dists = md.compute_distances(traj, pairs)
        contacts = (dists < cutoff).astype(float)
        return contacts.mean(axis=0), pairs

    occ_ref, pairs = contact_map(traj_ref)
    occ_gen, _ = contact_map(traj_model)

    # RMSE between occupancies
    occ_rmse = np.sqrt(np.mean((occ_ref - occ_gen) ** 2))
    print(f"Contact occupancy RMSE = {occ_rmse:.4f}")
    out['contact_occupancy_rmse'] = occ_rmse


    ### ====== CoM distance similarity ======
    lig_com_ref = md.compute_center_of_mass(traj_ref.atom_slice(lig_idx))
    pocket_com_ref = md.compute_center_of_mass(traj_ref.atom_slice(pocket_idx))
    d_ref = np.linalg.norm(lig_com_ref - pocket_com_ref, axis=1)

    lig_com_gen = md.compute_center_of_mass(traj_model.atom_slice(lig_idx))
    pocket_com_gen = md.compute_center_of_mass(traj_model.atom_slice(pocket_idx))
    d_gen = np.linalg.norm(lig_com_gen - pocket_com_gen, axis=1)

    com_emd = wasserstein_distance(d_ref, d_gen)
    print(f"COM distance EMD = {com_emd:.4f}")
    out['com_emd'] = com_emd

    if plot:
        pdb = 'anony'   # TODO: replace the name as you want
        # rmsd
        plot_metric_over_time([10.0 * np.array(rmsd_ref, dtype=float), 10.0 * np.array(rmsd_gen, dtype=float)], ['MD', 'PVB'], f'fig/{pdb}/rmsd.pdf', dt=0.08, ylabel='RMSD')
        # com
        plot_metric_over_time([10.0 * np.array(d_ref, dtype=float), 10.0 * np.array(d_gen, dtype=float)], ['MD', 'PVB'], f'fig/{pdb}/com_md.pdf', dt=0.08, ylabel='CoM')

    return out


def parse():
    arg_parser = ArgumentParser(description='simulation')
    arg_parser.add_argument('--top', type=str, default=None, help='topology file path (.pdb)')
    arg_parser.add_argument('--ref', type=str, required=True, help='reference trajectory file path')
    arg_parser.add_argument('--model', type=str, required=True, help='model generated trajectory file path')
    arg_parser.add_argument('--lagtime', type=int, help='lagtime for performing TICA')
    arg_parser.add_argument('--plot', action='store_true', help='plot figures if specified')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.model, args.ref, top=args.top, plot=args.plot)


if __name__ == "__main__":
    main(parse())
