import torch
import numpy as np
import mdtraj as md
from argparse import ArgumentParser
import os
import pickle
import shutil
import subprocess
import json
from rdkit import Chem
from scipy.stats import pearsonr

from utils.io import load_file
from utils.dock_utils import docking_eval
from post_process import traj_to_data, data_to_blocks, write_coord_to_sdf


def traj_analysis(gen_protein_path, gen_ligand_path, protein_path, ligand_path, pocket_idx, pdb="anony"):
    prefix = os.path.split(gen_protein_path)[-1][:-4]
    # find ligand center
    ligand = Chem.SDMolSupplier(ligand_path, removeHs=True)[0]
    ligand_conf = ligand.GetConformer()
    ligand_pos = ligand_conf.GetPositions().astype(float)
    lc = ligand_pos.mean(axis=0)        # ligand center, (3,)

    protein = md.load(protein_path)
    protein_pos = 10.0 * protein.xyz[0]

    # pocket_residues = []
    # for residue in protein.topology.residues:
    #     res_atoms = [a.index for a in residue.atoms]
    #     res_coords = protein_pos[res_atoms]
    #     min_distance = np.min(np.linalg.norm(res_coords - lc, axis=1))
    #     # threshold = 10.0 Angstrom for EPT
    #     if min_distance <= 10.0:
    #         pocket_residues.append(residue)
    
    pocket_residues = [protein.topology.residue(r) for r in pocket_idx]
    pocket_atom_index = np.concatenate([[a.index for a in res.atoms] for res in pocket_residues], dtype=np.compat.long)
    gen_protein_traj = md.load(gen_protein_path, top=protein.topology)
    gen_pocket_traj = gen_protein_traj.atom_slice(pocket_atom_index)

    save_dir = traj_to_data(gen_pocket_traj, gen_ligand_path, pdb=pdb)
    data_to_blocks(save_dir, pdb=pdb, n_samples=len(gen_pocket_traj))
    # ept_predict(save_dir, gpu=gpu)
    subprocess.run(f"cd ept && python inference.py {save_dir} && cd ..", shell=True, check=True)
    pred_results = load_file(os.path.join(save_dir, '_results.jsonl'))
    pred_kd = [item["pred"] for item in pred_results]
    # select top-1 candidates
    # pred_holo_index = np.argsort(-np.array(pred_kd, dtype=float))[:100]
    pred_holo_index = int(np.argmax(pred_kd))
    print(f"predicted holo index: {pred_holo_index}")
    # save pred protein
    gen_protein_traj[pred_holo_index].save_pdb(gen_protein_path := os.path.join(os.path.split(gen_protein_path)[0], f'{prefix}_pred_protein.pdb'))
    # save pred ligand
    gen_ligand = Chem.SDMolSupplier(gen_ligand_path, removeHs=True)[pred_holo_index]
    writer = Chem.SDWriter(gen_ligand_path := os.path.join(os.path.split(gen_ligand_path)[0], f'{prefix}_pred_ligand.sdf'))
    writer.write(gen_ligand)
    writer.close()
    # pocket_rmsd, ligand_rmsd, strain, clash
    metrics = docking_eval(gen_protein_path, gen_ligand_path, protein_path, ligand_path, pocket_idx)
    
    shutil.rmtree(save_dir)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


def parse():
    arg_parser = ArgumentParser(description='simulation')
    arg_parser.add_argument('--gen_protein', type=str, required=True, help='model generated protein trajectory path (.xtc)')
    arg_parser.add_argument('--gen_ligand', type=str, required=True, help='model generated ligand trajectory path (.sdf)')
    arg_parser.add_argument('--protein', type=str, required=True, help='reference protein topology file (.pdb)')
    arg_parser.add_argument('--ligand', type=str, required=True, help='reference ligand topology file (.sdf)')
    arg_parser.add_argument('--pdb', type=str, default='anony', help='pdb id for the complex')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.gen_protein, args.gen_ligand, args.protein, args.ligand, args.pdb)


if __name__ == "__main__":
    main(parse())
