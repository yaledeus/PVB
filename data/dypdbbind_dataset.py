import numpy as np
import sys
import os
import json
import mdtraj as md
from rdkit import Chem
from argparse import ArgumentParser

from subgraph import *
from mmap_dataset import create_mmap
from utils.constants import *
from utils.bio_utils import *


def load_file(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(s) for s in lines]
    return items


def preprocess_pdbbind(split_path):
    """
    :param split_path: path for dataset split file (.jsonl), each row contains one item
    """
    np.random.seed(42)

    items = load_file(split_path)
    base = os.path.dirname(split_path)

    for item in items:
        sim_base = os.path.join(base, 'sim', item['pdb'])

        try:
            # check if successfully simulated
            protein = md.load(os.path.join(sim_base, "protein_state0.pdb"))
            ligand = Chem.SDMolSupplier(os.path.join(sim_base, "ligand_state0.sdf"), removeHs=False)[0]
            holo_traj_npz = np.load(os.path.join(sim_base, "holo-traj-arrays.npz"))
        except BaseException:
            continue

        ligand_num = ligand.GetNumAtoms()
        positions = 10.0 * holo_traj_npz["positions"]  # (T, N, 3), Angstrom
        positions = positions[int(positions.shape[0] / 5):][::1000]     # frames spacing: 2fs * 1000 = 2ps
        T = positions.shape[0]
        Zp, Bp, ap_index, bp_index, p_bond_index = get_block_from_top(protein.topology)
        Zl, Bl, al_index, bl_index, l_bond_index = get_block_from_mol(ligand)
        Np, Nl = len(Zp), len(Zl)
        # merge bond index
        l_bond_index += Np
        bond_index = np.hstack([p_bond_index, l_bond_index])

        atype = np.concatenate([Zp, Zl], dtype=np.compat.long)
        btype = np.concatenate([Bp, Bl], dtype=np.compat.long)

        holo_pos = positions[0]
        x_ref = np.concatenate([holo_pos[:-ligand_num][ap_index], holo_pos[-ligand_num:][al_index]])

        # holo
        edge_mask = np.array([0] * Np + [1] * Nl, dtype=np.compat.long)   # 0: pocket, 1: ligand

        # subgraph
        max_indices, mask = graph_cut(x_ref, radius_min=10.0, radius_max=20.0, xc=x_ref[edge_mask == 1].mean(axis=0))
        max_indices_list = list(max_indices)
        
        # re-index bond index
        index_mapping = {x: max_indices_list.index(x) for x in max_indices}
        bond_index_slice = []
        src, dst = bond_index
        for begin, end in zip(src, dst):
            begin = index_mapping.get(begin, -1)
            end = index_mapping.get(end, -1)
            if begin == -1 or end == -1:
                continue
            bond_index_slice.append([begin, end])
        bond_index_slice = np.array(bond_index_slice, dtype=np.compat.long).T

        for i in range(T):
            ppos, lpos = positions[i, :-ligand_num], positions[i, -ligand_num:]

            xp0, bp0 = ppos[ap_index], ppos[bp_index]
            xl0, bl0 = lpos[al_index], lpos[bl_index]

            x0 = np.concatenate([xp0, xl0])
            b0 = np.concatenate([bp0, bl0])

            assert len(x0) == len(b0) == len(atype) == len(btype), "number of atoms mismatch"
            
            xc = x0[edge_mask == 1].mean(axis=0)
            x0 = x0 - xc
            b0 = b0 - xc
            data = {
                "atype": atype[max_indices].tolist(),
                "btype": btype[max_indices].tolist(),
                "x_ref": x_ref[max_indices].tolist(),
                "x0": x0[max_indices].tolist(),
                "b0": b0[max_indices].tolist(),
                "edge_mask": edge_mask[max_indices].tolist(),
                "mask": mask.tolist(),
                "bond_index": bond_index_slice.tolist(),
                "env": 9
            }
            env_atom_num = max_indices.shape[0]
            grad_atom_num = mask.sum()
            adj_atom_num = grad_atom_num + int(0.5 * (env_atom_num - grad_atom_num))
            yield f'{item["pdb"]}_holo_{i}', data, [adj_atom_num]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='Directory for dataset split files')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    base = args.data_dir
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_pdbbind(os.path.join(base, f"{_type}.jsonl")),
            os.path.join(base, f"{_type}_block")
        )
