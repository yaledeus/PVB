import numpy as np
import torch
import os
import json
import mdtraj as md
from rdkit import Chem
from argparse import ArgumentParser

from subgraph import *
from data.mmap_dataset import create_mmap
from utils.constants import *
from utils.bio_utils import *


def preprocess_crossdocked(split_path, _type="train"):
    """
    :param split_path: path for dataset split file (.pt)
    """
    np.random.seed(42)

    all_data = torch.load(split_path)["train"]
    all_len = len(all_data)
    train_len = int(0.8 * all_len)
    items = all_data[:train_len] if _type == "train" else all_data[train_len:]

    data_dir = os.path.join(os.path.split(split_path)[0], "crossdocked_pocket10")

    for item in items:
        pocket_path, ligand_path = item
        _id = pocket_path.split('/')[-1][:-4]

        pocket_path = os.path.join(data_dir, pocket_path)
        ligand_path = os.path.join(data_dir, ligand_path)

        try:
            pocket = md.load(pocket_path)
            ligand = Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
        except BaseException:
            continue

        Zp, Bp, ap_index, bp_index, p_bond_index = get_block_from_top(pocket.topology)
        Zl, Bl, al_index, bl_index, l_bond_index = get_block_from_mol(ligand)
        atype = np.concatenate([Zp, Zl], dtype=np.compat.long)
        btype = np.concatenate([Bp, Bl], dtype=np.compat.long)
        Np, Nl = len(Zp), len(Zl)
        # merge bond index
        l_bond_index += Np
        bond_index = np.hstack([p_bond_index, l_bond_index])

        edge_mask = np.array([0] * Np + [1] * Nl, dtype=np.compat.long)   # 0: pocket, 1: ligand

        pocket_pos = 10.0 * pocket.xyz[0]
        xp, bp = pocket_pos[ap_index], pocket_pos[bp_index]

        ligand_conf = ligand.GetConformer()
        ligand_pos = np.array([ligand_conf.GetAtomPosition(atom.GetIdx()) for atom in ligand.GetAtoms()], dtype=float)
        xl, bl = ligand_pos[al_index], ligand_pos[bl_index]

        x0 = np.concatenate([xp, xl])
        b0 = np.concatenate([bp, bl])
        xc = x0[edge_mask == 1].mean(axis=0)    # ligand center
        x0 = x0 - xc
        b0 = b0 - xc
        data = {
            "atype": atype.tolist(),
            "btype": btype.tolist(),
            "x0": x0.tolist(),
            "b0": b0.tolist(),
            "edge_mask": edge_mask.tolist(),
            "bond_index": bond_index.tolist()
        }
        yield _id, data, [len(edge_mask)]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--split_path', type=str, required=True, help='split file path (.pt)')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    split_path = args.split_path
    base = os.path.split(split_path)[0]
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_crossdocked(split_path, _type=_type),
            os.path.join(base, f"{_type}_block")
        )
