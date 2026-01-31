import numpy as np
import pandas as pd
import sys
import os
import json
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from pdbfixer import PDBFixer
import openmm as mm
from openmm.app import PDBFile
from openff.toolkit import Molecule
from openbabel import pybel
from argparse import ArgumentParser

from subgraph import *
from post_process import write_coord_to_sdf
from mmap_dataset import create_mmap
from utils.constants import *
from utils.bio_utils import *
from utils.geometry import kabsch_numpy
from utils.dock_utils import VinaDockingTask


def load_file(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(s) for s in lines]
    return items


def preprocess_pdbbind(split_path, _type="train"):
    """
    :param split_path: path for dataset split file (.csv)
    """
    np.random.seed(42)

    df = pd.read_csv(split_path)
    df = df[df["group"] == _type]

    for item in df.iterrows():
        item = item[1]
        pdb = item["pdb"]

        try:
            protein = md.load(item["pdbFile"])
            ligand = Chem.SDMolSupplier(item["ligandFile"], removeHs=False)[0]
        except BaseException:
            continue

        # AddHs for ligand
        # ligand = Chem.AddHs(ligand, addCoords=True)
        Zl, Bl, al_index, bl_index, ligand_bond_index = get_block_from_mol(ligand)
        conf = ligand.GetConformer()
        ligand_pos = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in ligand.GetAtoms()], dtype=float)
        xl, bl = ligand_pos[al_index], ligand_pos[bl_index]

        protein_pos = 10.0 * protein.xyz[0]
        Zp, Bp, ap_index, bp_index, protein_bond_index = get_block_from_top(protein.topology)
        xp, bp = protein_pos[ap_index], protein_pos[bp_index]
        Np, Nl = len(Zp), len(Zl)

        atype = np.concatenate([Zp, Zl], dtype=np.compat.long)
        btype = np.concatenate([Bp, Bl], dtype=np.compat.long)
        edge_mask = np.array([0] * Np + [1] * Nl, dtype=np.compat.long)   # 0: pocket, 1: ligand

        # merge bond index
        ligand_bond_index += Np     # start from Np
        bond_index = np.hstack([protein_bond_index, ligand_bond_index])     # (2, E)

        x0 = np.concatenate([xp, xl])
        b0 = np.concatenate([bp, bl])
        assert len(x0) == len(b0) == len(atype) == len(btype), "number of atoms mismatch"

        xc = xl.mean(axis=0)

        max_indices, mask = graph_cut(x0, xc=xc, radius_min=20.0, radius_max=40.0)
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

        x0 = x0 - xc
        b0 = b0 - xc

        data = {
            "atype": atype[max_indices].tolist(),
            "btype": btype[max_indices].tolist(),
            "x0": x0[max_indices].tolist(),
            "b0": b0[max_indices].tolist(),
            "edge_mask": edge_mask[max_indices].tolist(),
            "mask": mask.tolist(),
            "bond_index": bond_index_slice.tolist(),
            "env": 10
        }
        env_atom_num = max_indices.shape[0]
        grad_atom_num = mask.sum()
        adj_atom_num = grad_atom_num + int(0.5 * (env_atom_num - grad_atom_num))
        yield f'{pdb}_holo', data, [adj_atom_num]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--split', type=str, required=True, help='split file for pdbbind_v11 (curated by Lu et al.)')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    base = os.path.split(args.split)[0]
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_pdbbind(args.split, _type=_type),
            os.path.join(base, f"{_type}_block")
        )
