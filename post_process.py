import numpy as np
import mdtraj as md
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
import json
import os
import subprocess
import yaml
from tqdm import tqdm

from ept.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from ept.data.converter.rdkit_to_blocks import rdkit_to_blocks
from ept.data.converter.blocks_to_data import blocks_to_data
from ept.data.mmap_dataset import create_mmap


def write_coord_to_sdf(mol, coord, save_path):
    writer = Chem.SDWriter(save_path)
    # only one conformation
    if coord.ndim == 2:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(len(coord)):
            conf.SetAtomPosition(i, coord[i])
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        writer.write(mol)
    # multiple conformations
    else:
        T, N = coord.shape[0], coord.shape[1]
        mol.RemoveAllConformers()
        for t in range(T):
            conf = Chem.Conformer(N)
            for i in range(N):
                conf.SetAtomPosition(i, coord[t, i])
            conf_id = mol.AddConformer(conf, assignId=True)
            writer.write(mol, confId=conf_id)
    writer.close()

def traj_to_data(gen_pocket_traj: md.Trajectory, gen_ligand_path: str, pdb: str):
    save_dir = os.path.join(os.path.abspath('.'), 'tmp', pdb)
    os.makedirs(save_dir, exist_ok=True)

    suppl = Chem.SDMolSupplier(gen_ligand_path, removeHs=True)

    T = len(gen_pocket_traj)

    # save pockets
    for i in range(T):
        gen_pocket_traj[i].save_pdb(os.path.join(save_dir, f"{pdb}_pocket_{i}_model.pdb"))
    
    # save ligands
    for i, mol in enumerate(suppl):
        writer = Chem.SDWriter(os.path.join(save_dir, f"{pdb}_ligand_{i}_model.sdf"))
        writer.write(mol)
        writer.close()
    
    return save_dir


def process_iterator(save_dir: str, pdb: str, n_samples: int):
    for id in range(n_samples):
        pocket_fname = os.path.join(save_dir, f"{pdb}_pocket_{id}_model.pdb")
        ligand_fname = os.path.join(save_dir, f"{pdb}_ligand_{id}_model.sdf")

        list_blocks1 = pdb_to_list_blocks(pocket_fname)
        mol = Chem.SDMolSupplier(ligand_fname, removeHs=False)[0]
        blocks2 = rdkit_to_blocks(mol, using_hydrogen=False, hydrogen_as_block=False)
        blocks1 = []
        for b in list_blocks1:
            blocks1.extend(b)

        data = blocks_to_data(blocks1, blocks2)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        yield pdb, data, [len(data['B'])]


def data_to_blocks(save_dir: str, pdb: str, n_samples: int):
    create_mmap(
        process_iterator(save_dir, pdb, n_samples),
        save_dir
    )
