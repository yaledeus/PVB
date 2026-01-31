'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''

from pathlib import Path
import json
import h5py
import gzip
import shutil
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import mdtraj as md
import pytraj as pt
from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset
from subgraph import *
from data.mmap_dataset import create_mmap
from utils.constants import *
from utils.bio_utils import *


def get_entries(struct, f):
    """
    Get the trajectory coordinates from the h5 file.
    Args:
    struct (str): PDB code of the structure
    f (h5py file): h5py file containing the dataset
    
    """
    pitem = f[struct.upper()]
    trajectory_coordinates = pitem["trajectory_coordinates"][:]
    return trajectory_coordinates


def open_restart_file(struct, rstPath):
    """
    Open the restart file of the structure.
    Args:
    struct (str): PDB code of the structure
    rstPath (str): Path to the directory of the structures (containing a separate structure folder containing production.rst,production.top files)
    """
    with gzip.open(os.path.join(rstPath, struct, "production.top.gz"), 'rb') as f_in:
        with open(os.path.join(rstPath, struct, "production.top"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    traj = pt.iterload(os.path.join(rstPath, struct, "production.rst"), os.path.join(rstPath, struct, "production.top"))
    return traj


def create_new_traj(traj, h5_coordinates):
    """
    Coordinates from h5 file are appended as frames to stripped traj.  

    Args:
    traj (pytraj trajectory): Trajectory from the restart file
    h5_coordinates (np.array): Coordinates
    
    """
    h5_coordinates = np.expand_dims(h5_coordinates, axis=2)
    for frameNum in range(np.shape(h5_coordinates)[0]):
        frame = pt.Frame()
        for i in range(traj.n_atoms):
            frame.append_xyz(h5_coordinates[frameNum][i])
        traj.append(frame)
    return traj[1:]


def create_topology(topNameNew, file_top, mask):
    """
    A stripped toplogy is written to file.
    Args:
    topNameNew (str): Name of the new topology file
    file_top (str): Original topology file
    mask (str): Mask to strip the topology

    """
    top = pt.load_topology(file_top)
    top = pt.strip(top, "!({})".format(mask))
    top.save(topNameNew)


def preprocess_misato_pl(md_data_file, idx_file, rst_path):
    """
        Load the MD dataset
    """
    with open(idx_file, 'r') as f:
        ids = f.read().splitlines()
    
    md_data = h5py.File(md_data_file, 'r')

    for id in tqdm(ids):
        # cutoff = pitem["molecules_begin_atom_index"][:][-1]
        # z = np.array(pitem["atoms_number"][:][:cutoff], dtype=np.int64) - 1     # start from H = 0
        xyz = np.array(get_entries(id, md_data), dtype=np.float32)  # (100, N, 3)
        top_path = os.path.join(rst_path, id.lower(), f"{id}.pdb")
        if not os.path.exists(top_path):
            continue
        try:
            top = md.load(top_path).topology
            atype, btype, a_index, b_index, bond_index, edge_mask = get_block_from_complex(top)
        except BaseException as e:
            print(f"errno: {e}")
            continue

        # rule out complexes w/o ligand
        if sum(edge_mask == 1) == 0:
            continue

        pocket_residues = []
        pos = xyz[0]
        lig_idx = top.select('resname MOL')
        lc = pos[lig_idx].mean(axis=0)
        for residue in top.residues:
            if residue.name == 'MOL':
                continue
            res_atoms = [a.index for a in residue.atoms]
            res_coords = pos[res_atoms]
            min_distance = np.min(np.linalg.norm(res_coords - lc, axis=1))
            # threshold = 20.0 Angstrom
            if min_distance <= 20.0:
                pocket_residues.append(residue)
        pocket_bb_idx = np.concatenate([[a.index for a in res.atoms if (a.name == 'CA' or a.name == 'C' or a.name == 'N')] for res in pocket_residues], dtype=np.compat.long)

        traj = md.Trajectory(xyz / 10.0, top)
        traj = traj.superpose(traj, atom_indices=pocket_bb_idx, ref_atom_indices=pocket_bb_idx)
        xyz = traj.xyz * 10.0

        for i in range(xyz.shape[0] - 1):
            pos0, pos1 = xyz[i], xyz[i + 1]         # frame spacing: 8ns/100=80ps
            x0, b0 = pos0[a_index], pos0[b_index]
            x1, b1 = pos1[a_index], pos1[b_index]
            assert len(x0) == len(b0) == len(atype) == len(btype), "number of atoms mismatch"

            # subgraph
            xc = x0[edge_mask == 1].mean(axis=0)    # ligand center
            max_indices, mask = graph_cut(x0, radius_min=15.0, radius_max=25.0, xc=xc)
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
            x1 = x1 - xc
            b1 = b1 - xc

            data = {
                "atype": atype[max_indices].tolist(),
                "btype": btype[max_indices].tolist(),
                "x0": x0[max_indices].tolist(),
                "b0": b0[max_indices].tolist(),
                "x1": x1[max_indices].tolist(),
                "b1": b1[max_indices].tolist(),
                "edge_mask": edge_mask[max_indices].tolist(),
                "mask": mask.tolist(),
                "bond_index": bond_index_slice.tolist(),
                "env": 11
            }
            env_atom_num = max_indices.shape[0]
            grad_atom_num = mask.sum()
            adj_atom_num = grad_atom_num + int(0.5 * (env_atom_num - grad_atom_num))
            yield f'{id}_{i}', data, [adj_atom_num]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='Directory for MISATO dataset')
    return arg_parser.parse_args()
    

if __name__ == "__main__":
    args = parse()
    datasetMD = os.path.join(args.data_dir, "MD.hdf5")
    rstPath = os.path.join(args.data_dir, "parameter_restart_files_MD")
    md_data = h5py.File(datasetMD, 'r')
    splits = ["train", "val", "test"]   # "test"
    # pre-process .top + .rst to .pdb format
    for split in splits:
        idx_file = os.path.join(args.data_dir, f"{split}_MD.txt")
        with open(idx_file, 'r') as f:
            ids = f.read().splitlines()
        for id in tqdm(ids):
            top_file = os.path.join(rstPath, id.lower(), f"{id}.pdb")
            if os.path.exists(top_file):
                continue
            traj = open_restart_file(id.lower(), rstPath)
            h5_coordinates = get_entries(id, md_data)[:1]     # construct topology with xyz[0]
            mask = "!:WAT,Na+,Cl-"
            new_traj = create_new_traj(traj[mask], h5_coordinates)
            pt.write_traj(top_file, new_traj, overwrite=True, options='conect')
    
    for split in splits:
        idx_file = os.path.join(args.data_dir, f"{split}_MD.txt")
        create_mmap(
            preprocess_misato_pl(datasetMD, idx_file, rstPath),
            os.path.join(args.data_dir, f"{split}_block")
        )
        # with open(idx_file, 'r') as f:
        #     ids = f.read().splitlines()
        # data = []
        # for id in tqdm(ids):
        #     # cutoff = pitem["molecules_begin_atom_index"][:][-1]
        #     # z = np.array(pitem["atoms_number"][:][:cutoff], dtype=np.int64) - 1     # start from H = 0
        #     xyz = np.array(get_entries(id, md_data), dtype=np.float32)  # (100, N, 3)
        #     top_path = os.path.join(rstPath, id.lower(), f"{id}.pdb")
        #     if not os.path.exists(top_path):
        #         continue
        #     try:
        #         top = md.load(top_path).topology
        #         atype, btype, a_index, b_index, bond_index, edge_mask = get_block_from_complex(top)
        #     except BaseException as e:
        #         print(f"errno: {e}")
        #         continue

        #     # rule out complexes w/o ligand
        #     if sum(edge_mask == 1) == 0:
        #         continue
            
        #     top = md.load(top_path)
        #     new_top = top.atom_slice(a_index)
        #     new_top.save_pdb(state0_path := os.path.join(rstPath, id.lower(), f"{id}_post.pdb"))

        #     xyz = xyz[:, a_index] / 10   # Angstrom => nm
        #     md.Trajectory(
        #         xyz,
        #         new_top.topology
        #     ).save_xtc(traj_path := os.path.join(rstPath, id.lower(), f"{id}_post.xtc"))

        #     data.append({
        #         "pdb": id,
        #         "state0_path": state0_path,
        #         "traj_path": traj_path
        #     })
        
        # with open(os.path.join(args.data_dir, f"{split}.jsonl"), "w") as f:
        #     for item in data:
        #         item_str = json.dumps(item)
        #         f.write(f"{item_str}\n")
