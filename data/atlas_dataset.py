import numpy as np
import sys
import os
import pickle
from argparse import ArgumentParser

import mdtraj as md
import torch
from tqdm import tqdm

from utils import load_file
from utils.bio_utils import *
from data.subgraph import graph_cut, SG_THRESH
from data.mmap_dataset import create_mmap
from utils.constants import *


### preprocess openmm simulation output to curate Peptide train/valid data
def preprocess_atlas(split_path, delta=1, _type="train"):
    """
    :param split_path: split summary file, jsonl format
    :param delta: maximum time interval between training pairs, default 10(x10)ps
    """
    items = load_file(split_path)

    np.random.seed(42)

    for item in tqdm(items):
        traj = md.load(item["traj_xtc_path"], top=item["state0_path"])
        top = traj.topology
        xyz = 10.0 * traj.xyz   # Angstrom
        T = len(traj)
        atype, btype, a_index, b_index, bond_index = get_block_from_top(top)

        # data split
        valid_length = list(range(T - delta))
        train_len = int(0.8 * len(valid_length))
        train_idx = np.random.choice(valid_length[:train_len], 500, replace=False)
        valid_idx = np.random.choice(valid_length[train_len:], 100, replace=False)

        idx = train_idx if _type == "train" else valid_idx

        for i in idx:
            # get protein pairs
            pos0, pos1 = xyz[i], xyz[i + delta]
            x0, b0 = pos0[a_index], pos0[b_index]
            x1, b1 = pos1[a_index], pos1[b_index]
            n_subgraph = int(atype.shape[0] / SG_THRESH) + 1
            if n_subgraph <= 1:
                xc = x0.mean(axis=0)
                x0 = x0 - xc
                b0 = b0 - xc
                x1 = x1 - xc
                b1 = b1 - xc
                data = {
                    "atype": atype.tolist(),
                    "btype": btype.tolist(),
                    "x0": x0.tolist(),
                    "b0": b0.tolist(),
                    "x1": x1.tolist(),
                    "b1": b1.tolist(),
                    "bond_index": bond_index.tolist(),
                    "env": 3
                }
                yield f'{item["pdb"]}_{i}', data, [atype.shape[0]]
            else:
                atom_indices = list(range(atype.shape[0]))
                centers = [np.random.choice(atom_indices[i * SG_THRESH: (i + 1) * SG_THRESH]) for i in range(n_subgraph - 1)]
                for center in centers:
                    max_indices, mask = graph_cut(x0, xc=x0[center], radius_min=15.0, radius_max=25.0)
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

                    x0_slice = x0[max_indices]
                    b0_slice = b0[max_indices]
                    x1_slice = x1[max_indices]
                    b1_slice = b1[max_indices]
                    # make sure in a conservative field by CoM
                    xc = x0_slice.mean(axis=0)
                    x0_slice = x0_slice - xc
                    b0_slice = b0_slice - xc
                    x1_slice = x1_slice - xc
                    b1_slice = b1_slice - xc
                    data = {
                        "atype": atype[max_indices].tolist(),
                        "btype": btype[max_indices].tolist(),
                        "mask": mask.tolist(),
                        "x0": x0_slice.tolist(),
                        "b0": b0_slice.tolist(),
                        "x1": x1_slice.tolist(),
                        "b1": b1_slice.tolist(),
                        "bond_index": bond_index_slice.tolist(),
                        "env": 3
                    }
                    env_atom_num = max_indices.shape[0]
                    grad_atom_num = mask.sum()
                    adj_atom_num = grad_atom_num + int(0.5 * (env_atom_num - grad_atom_num))
                    yield f'{item["pdb"]}_{i}-{center}', data, [adj_atom_num]


def parse():
    arg_parser = ArgumentParser(description='curate ATLAS dataset')
    arg_parser.add_argument('--split', type=str, required=True, help='dataset split file')
    arg_parser.add_argument('--delta', type=int, default=1, help='time interval between training pairs, unit: (x100)ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_atlas(args.split, delta=args.delta, _type=_type),
            os.path.join(os.path.split(args.split)[0], f"{_type}_block_{args.delta * 100}ps")
        )
