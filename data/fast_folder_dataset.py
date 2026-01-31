import numpy as np
import sys
import os
import pickle
from argparse import ArgumentParser

import mdtraj as md
import torch
import glob
from tqdm import tqdm

from utils import load_file
from utils.bio_utils import *
from data.mmap_dataset import create_mmap
from utils.constants import *


def preprocess_fast_folder(base, name, delta=1, _type="train"):
    """
    :param delta: maximum time interval between training pairs, default 1(x200) ps
    """
    np.random.seed(42)

    prefix = f'{name}-0-protein'

    pdb_path = os.path.join(base, f'DESRES-Trajectory_{prefix}', prefix, f'{prefix}.pdb')
    state0 = md.load(pdb_path)
    atype, btype, a_index, b_index, bond_index = get_block_from_top(state0.topology)
    dcd_paths = glob.glob(os.path.join(base, f'DESRES-Trajectory_{prefix}', prefix, f'{prefix}*.dcd'))

    if _type == "train":
        dcd_paths = dcd_paths[:10]
    elif _type == "valid":
        dcd_paths = dcd_paths[10:20]

    for dcd_path in tqdm(dcd_paths):
        traj = md.load(dcd_path, top=pdb_path)
        xyz = 10.0 * traj.xyz
        T = len(traj)

        # frame spacing = 200 ps, step = 200 * 5 = 1 ns
        for i in range(T)[::5]:
            # get protein pairs
            pos0, pos1 = xyz[i], xyz[i + delta]
            x0, b0 = pos0[a_index], pos0[b_index]
            x1, b1 = pos1[a_index], pos1[b_index]
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
                "env": 12
            }
            yield f'{name}_{i}', data, [atype.shape[0]]


def parse():
    arg_parser = ArgumentParser(description='curate ATLAS dataset')
    arg_parser.add_argument('--base', type=str, required=True, help='database directory')
    arg_parser.add_argument('--name', type=str, required=True, help='protein name')
    arg_parser.add_argument('--delta', type=int, default=1, help='time interval between training pairs, unit: (x200) ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    prefix = f'{args.name}-0-protein'
    # process to pdb file w/o H
    raw_protein_path = os.path.join(args.base, f'DESRES-Trajectory_{prefix}', prefix, f'{prefix}.pdb')
    state0 = md.load(raw_protein_path)
    atype, btype, a_index, b_index, bond_index = get_block_from_top(state0.topology)
    state0_noh = state0.atom_slice(a_index)
    state0_noh.save_pdb(os.path.join(args.base, f'DESRES-Trajectory_{prefix}', prefix, f'{prefix}_noh.pdb'))
    # frame spacing = 200 ps
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_fast_folder(args.base, args.name, delta=args.delta, _type=_type),
            os.path.join(args.base, f"{_type}_block_{args.name}_{args.delta * 200}ps")
        )
