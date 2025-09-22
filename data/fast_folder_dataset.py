import numpy as np
import sys
import os
import pickle
from argparse import ArgumentParser

import mdtraj as md
import torch
import glob
import subprocess
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import load_file
from utils.geometry import kabsch_numpy
from utils.bio_utils import *
from data.mmap_dataset import create_mmap
from utils.constants import *


def preprocess_fast_folder(base, train_set, delta=1, _type="train"):
    """
    :param delta: maximum time interval between training pairs, default 1(x200) ps
    """
    np.random.seed(42)

    for name in train_set:
        print(f"[+] start processing protein {name}.")

        prefix = f'{name}-0-protein'

        pdb_path = os.path.join(base, f'DESRES-Trajectory_{prefix}', prefix, f'{prefix}.pdb')
        state0 = md.load(pdb_path)
        atype, btype, a_index, b_index, bond_index = get_block_from_top(state0.topology)
        dcd_paths = glob.glob(os.path.join(base, f'DESRES-Trajectory_{prefix}', prefix, f'{prefix}*.dcd'))
        dcd_paths = sorted(dcd_paths)

        train_paths, valid_paths = train_test_split(dcd_paths, test_size=0.2, random_state=42)
        dcd_paths = train_paths if _type == "train" else valid_paths

        for dcd_path in tqdm(dcd_paths):
            traj = md.load(dcd_path, top=pdb_path)
            traj = traj.superpose(traj)
            xyz = 10.0 * traj.xyz
            T = len(traj)

            # frame spacing = 200 ps, step = 200 * 50 = 10 ns
            for i in range(T - delta)[::50]:
                # get protein pairs
                pos0, pos1 = xyz[i], xyz[i + delta]
                # kabsch first
                pos1, _, _ = kabsch_numpy(pos1, pos0)
                x0, b0 = pos0[a_index], pos0[b_index]
                x1, b1 = pos1[a_index], pos1[b_index]
                xc = x0.mean(axis=0)
                x0 = x0 - xc
                b0 = b0 - xc
                x1 = x1 - xc
                b1 = b1 - xc
                diff_norm = np.sqrt(np.sum((x1 - x0)**2, axis=1)).mean()
                if diff_norm > 6.0:
                    # print(f"{name}: {i} bad case, norm: {diff_norm}, skip.")
                    continue
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
    arg_parser.add_argument('--delta', type=int, default=1, help='time interval between training pairs, unit: (x200) ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    with open(os.path.join(args.base, 'train.txt'), 'r') as f:
        train_set = f.readlines()
    train_set = [item.strip() for item in train_set]
    for name in train_set:
        prefix = f'{name}-0-protein'
        data_path = os.path.join(args.base, f'DESRES-Trajectory_{prefix}', prefix)
        if not os.path.exists(data_path):
            cmd = f'tar -xvJf {os.path.join(args.base, f"DESRES-Trajectory_{prefix}.tar.xz")}'.split()
            subprocess.run(cmd, cwd=args.base, check=True)
        # process to pdb file w/o H
        raw_mae_path = os.path.join(data_path, f'{prefix}.mae')
        raw_protein_path = os.path.join(data_path, f'{prefix}.pdb')
        schrodinger_path = os.environ.get('SCHRODINGER')
        subprocess.run([f'{schrodinger_path}/utilities/structconvert', raw_mae_path, raw_protein_path, '-no_reorder', '-no_renum'], check=True)    
        state0 = md.load(raw_protein_path)
        atype, btype, a_index, b_index, bond_index = get_block_from_top(state0.topology)
        state0_noh = state0.atom_slice(a_index)
        state0_noh.save_pdb(os.path.join(data_path, f'{prefix}_noh.pdb'))
    
    for _type in ["train", "valid"]:
        # frame spacing = 200 ps
        create_mmap(
            preprocess_fast_folder(args.base, train_set, delta=args.delta, _type=_type),
            os.path.join(args.base, f"{_type}_block_{args.delta * 200}ps_filtered")
        )
