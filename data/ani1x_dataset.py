import h5py
import numpy as np
import torch
import sys
from argparse import ArgumentParser
import os
from openbabel import openbabel
import time
from rdkit import Chem

from mmap_dataset import create_mmap
from utils.constants import *
from utils.bio_utils import *


# List of keys to point to requested data
data_keys = ['wb97x_dz.energy', 'wb97x_dz.forces']  # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
# data_keys = ['wb97x_tz.energy','wb97x_tz.forces'] # CHNO portion of the data set used in AIM-Net (https://doi.org/10.1126/sciadv.aav6490)
# data_keys = ['ccsd(t)_cbs.energy'] # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
# data_keys = ['wb97x_dz.dipoles'] # A subset of this data was used for training the ACA charge model (https://doi.org/10.1021/acs.jpclett.8b01939)


def atomic_nums_to_mol(atom_nums, coords):
    obmol = openbabel.OBMol()

    coords = coords.astype(np.float64)
    for i, Z in enumerate(atom_nums):
        atom = obmol.NewAtom()
        atom.SetAtomicNum(int(Z))
        x, y, z = coords[i]
        atom.SetVector(x, y, z)

    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    obmol.AssignSpinMultiplicity(True)

    return obmol

def save_obmol_as_sdf(obmol, filename):
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("sdf")
    obConversion.WriteFile(obmol, filename)


def iter_data_buckets(h5filename, keys=None):
    """ Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d


def preprocess_ani1x(data_path, _type="train"):
    data = list(iter_data_buckets(data_path, keys=data_keys))

    indices = list(range(len(data)))
    train_len = int(0.8 * len(data))
    split_indices = indices[:train_len] if _type == "train" else indices[train_len:]

    for i in split_indices:
        df = data[i]
        obmol = atomic_nums_to_mol(df['atomic_numbers'], df['coordinates'][0])
        save_obmol_as_sdf(obmol, tmp_fpath := f"./tmp_{time.time()}.sdf")
        mol = Chem.SDMolSupplier(tmp_fpath, sanitize=True, removeHs=False)[0]
        if not mol:
            continue
        res = get_block_from_mol(mol)
        if not res:
            continue
        atype, btype, a_index, b_index, bond_index = res
        os.remove(tmp_fpath)

        X = df['coordinates']           # (M, N, 3), Angstrom
        Z = df['atomic_numbers'] - 1    # (N,)

        # B = np.array([BLOCK_TYPE.index((ATOM_TYPE[z].lower(), ATOM_TYPE[z].upper())) for z in Z], dtype=np.compat.long)
        # # heavy atom as the block center
        # a_index, b_index = [], []
        # for idx, z in enumerate(Z):
        #     # rule out H
        #     if z != 0:
        #         a_index.append(idx)
        #         b_index.append(idx)
        #     # else:
        #     #     pos_h = pos_ref[idx]
        #     #     dist_to_h = np.linalg.norm(pos_ref - pos_h, axis=1)
        #     #     dist_idx = np.argsort(dist_to_h)
        #     #     for i in dist_idx:
        #     #         if Z[i] != 0:
        #     #             block_center.append(i)
        #     #             break

        for m in range(X.shape[0]):
            x = X[m] - X[m].mean(axis=0)
            a_pos = x[a_index]
            b_pos = x[b_index]
            dp = {
                "atype": atype.tolist(),
                "btype": btype.tolist(),
                "x0": a_pos.tolist(),
                "b0": b_pos.tolist(),
                "bond_index": bond_index.tolist(),
                "env": 2
            }
            yield f'{i}-{m}', dp, [Z.shape[0]]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_path', type=str, required=True, help='path to ANI-1x dataset file (.h5)')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    # Path to the ANI-1x data set
    data_path = args.data_path

    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_ani1x(data_path, _type=_type),
            os.path.join(os.path.split(data_path)[0], f"{_type}_block")
        )
