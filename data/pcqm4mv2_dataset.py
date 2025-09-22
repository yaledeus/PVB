from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from argparse import ArgumentParser
import os

from utils.bio_utils import *
from mmap_dataset import create_mmap


def preprocess_pcqm4mv2(data_path, _type="train"):
    """
    :param data_path: path to pcqm4m-v2-train.sdf
    """
    suppl = Chem.SDMolSupplier(data_path, removeHs=False)

    np.random.seed(42)
    indexes = list(range(len(suppl)))
    np.random.shuffle(indexes)

    train_len = int(0.8 * len(indexes))
    valid_indexes = indexes[:train_len] if _type == "train" else indexes[train_len:]

    for idx in valid_indexes:
        mol = suppl[idx]
        res = get_block_from_mol(mol)
        if not res:
            continue
        atype, btype, a_index, b_index, bond_index = res
        conf = mol.GetConformer()
        pos = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()], dtype=float)
        a_pos, b_pos = pos[a_index], pos[b_index]
        xc = a_pos.mean(axis=0)
        a_pos = a_pos - xc
        b_pos = b_pos - xc
        dp = {
            "atype": atype.tolist(),
            "btype": btype.tolist(),
            "x0": a_pos.tolist(),
            "b0": b_pos.tolist(),
            "bond_index": bond_index.tolist(),
            "env": 4
        }

        yield idx, dp, [atype.shape[0]]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_path', type=str, required=True, help='path to pcqm4m-v2-train.sdf')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    # Path to the PCQM4Mv2 data set
    data_path = args.data_path

    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_pcqm4mv2(data_path, _type=_type),
            os.path.join(os.path.split(data_path)[0], f"{_type}_block")
        )
