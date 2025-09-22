import numpy as np
import mdtraj as md
import torch
from rdkit import Chem
from utils.bio_utils import *


def make_batch(state0file, bs, _format="pdb"):
    if _format == "pdb":
        state0 = md.load(state0file)
        top = state0.topology
        xyz = 10 * state0.xyz[0] # (N, 3), Angstrom
        atype, btype, a_index, b_index, bond_index = get_block_from_top(top)
        x0, b0 = xyz[a_index], xyz[b_index]
        xc = x0.mean(axis=0)
        x0 = x0 - xc
        b0 = b0 - xc
        atype = torch.from_numpy(atype).long()
        btype = torch.from_numpy(btype).long()
        x0 = torch.from_numpy(x0).float()
        b0 = torch.from_numpy(b0).float()
        bond_index = torch.from_numpy(bond_index).long()
    elif _format == "sdf":
        mol = Chem.SDMolSupplier(state0file, removeHs=False)[0]
        conf = mol.GetConformer()
        atype, btype, a_index, b_index, bond_index = get_block_from_mol(mol)
        pos = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()], dtype=float)
        x0, b0 = pos[a_index], pos[b_index]
        xc = x0.mean(axis=0)
        x0 = x0 - xc
        b0 = b0 - xc
        atype = torch.from_numpy(atype).long()
        btype = torch.from_numpy(btype).long()
        x0 = torch.from_numpy(x0).float()
        b0 = torch.from_numpy(b0).float()
        bond_index = torch.from_numpy(bond_index).long()
    else:
        raise NotImplementedError(f'File format {_format} cannnot be recognized.')

    # batch id (atom)
    abid = torch.tensor([[i] * atype.shape[0] for i in range(bs)], dtype=torch.long).flatten()
    # mask
    mask = torch.tensor([[1] * atype.shape[0] for _ in range(bs)], dtype=torch.bool).flatten()
    # edge mask
    edge_mask = torch.tensor([[0] * atype.shape[0] for _ in range(bs)], dtype=torch.long).flatten()

    # special: bond index
    atom_num = 0
    bond_index_batch = []
    for _ in range(bs):
        bond_index_batch.append(bond_index + atom_num)
        atom_num += atype.shape[0]
    bond_index_batch = torch.cat(bond_index_batch, dim=1)    # (2, E)

    batch = {
        "atype": atype.repeat(bs),
        "btype": btype.repeat(bs),
        "x0": x0.repeat(bs, 1),
        "b0": b0.repeat(bs, 1),
        "abid": abid,
        "mask": mask,
        "edge_mask": edge_mask,
        "bond_index": bond_index_batch
    }

    return batch, (a_index,)


def make_batch_complex_pl(protein_state0_path, ligand_state0_path, bs):
    # preprocess protein
    prot_state0 = md.load(protein_state0_path)
    top = prot_state0.topology
    prot_pos = 10.0 * prot_state0.xyz[0]    # nm => Angstrom
    ap_type, bp_type, ap_index, bp_index, p_bond_index = get_block_from_top(top)
    ap_pos, bp_pos = prot_pos[ap_index], prot_pos[bp_index]

    # preprocess ligand
    lig_state0 = Chem.SDMolSupplier(ligand_state0_path, removeHs=False)[0]
    lig_conf = lig_state0.GetConformer()
    lig_pos = np.array([lig_conf.GetAtomPosition(atom.GetIdx()) for atom in lig_state0.GetAtoms()], dtype=float)
    al_type, bl_type, al_index, bl_index, l_bond_index = get_block_from_mol(lig_state0)
    al_pos, bl_pos = lig_pos[al_index], lig_pos[bl_index]

    Np, Nl = len(ap_type), len(al_type)
    l_bond_index += Np
    bond_index = np.hstack([p_bond_index, l_bond_index])

    atype = torch.from_numpy(np.concatenate([ap_type, al_type], dtype=np.compat.long)).long()
    btype = torch.from_numpy(np.concatenate([bp_type, bl_type], dtype=np.compat.long)).long()
    x0 = torch.from_numpy(np.concatenate([ap_pos, al_pos], dtype=float)).float()
    b0 = torch.from_numpy(np.concatenate([bp_pos, bl_pos], dtype=float)).float()
    bond_index = torch.from_numpy(bond_index).long()

    # batch id (atom)
    abid = torch.tensor([[i] * atype.shape[0] for i in range(bs)], dtype=torch.long).flatten()
    # mask
    mask = torch.tensor([[1] * atype.shape[0] for _ in range(bs)], dtype=torch.bool).flatten()
    # edge mask
    edge_mask = torch.tensor([[0] * Np + [1] * Nl for _ in range(bs)], dtype=torch.long).flatten()

    xc = x0[edge_mask == 1].mean(dim=0)
    x0 = x0 - xc
    b0 = b0 - xc

    # special: bond index
    atom_num = 0
    bond_index_batch = []
    for _ in range(bs):
        bond_index_batch.append(bond_index + atom_num)
        atom_num += atype.shape[0]
    bond_index_batch = torch.cat(bond_index_batch, dim=1)    # (2, E)

    batch = {
        "atype": atype.repeat(bs),
        "btype": btype.repeat(bs),
        "x0": x0.repeat(bs, 1),
        "b0": b0.repeat(bs, 1),
        "abid": abid,
        "mask": mask,
        "edge_mask": edge_mask,
        "bond_index": bond_index_batch
    }

    return batch, (ap_index, al_index)


def make_batch_complex_mix(state0_path, bs):
    state0 = md.load(state0_path)
    top = state0.topology
    xyz = 10 * state0.xyz[0] # (N, 3), Angstrom
    atype, btype, a_index, b_index, bond_index, edge_mask = get_block_from_complex(top)
    x0, b0 = xyz[a_index], xyz[b_index]
    xc = x0[edge_mask == 1].mean(axis=0)
    x0 = x0 - xc
    b0 = b0 - xc
    atype = torch.from_numpy(atype).long()
    btype = torch.from_numpy(btype).long()
    x0 = torch.from_numpy(x0).float()
    b0 = torch.from_numpy(b0).float()
    bond_index = torch.from_numpy(bond_index).long()

    # batch id (atom)
    abid = torch.tensor([[i] * atype.shape[0] for i in range(bs)], dtype=torch.long).flatten()
    # mask
    mask = torch.tensor([[1] * atype.shape[0] for _ in range(bs)], dtype=torch.bool).flatten()
    # edge mask
    edge_mask = torch.from_numpy(edge_mask).long()

    # special: bond index
    atom_num = 0
    bond_index_batch = []
    for _ in range(bs):
        bond_index_batch.append(bond_index + atom_num)
        atom_num += atype.shape[0]
    bond_index_batch = torch.cat(bond_index_batch, dim=1)    # (2, E)

    batch = {
        "atype": atype.repeat(bs),
        "btype": btype.repeat(bs),
        "x0": x0.repeat(bs, 1),
        "b0": b0.repeat(bs, 1),
        "abid": abid,
        "mask": mask,
        "edge_mask": edge_mask,
        "bond_index": bond_index_batch
    }

    return batch, (a_index,)