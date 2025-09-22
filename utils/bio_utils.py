import numpy as np
# from mendeleev import element
from Bio import PDB
from Bio.PDB import PDBParser, Polypeptide
from rdkit import Chem


parser = PDBParser(QUIET=True)


ATOM_TYPE = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
# ATOM_MASS = [element(a).atomic_weight for a in ATOM_TYPE]
NUM_ATOM_TYPE = len(ATOM_TYPE)

# BIO_ATOM_TYPE = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", 'Si', "P", "S", "Cl",
#                  "K", "Ca", "Mn", "Fe", "Co", "Cu", "Zn", "As", "Se", "Br", "I", "Xe", "Au", "Hg"]
# NUM_BIO_ATOM_TYPE = len(BIO_ATOM_TYPE)

# ATOM2BIO_MAP = [BIO_ATOM_TYPE.index(atom) if atom in BIO_ATOM_TYPE else -1 for atom in ATOM_TYPE]

RES_TYPE_3 = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR', 'ASP', 'HIS', 'ASN', 'GLU',
              'LYS', 'GLN', 'MET', 'ARG', 'SER', 'THR', 'CYS', 'PRO']
RES_TYPE_1 = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S',
              'T', 'C', 'P']
NUM_RES_TYPE = len(RES_TYPE_3)

BLOCK_TYPE = [(e.lower(), e.upper()) for e in ATOM_TYPE] + list(zip(RES_TYPE_1, RES_TYPE_3))
NUM_BLOCK_TYPE = len(BLOCK_TYPE)

BOND_TYPE = [
    0,      # No bond / unrecognized bond
    1,      # Single Bond
    2,      # Double bond
    3,      # Triple bond
    4,      # Aromatic bond
]
NUM_BOND_TYPE = len(BOND_TYPE)


def get_block_from_complex(top):
    """
    parse complex (.pdb) to blocks
    :param top: mdtraj Topology
    """
    atype, btype, a_pos_index, b_pos_index, edge_mask = [], [], [], [], []
    res2ca = {}
    for residue in top.residues:
        res_name = residue.name.upper()
        if res_name == 'MOL' or res_name == 'UNK':
            for atom in residue.atoms:
                atomic_num = atom.element.atomic_number
                # rule out H
                if atomic_num == 1:
                    continue
                symbol = atom.element.symbol
                block_symbol = (symbol.lower(), symbol.upper())
                atype.append(atomic_num - 1)
                btype.append(BLOCK_TYPE.index(block_symbol))
                a_pos_index.append(atom.index)
                b_pos_index.append(atom.index)
                edge_mask.append(1)
        else:
            if res_name == 'MSE':
                res_name = 'MET'    # MET is usually transformed to MSE for structural analysis
            if not res_name in RES_TYPE_3:
                continue
            res_idx = RES_TYPE_3.index(res_name)
            block_symbol = (RES_TYPE_1[res_idx], RES_TYPE_3[res_idx])
            if residue.resSeq in res2ca:
                ca_idx = res2ca[residue.resSeq]
            else:
                ca_idx = [atom.index for atom in residue.atoms if atom.name == 'CA'][0]
                res2ca[residue.resSeq] = ca_idx
            for atom in residue.atoms:
                atomic_num = atom.element.atomic_number
                # rule out H
                if atomic_num == 1:
                    continue
                atype.append(atomic_num - 1)     # start from 0
                btype.append(BLOCK_TYPE.index(block_symbol))
                a_pos_index.append(atom.index)
                b_pos_index.append(ca_idx)
                edge_mask.append(0)
    bond_index = []
    index_mapping = {x: a_pos_index.index(x) for x in a_pos_index}
    for bond in top.bonds:
        begin, end = bond[0].index, bond[1].index
        begin = index_mapping.get(begin, -1)
        end = index_mapping.get(end, -1)
        if begin == -1 or end == -1:
            continue
        bond_index.append([begin, end])
        bond_index.append([end, begin])
    bond_index = np.array(bond_index, dtype=np.compat.long).T   # (2, E)
    atype = np.array(atype, dtype=np.compat.long)
    btype = np.array(btype, dtype=np.compat.long)
    a_pos_index = np.array(a_pos_index, dtype=np.compat.long)
    b_pos_index = np.array(b_pos_index, dtype=np.compat.long)
    edge_mask = np.array(edge_mask, dtype=np.compat.long)
    return atype, btype, a_pos_index, b_pos_index, bond_index, edge_mask


def get_block_from_top(top):
    """
    parse proteins to blocks
    :param top: mdtraj Topology
    """
    atype, btype, a_pos_index, b_pos_index = [], [], [], []
    res2ca = {}
    for residue in top.residues:
        res_name = residue.name.upper()
        if res_name == 'MSE':
            res_name = 'MET'    # MET is usually transformed to MSE for structural analysis
        if not res_name in RES_TYPE_3:
            continue
        res_idx = RES_TYPE_3.index(res_name)
        block_symbol = (RES_TYPE_1[res_idx], RES_TYPE_3[res_idx])
        if residue.resSeq in res2ca:
            ca_idx = res2ca[residue.resSeq]
        else:
            ca_idx = [atom.index for atom in residue.atoms if atom.name == 'CA'][0]
            res2ca[residue.resSeq] = ca_idx
        for atom in residue.atoms:
            atomic_num = atom.element.atomic_number
            # rule out H
            if atomic_num == 1:
                continue
            atype.append(atomic_num - 1)     # start from 0
            btype.append(BLOCK_TYPE.index(block_symbol))
            a_pos_index.append(atom.index)
            b_pos_index.append(ca_idx)
    bond_index = []
    index_mapping = {x: a_pos_index.index(x) for x in a_pos_index}
    for bond in top.bonds:
        begin, end = bond[0].index, bond[1].index
        begin = index_mapping.get(begin, -1)
        end = index_mapping.get(end, -1)
        if begin == -1 or end == -1:
            continue
        bond_index.append([begin, end])
        bond_index.append([end, begin])
    bond_index = np.array(bond_index, dtype=np.compat.long).T   # (2, E)
    atype = np.array(atype, dtype=np.compat.long)
    btype = np.array(btype, dtype=np.compat.long)
    a_pos_index = np.array(a_pos_index, dtype=np.compat.long)
    b_pos_index = np.array(b_pos_index, dtype=np.compat.long)
    return atype, btype, a_pos_index, b_pos_index, bond_index


def get_block_from_mol(mol):
    """
    parse mol to blocks
    """
    atype, btype, a_pos_index, b_pos_index = [], [], [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        block_symbol = (symbol.lower(), symbol.upper())
        if block_symbol not in BLOCK_TYPE:
            return None
        if symbol.upper() != 'H':
            atype.append(atom.GetAtomicNum() - 1)
            btype.append(BLOCK_TYPE.index(block_symbol))
            a_pos_index.append(atom.GetIdx())
            b_pos_index.append(atom.GetIdx())
        # else:
        #     # the heavy atom linked with covalent bond w.r.t. H
        #     try:
        #         neighbor = atom.GetNeighbors()[0]
        #     except:
        #         return None
        #     b_pos_index.append(neighbor.GetIdx())
    # get bonds
    bond_index = []
    index_mapping = {x: a_pos_index.index(x) for x in a_pos_index}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin = index_mapping.get(begin, -1)
        end = index_mapping.get(end, -1)
        if begin == -1 or end == -1:
            continue
        bond_index.append([begin, end])
        bond_index.append([end, begin])
    bond_index = np.array(bond_index, dtype=np.compat.long).T   # (2, E)
    atype = np.array(atype, dtype=np.compat.long)
    btype = np.array(btype, dtype=np.compat.long)
    a_pos_index = np.array(a_pos_index, dtype=np.compat.long)
    b_pos_index = np.array(b_pos_index, dtype=np.compat.long)
    return atype, btype, a_pos_index, b_pos_index, bond_index


def get_seq(pdb_file):
    sequence = ''
    structure = parser.get_structure('anony', pdb_file)
    for model in structure:
        for chain in model:
            polypeptides = Polypeptide.PPBuilder().build_peptides(chain)
            for poly in polypeptides:
                sequence += poly.get_sequence()
    return sequence


def remove_hydrogens(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    
    class NonHydrogenAtomSelect(PDB.Select):
        def accept_atom(self, atom):
            return not atom.get_name().startswith('H')
    
    io.save(output_pdb, NonHydrogenAtomSelect())


def get_res_mask(top):
    """
    :param top: mdtraj topology
    :return: residue mask: (N,)
    """
    rmask = [atom.residue.index for atom in top.atoms]
    rmask = np.array(rmask, dtype=np.compat.long)
    return rmask


def get_backbone_index(top):
    """
    :param top: mdtraj topology
    :return: backbone index of each residue, order: (N, CA, C, O), shape: (B, 4)
    """
    bb_index = []
    for residue in top.residues:
        backbone = [residue.atom(atom_name) for atom_name in ['N', 'CA', 'C', 'O'] if
                    residue.atom(atom_name) is not None]
        bb_index.append([atom.index for atom in backbone])
    bb_index = np.array(bb_index, dtype=np.compat.long)
    return bb_index


if __name__ == "__main__":
    # import mdtraj as md
    # pdb = "/data/MISATO/parameter_restart_files_MD/5wij/5WIJ.pdb"
    # top = md.load(pdb).topology
    # atype, btype, a_pos_index, b_pos_index, bond_index, edge_mask = get_block_from_complex(top)
    # print(f"atype: {atype.shape}")
    # print(f"edge_mask: {(edge_mask == 1).sum()}")
    # print(f"a_pos_index: {a_pos_index}")
    # print(f"bond_index: {bond_index[:, -20:]}")
    pass