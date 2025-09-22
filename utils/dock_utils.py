"""
Evaluation for protein-ligand binding affinity.
Borrowed from https://github.com/AlgoMole/MolCRAFT/blob/master/sample_for_pocket.py.
"""

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from .scoring_func import get_chem
# from .docking_vina import VinaDockingTask
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import mdtraj as md
import os
from posecheck import PoseCheck

from .geometry import kabsch_numpy
from .backbone_utils import compute_crmsd


class Metrics:
    def __init__(self, protein_fn, ref_ligand_fn, ligand_fn):
        self.protein_fn = protein_fn
        self.ref_ligand_fn = ref_ligand_fn
        self.ligand_fn = ligand_fn
        self.exhaustiveness = 16

    def vina_dock(self, mol):
        chem_results = {}

        try:
            # qed, logp, sa, lipinski, ring size, etc
            chem_results.update(get_chem(mol))
            chem_results['atom_num'] = mol.GetNumAtoms()

            # docking                
            vina_task = VinaDockingTask.from_generated_mol(
                mol, protein_path=self.protein_fn)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.exhaustiveness)
            docking_results = vina_task.run(mode='dock', exhaustiveness=self.exhaustiveness)

            chem_results['vina_score'] = score_only_results[0]['affinity']
            chem_results['vina_minimize'] = minimize_results[0]['affinity']
            chem_results['vina_dock'] = docking_results[0]['affinity']
            # chem_results['vina_dock_pose'] = docking_results[0]['pose']
            return chem_results
        except Exception as e:
            print(e)
        
        return chem_results

    def pose_check(self, mol):
        pc = PoseCheck()

        pose_check_results = {}

        protein_ready = False
        try:
            pc.load_protein_from_pdb(self.protein_fn)
            protein_ready = True
        except ValueError as e:
            return pose_check_results

        ligand_ready = False
        try:
            pc.load_ligands_from_mols([mol])
            ligand_ready = True
        except ValueError as e:
            return pose_check_results

        if ligand_ready:
            try:
                strain = pc.calculate_strain_energy()[0]
                pose_check_results['strain'] = strain
            except Exception as e:
                pass

        if protein_ready and ligand_ready:
            try:
                clash = pc.calculate_clashes()[0]
                pose_check_results['clash'] = clash
            except Exception as e:
                pass

            try:
                df = pc.calculate_interactions()
                columns = np.array([column[2] for column in df.columns])
                flags = np.array([df[column][0] for column in df.columns])
                
                def count_inter(inter_type):
                    if len(columns) == 0:
                        return 0
                    count = sum((columns == inter_type) & flags)
                    return count

                # ['Hydrophobic', 'HBDonor', 'VdWContact', 'HBAcceptor']
                hb_donor = count_inter('HBDonor')
                hb_acceptor = count_inter('HBAcceptor')
                vdw = count_inter('VdWContact')
                hydrophobic = count_inter('Hydrophobic')

                pose_check_results['hb_donor'] = hb_donor
                pose_check_results['hb_acceptor'] = hb_acceptor
                pose_check_results['vdw'] = vdw
                pose_check_results['hydrophobic'] = hydrophobic
            except Exception as e:
                pass

        for k, v in pose_check_results.items():
            mol.SetProp(k, str(v))

        return pose_check_results
    
    def evaluate(self):
        mol = Chem.SDMolSupplier(self.ligand_fn, removeHs=False, sanitize=False)[0]
       
        # chem_results = self.vina_dock(mol)
        chem_results = {}
        pose_check_results = self.pose_check(mol)
        chem_results.update(pose_check_results)

        return chem_results


def check_topology(mol1, mol2):
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return False

    for a1, a2 in zip(mol1.GetAtoms(), mol2.GetAtoms()):
        if a1.GetAtomicNum() != a2.GetAtomicNum():
            return False
    
    if mol1.GetNumBonds() != mol2.GetNumBonds():
        return False
    
    def bond_set(mol):
        bonds = set()
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.add((i, j, b.GetBondType()))
            bonds.add((j, i, b.GetBondType()))
        return bonds

    return bond_set(mol1) == bond_set(mol2)


def docking_eval(gen_protein_path, gen_ligand_path, ref_protein_path, ref_ligand_path, pocket_idx):
    gen_ligand = Chem.SDMolSupplier(gen_ligand_path, removeHs=True)[0]
    ref_ligand = Chem.SDMolSupplier(ref_ligand_path, removeHs=True)[0]
    # align to canonical atom numberings
    gen_ligand = Chem.RenumberAtoms(gen_ligand, np.argsort(list(rdmolfiles.CanonicalRankAtoms(gen_ligand))).tolist())
    ref_ligand = Chem.RenumberAtoms(ref_ligand, np.argsort(list(rdmolfiles.CanonicalRankAtoms(ref_ligand))).tolist())
    # assert check_topology(gen_ligand, ref_ligand)
    gen_ligand_conf = gen_ligand.GetConformer()
    ref_ligand_conf = ref_ligand.GetConformer()
    gen_ligand_pos = gen_ligand_conf.GetPositions().astype(float)
    ref_ligand_pos = ref_ligand_conf.GetPositions().astype(float)
    # compute ligand center
    lc = ref_ligand_pos.mean(axis=0)
    gen_protein = md.load(gen_protein_path)
    ref_protein = md.load(ref_protein_path)
    gen_protein_pos, ref_protein_pos = 10.0 * gen_protein.xyz[0], 10.0 * ref_protein.xyz[0]
    # find pocket residues
    # pocket_residues = []
    # for residue in ref_protein.topology.residues:
    #     res_atoms = [a.index for a in residue.atoms]
    #     res_coords = ref_protein_pos[res_atoms]
    #     min_distance = np.min(np.linalg.norm(res_coords - lc, axis=1))
    #     # threshold = 20.0 Angstrom
    #     if min_distance <= 20.0:
    #         pocket_residues.append(residue)
    # align to holo structure by pocket backbone atoms
    pocket_residues = [ref_protein.topology.residue(r) for r in pocket_idx]
    pocket_bb_idx = np.concatenate([[a.index for a in res.atoms if (a.name == 'CA' or a.name == 'C' or a.name == 'N')] for res in pocket_residues], dtype=np.compat.long)
    gen_bb_pos, ref_bb_pos = gen_protein_pos[pocket_bb_idx], ref_protein_pos[pocket_bb_idx]
    # kabsch alignemnt
    _, R, t = kabsch_numpy(gen_bb_pos, ref_bb_pos)
    gen_protein_pos = gen_protein_pos @ R + t
    gen_ligand_pos = gen_ligand_pos @ R + t
    ligand_rmsd = compute_crmsd(gen_ligand_pos, ref_ligand_pos, aligned=True)
    pocket_idx = np.concatenate([[a.index for a in res.atoms] for res in pocket_residues], dtype=np.compat.long)
    pocket_rmsd = compute_crmsd(gen_protein_pos[pocket_idx], ref_protein_pos[pocket_idx], aligned=True)
    
    # metrics = Metrics(ref_protein_path, ref_ligand_path, gen_ligand_path).evaluate()

    out = {
        "pocket_rmsd": pocket_rmsd,
        "ligand_rmsd": ligand_rmsd,
        # "strain": metrics["strain"],
        # "clash": metrics["clash"]
    }
    
    return out