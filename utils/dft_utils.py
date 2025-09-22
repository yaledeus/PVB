from pyscf import gto, dft, scf, grad
from lxml import etree
import numpy as np
import os


### Borrowed from https://github.com/heini-phys-chem/ASE_calculators/blob/master/calculators/pyscf.py
def ase_atoms_to_pyscf(ase_atoms):
    """
    Convert ASE atoms to PySCF atom.

    Note: ASE atoms always use A.
    """
    return [[ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]] for i in
            range(len(ase_atoms.get_positions()))]


def dft_rks(molecule, prev_dm=None):
    mol = gto.Mole()
    # mol.build(atom=molecule, basis='ccpvdz', unit='Angstrom', charge=0, spin=0)
    mol.build(atom=molecule, basis='6-31G', unit='Angstrom', charge=0, spin=0)

    mf = dft.RKS(mol, xc='PBE').density_fit()
    mf.verbose = 0  # quiet mode
    mf.conv_tol = 1e-4

    if prev_dm is not None:
        mf.init_guess = prev_dm

    total_dft_energy = mf.kernel()
    total_dft_force = -1 * mf.nuc_grad_method().grad()

    return total_dft_energy, total_dft_force, mf.make_rdm1()


def ase_atoms_to_pw_input(input_file, ase_atoms, prefix, outdir, pseudo_dir):
    n_atoms = len(ase_atoms.get_positions())

    input_content = f"""&CONTROL
  calculation = 'scf',
  prefix = '{prefix}',
  outdir = '{outdir}',
  pseudo_dir = '{pseudo_dir}',
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 1,
  celldm(1) = 15.0,
  nat = {n_atoms},
  ntyp = 3,
  ecutwfc = 25.0,
  ecutrho = 100.0,
  occupations = 'fixed',
  input_dft = 'pbe',
  vdw_corr = 'grimme-d3',
/
&ELECTRONS
  conv_thr = 1.d-4,
  mixing_beta = 0.7,
/
ATOMIC_SPECIES
  C  12.011   C.pbe-n-kjpaw_psl.1.0.0.UPF
  H  1.008    H.pbe-rrkjus_psl.1.0.0.UPF
  O  15.999   O.pbe-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS {{angstrom}}
"""
    positions = ase_atoms.get_positions()
    symbols = ase_atoms.get_chemical_symbols()
    for i in range(n_atoms):
        input_content += f"  {symbols[i]}  {positions[i, 0]:12.6f}  {positions[i, 1]:12.6f}  {positions[i, 2]:12.6f}\n"

    input_content += """K_POINTS gamma"""

    with open(input_file, 'w') as f:
        f.write(input_content)


def parse_pw_output(output_file):
    tree = etree.parse(output_file)

    # parse total energy, unit: Hartree
    e_tot = float(tree.xpath('//total_energy/etot/text()')[0].strip())

    # parse forces, unit: Hartree/au
    forces_text = tree.xpath('//forces/text()')[1].strip()
    dft_force = np.array([list(map(float, line.split())) for line in forces_text.splitlines()], dtype=float)

    return e_tot, dft_force

