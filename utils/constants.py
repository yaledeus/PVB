import numpy as np
import torch

BOND_N_CA = 0.1459  # N-CA bond length, unit: nm
BOND_CA_C = 0.1525  # CA-C
BOND_C_O = 0.1229   # C=O
BOND_C_N = 0.1336   # C-N in peptide bond
ANGLE_N_CA_C = 1.9373   # N-CA-C angle, unit: radius
ANGLE_CA_C_N = 2.0455   # CA-C-N
ANGLE_CA_C_O = 2.0961   # CA-C=O
ANGLE_O_C_N = 2.1415    # O=C-N
ANGLE_C_N_CA = 2.1241   # C-N-CA


# peptide plane centered on N, order: N, CA, C, O
PEPTIDE_PLANE = np.array([[0, 0, 0], [0.1459, 0, 0], [0.20055, -0.14237, 0], [0.123368, -0.23801, 0]], dtype=float)
PEPTIDE_PLANE_TORCH = torch.from_numpy(PEPTIDE_PLANE)


### units, reference: http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
kB = 0.0083144626  # KJ/mole/Kelvin
Ry_to_Hartree = 0.5
Hartree_to_kJ_per_mol = 2625.5
Hartree_to_eV = 27.2107
eV_to_kJ_per_mol = 96.4869
kcal_to_kJ = 4.18400
Bohr_to_Angstrom = 0.529177
### atom reference energy at 300K (unit: eV)
atom_ref_energy = {
    ### http://quantum-machine.org/datasets/, Solvated protein fragments
    'H': -13.717939590030356,
    'C': -1029.831662730747,
    'N': -1485.40806126101,
    'O': -2042.7920344362644,
    'F': -2713.352229159,
    'S': -10831.264715514206,
    ### http://quantum-machine.org/datasets/, SN2 reactions
    'Cl': -12518.663203367176,
    'Br': -70031.09203874589,
    'I': -8096.587166328217
}


def norm_energy(E):
    """
    :param E: raw data of energies
    :return: min-max normalization, scale to [-1,0]
    """
    E = (E - E.max()) / (E.max() - E.min())
    return E


def get_inv_temperature(T):
    return 1 / (kB * T)


MAX_DELTA = 100     # ps
