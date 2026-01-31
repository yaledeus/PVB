import torch
from torch.utils.data import DataLoader
from scipy.special import softmax, logsumexp
from openmm.app import PDBFile
import time
import glob
import math
import shutil
import json
import os
import yaml
import pandas as pd

from config import infer_config, dict_to_namespace
from post_process import write_coord_to_sdf
from data import *
from utils import *
from utils.random_seed import setup_seed, SEED

### set backend == "pytorch"
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

setup_seed(SEED)
torch.set_default_dtype(torch.float32)


def create_save_dir(args):
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def main(args):
    save_dir = create_save_dir(args)

    # load model
    print("Loading checkpoints...")
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()
    print(f"Model loaded.")

    if args.mode == 'single':
        items = [args.test_set]
    elif args.mode == "all":
        items = load_file(args.test_set)
    else:
        raise NotImplementedError(f"test mode {args.mode} not implemented.")
    
    cols = ['PDB', 'TIME']
    res = []

    ### pdbbind dataset, protein and ligand were saved in .pdb and .sdf, respectively
    if args.dataset == 'pdbbind':
        for item in items:
            pdb, apo_protein_path, apo_ligand_path = item["pdb"], item["apo_protein_path"], item["apo_ligand_path"]
            out_dir = os.path.join(save_dir, pdb)
            os.makedirs(out_dir, exist_ok=True)
            
            # make test batch
            batch, _ = make_batch_complex_pl(apo_protein_path, apo_ligand_path, args.batch_size)
            batch = to_device(batch, device)

            positions = [ batch["x0"].cpu().numpy() / 10 ]

            print(f"Start inference for PDB {pdb}.")
            start = time.time()

            with torch.no_grad():
                for _ in tqdm(range(args.inf_step)):
                    x = model.inference(batch, sde_step=args.sde_step)
                    # save positions, Angstrom => nm
                    x_numpy = x.cpu().numpy() / 10
                    positions.append(x_numpy)
                    # update batch
                    batch["x0"] = x

            positions = np.array(positions, dtype=float)    # (T, N, 3)
            edge_mask = batch["edge_mask"].detach().cpu().numpy()
            # protein index = 0, ligand index = 1
            protein_pos, ligand_pos = positions[:, edge_mask == 0], positions[:, edge_mask == 1]

            md.Trajectory(
                protein_pos,
                md.load(apo_protein_path).topology
            ).save_xtc(os.path.join(out_dir, f'{pdb}_model_ode{args.sde_step}_inf{args.inf_step}.xtc'))

            ligand = Chem.SDMolSupplier(apo_ligand_path, removeHs=True)[0]
            write_coord_to_sdf(ligand, 10.0 * ligand_pos, save_path=os.path.join(out_dir, f'{pdb}_model_ode{args.sde_step}_inf{args.inf_step}.sdf'))

            end = time.time()
            elapsed_time = end - start

            print(f"[*] Inference for PDB {pdb} finished, total elapsed time: {elapsed_time}.")

            res.append([pdb, elapsed_time])

    ### misato dataset, protein and ligand were saved in one PDB topology file
    elif args.dataset == 'misato':
        for item in items:
            pdb, state0_path = item['pdb'], item['state0_path']
            out_dir = os.path.join(save_dir, pdb)
            os.makedirs(out_dir, exist_ok=True)

            # top_path = os.path.join(
            #     os.path.split(args.test_set)[0],
            #     'parameter_restart_files_MD',
            #     pdb.lower(), f'{pdb}.pdb'
            # )
            state0 = md.load(state0_path)
            batch, (atom_index,) = make_batch_complex_mix(state0_path, args.batch_size)
            batch = to_device(batch, device)

            # save new topology file
            new_top = state0.atom_slice(atom_index)
            new_top.save_pdb(os.path.join(out_dir, f"{pdb}.pdb"))

            positions = []

            print(f"Start inference for PDB {pdb}.")
            start = time.time()

            with torch.no_grad():
                for _ in tqdm(range(args.inf_step)):
                    x = model.inference(batch, sde_step=args.sde_step)
                    # save positions, Angstrom => nm
                    x_numpy = x.cpu().numpy() / 10
                    positions.append(x_numpy)
                    # update batch
                    batch["x0"] = x

            positions = np.array(positions, dtype=float)    # (T, N, 3)

            md.Trajectory(
                positions,
                new_top.topology
            ).save_xtc(os.path.join(out_dir, f'{pdb}_model_ode{args.sde_step}_inf{args.inf_step}.xtc'))

            end = time.time()
            elapsed_time = end - start

            print(f"[*] Inference for PDB {pdb} finished, total elapsed time: {elapsed_time}.")

            res.append([pdb, elapsed_time])

    df = pd.DataFrame(res, columns=cols)
    df.to_csv(os.path.join(save_dir, f'process_time_ode{args.sde_step}_inf{args.inf_step}.csv'), index=False)


if __name__ == "__main__":
    args = infer_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    main(config)
