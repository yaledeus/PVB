import torch
from torch.utils.data import DataLoader
import time
import glob
import math
import shutil
import json
import os
import io
import yaml
import pandas as pd

from config import infer_config, dict_to_namespace
from data import *
from utils import load_file
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

    # load refiner
    using_refiner = getattr(args, 'refiner')
    if using_refiner:
        print("Loading refiner...")
        refiner = torch.load(args.refiner, map_location='cpu')
        device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
        refiner.to(device)
        refiner.eval()
        print(f"Refiner loaded.")
    else:
        print("No refiner specified.")

    if args.mode == "single":
        pdbs = [args.name]
        test_set = [args.test_set]
    elif args.mode == "all":
        items = load_file(args.test_set)
        pdbs = [item["pdb"] for item in items]
        test_set = [item["state0_path"] for item in items]
    else:
        raise NotImplementedError(f"test mode {args.mode} not implemented.")
    
    cols = ['PDB', 'TIME']
    res = []

    for pdb_name, pdb_path in zip(pdbs, test_set):
        out_dir = os.path.join(save_dir, pdb_name)
        os.makedirs(out_dir, exist_ok=True)
        state0 = md.load(pdb_path)

        # make test batch
        batch, (atom_index,) = make_batch(pdb_path, args.batch_size)
        batch = to_device(batch, device)

        # save new topology file
        new_top = state0.atom_slice(atom_index)
        new_top.save_pdb(os.path.join(out_dir, f"{pdb_name}.pdb"))

        positions = []

        print(f"Start inference for PDB {pdb_name}.")
        start = time.time()

        with torch.no_grad():
            for step in tqdm(range(args.inf_step)):
                x = model.inference(batch, sde_step=args.sde_step)
                if using_refiner:
                    batch["x0"] = x
                    x = refiner.inference(batch, sde_step=args.sde_step)
                # save positions, Angstrom => nm
                x_numpy = x.cpu().numpy() / 10
                positions.append(x_numpy)
                # update batch
                batch["x0"] = x  # nm => Angstrom

        positions = np.array(positions, dtype=float)    # (T, N, 3)

        md.Trajectory(
            positions,
            new_top.topology
        ).save_xtc(os.path.join(out_dir, f'{pdb_name}_model_ode{args.sde_step}_inf{args.inf_step}.xtc'))

        end = time.time()
        elapsed_time = end - start

        print(f"[*] Inference for PDB {pdb_name} finished, total elapsed time: {elapsed_time}.")

        res.append([pdb_name, elapsed_time])

    df = pd.DataFrame(res, columns=cols)
    mode = 'all' if args.mode == 'all' else args.name
    df.to_csv(os.path.join(save_dir, f'{mode}_ode{args.sde_step}_inf{args.inf_step}.csv'), index=False)


if __name__ == "__main__":
    args = infer_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    main(config)
