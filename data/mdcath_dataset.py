### Borrowed from https://github.com/compsciencelab/mdCATH/blob/main/user

import os
from os.path import join as opj
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import hf_hub_url
import h5py as h5
import numpy as np
import pandas as pd
import json
from glob import glob
import mdtraj
from argparse import ArgumentParser

from utils import load_file
from utils.bio_utils import *
from utils.geometry import kabsch_numpy
from data.mmap_dataset import create_mmap
from utils.constants import *


api = HfApi()


def do_download(domain_id, data_root):
    return hf_hub_download(repo_id="compsciencelab/mdCATH", 
                    filename=f"mdcath_dataset_{domain_id}.h5",
                    subfolder='data',
                    local_dir=data_root,
                    repo_type="dataset")


def preprocess_mdcath(data_dir, split='train', temperature='320', max_workers=8):
    split_path = opj(data_dir, f'{split}.txt')
    data_root = opj(data_dir, 'raw')
    process_dir = opj(data_dir, 'processed')

    with open(split_path, 'r') as f:
        domains = [domain.strip() for domain in f.readlines() if domain.strip()]

    os.makedirs(process_dir, exist_ok=True)
    print(f"[*] Start pipeline: {len(domains)} domains, {max_workers} threads\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_domain = {
            executor.submit(do_download, domain_id, data_root): domain_id
            for domain_id in domains
        }

        for future in as_completed(future_to_domain):
            domain_id = future_to_domain[future]
            try:
                h5_path = future.result()
                print(f"[+] Downloaded {domain_id}")

                save_dir = opj(process_dir, domain_id)
                os.makedirs(save_dir, exist_ok=True)

                with h5.File(h5_path, 'r') as f:
                    natoms = f[domain_id].attrs['numProteinAtoms']
                    print(f"{domain_id}: numProteinAtoms = {natoms}")

                    pdbProteinAtoms = f[domain_id]['pdbProteinAtoms'][()].decode('utf-8')
                    with open(pdb_path := opj(save_dir, f'{domain_id}.pdb'), 'w') as fp:
                        fp.write(pdbProteinAtoms)

                    traj = mdtraj.load(pdb_path)
                    top = traj.topology
                    atype, btype, a_index, b_index, bond_index = get_block_from_top(top)

                    repls = list(f[domain_id][temperature].keys())

                    for repl in repls:
                        xyz = f[domain_id][temperature][str(repl)]['coords']
                        for i in range(xyz.shape[0] - 1)[::5]:
                            pos0, pos1 = xyz[i], xyz[i + 1]     # 1 ns spacing
                            # elliminate roto-translational degrees of freedom for (x0, x1)
                            pos1, _, _ = kabsch_numpy(pos1, pos0)
                            x0, b0 = pos0[a_index], pos0[b_index]
                            x1, b1 = pos1[a_index], pos1[b_index]
                            xc = x0.mean(axis=0)
                            x0 = x0 - xc
                            b0 = b0 - xc
                            x1 = x1 - xc
                            b1 = b1 - xc
                            data = {
                                "atype": atype.tolist(),
                                "btype": btype.tolist(),
                                "x0": x0.tolist(),
                                "b0": b0.tolist(),
                                "x1": x1.tolist(),
                                "b1": b1.tolist(),
                                "bond_index": bond_index.tolist(),
                                "env": 13
                            }
                            yield f'{domain_id}_{repl}_{i}', data, [atype.shape[0]]

                os.remove(h5_path)

            except Exception as e:
                print(f"[!] Failed {domain_id}: {e}")


def preprocess_mdcath_test(data_dir, temperature='320', max_workers=8):
    split_path = opj(data_dir, f'test.txt')
    data_root = opj(data_dir, 'raw')
    process_dir = opj(data_dir, 'processed')

    with open(split_path, 'r') as f:
        domains = [domain.strip() for domain in f.readlines() if domain.strip()]
    
    test_data = []

    os.makedirs(process_dir, exist_ok=True)
    print(f"[*] Start pipeline: {len(domains)} domains, {max_workers} threads\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_domain = {
            executor.submit(do_download, domain_id, data_root): domain_id
            for domain_id in domains
        }

        for future in as_completed(future_to_domain):
            domain_id = future_to_domain[future]
            try:
                h5_path = future.result()
                print(f"[+] Downloaded {domain_id}")

                save_dir = opj(process_dir, domain_id)
                os.makedirs(save_dir, exist_ok=True)

                with h5.File(h5_path, 'r') as f:
                    natoms = f[domain_id].attrs['numProteinAtoms']
                    print(f"{domain_id}: numProteinAtoms = {natoms}")

                    pdbProteinAtoms = f[domain_id]['pdbProteinAtoms'][()].decode('utf-8')
                    with open(pdb_path := opj(save_dir, f'{domain_id}.pdb'), 'w') as fp:
                        fp.write(pdbProteinAtoms)

                    traj = mdtraj.load(pdb_path)
                    top = traj.topology
                    atype, btype, a_index, b_index, bond_index = get_block_from_top(top)
                    traj = traj.atom_slice(a_index)
                    traj.save_pdb(pdb_noh_path := opj(save_dir, f'{domain_id}_noh.pdb'))

                    repls = list(f[domain_id][temperature].keys())

                    for repl in repls:
                        xyz = f[domain_id][temperature][str(repl)]['coords']
                        mdtraj.Trajectory(
                            xyz[:, a_index] / 10.0,
                            traj.topology
                        ).save_xtc(opj(save_dir, f'{domain_id}_R{repl}.xtc'))
                    
                    test_data.append({
                        'domain': domain_id,
                        'state0_path': pdb_noh_path,
                        'traj_path': opj(save_dir, f'{domain_id}_R0.xtc')
                    })

            except Exception as e:
                print(f"[!] Failed {domain_id}: {e}")
    
    with open(opj(data_dir, 'test.jsonl'), 'w') as fp:
        for item in test_data:
            item_str = json.dumps(item)
            fp.write(f'{item_str}\n')


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='Directory for dataset split files')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    base = args.data_dir
    for _type in ["train", "val"]:
        create_mmap(
            preprocess_mdcath(base, split=_type),
            os.path.join(base, f"{_type}_block")
        )
    # preprocess_mdcath_test(base)