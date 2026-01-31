import yaml
import os
import pandas as pd
import numpy as np
from config import infer_config, dict_to_namespace
from utils.io import load_file


def analyze(metric, metrics_list):
    metrics_list = np.array(metrics_list)
    metrics_mean = np.mean(metrics_list)
    metrics_2A = (metrics_list < 2).sum() / metrics_list.shape[0]
    metrics_3A = (metrics_list < 3).sum() / metrics_list.shape[0]
    metrics_4A = (metrics_list < 4).sum() / metrics_list.shape[0]
    metrics_5A = (metrics_list < 5).sum() / metrics_list.shape[0]

    metrics_25 = np.percentile(metrics_list, 25)
    metrics_50 = np.percentile(metrics_list, 50)
    metrics_75 = np.percentile(metrics_list, 75)
    # print(f"-----------------------------{method}-----------------------------")
    print(f"num samples: {len(metrics_list)}")
    print(f"{metric}: {metrics_mean}, {metric} < 2A: {metrics_2A}, {metric} < 3A: {metrics_3A}, {metric} < 4A: {metrics_4A}, {metric} < 5A: {metrics_5A}, "
          f"{metric} 25%: {metrics_25}, {metric} 50%: {metrics_50}, {metric} 75%: {metrics_75}")


def main(args):
    test_pdbs = load_file(args.test_set)
    gen_dir = args.gen_dir
    postfix = f'model_ode{args.sde_step}_inf{args.inf_step}'
    save_path = os.path.join(gen_dir, f'{postfix}.csv')
    if args.data_type == "protein" or args.data_type == "misato":
        if args.data_type == "protein":
            from eval_prot import traj_analysis
        elif args.data_type == "misato":
            from eval_complex import traj_analysis
        res_list = []
        for pdb in test_pdbs:
            name, top, ref = pdb['pdb'], pdb['state0_path'], pdb['traj_path']
            model = os.path.join(gen_dir, name, f'{name}_{postfix}.xtc')
            res = traj_analysis(model, ref, top=top, lagtime=args.lagtime, reduced=args.reduced, use_distances=args.use_distances)
            res["PDB"] = name
            res_list.append(res)
    elif args.data_type == "pdbbind":
        from eval_dock import traj_analysis
        res_list = []
        for pdb in test_pdbs:
            name, protein_path, ligand_path, pocket_idx = pdb['pdb'], pdb["ref_protein_path"], pdb["ref_ligand_path"], pdb['pocket_idx']
            gen_protein_path = os.path.join(gen_dir, name, f'{name}_{postfix}.xtc')
            gen_ligand_path = os.path.join(gen_dir, name, f'{name}_{postfix}.sdf')
            res = traj_analysis(gen_protein_path, gen_ligand_path, protein_path, ligand_path, pocket_idx, pdb=name)
            res["PDB"] = name
            res_list.append(res)
    else:
        raise NotImplementedError(f"Data type {args.data_type} not recognized.")
    df = pd.DataFrame(res_list)
    df.to_csv(save_path, index=False)
    # output to terminal
    cols = df.columns.tolist()
    for column in cols:
        if column == "PDB":
            continue
        if 'rmsd' in column:
            analyze(column, df[column])
        else:
            mean = df[column].mean()
            std = df[column].std()
            print(f"{column}: mean {mean}, std {std}")


if __name__ == "__main__":
    args = infer_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    main(config)
