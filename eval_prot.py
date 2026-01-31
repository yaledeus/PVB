import torch
import numpy as np
import mdtraj as md
import pandas as pd
from argparse import ArgumentParser
import os
import pickle
import glob
from statsmodels.tsa.stattools import acovf, acf
from utils.tica_utils import *
from utils.backbone_utils import *
from plots import *


def load_traj(trajfile, top):
    _, ext = os.path.splitext(trajfile)
    if ext in ['.pdb']:
        traj = md.load(trajfile)
    elif ext in ['.xtc']:
        traj = md.load(trajfile, top=top)
    elif ext in ['.npz']:
        positions = np.load(trajfile)["positions"]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    elif ext in ['.npy']:
        positions = np.load(trajfile)
        if positions.ndim == 4:
            positions = positions[0]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    else:
        raise NotImplementedError
    return traj


def traj_analysis(traj_model_path, traj_ref_path, top=None, lagtime=10, reduced=False, use_distances=True, plot=False):
    out = {}

    traj_model = load_traj(traj_model_path, top=top)
    traj_ref = load_traj(traj_ref_path, top=top)

    # TICA can be loaded if constructed before
    tica_model_path = os.path.join(os.path.dirname(traj_ref_path), "tica_torus_ldist_model.pic")
    if os.path.exists(tica_model_path):
        with open(tica_model_path, "rb") as f:
            tica_model = pickle.load(f)
    else:
    # lagtime: ATLAS 10 (x10=100ps), mdCATH 1 (x1=1ns)
        tica_model = run_tica(traj_ref, lagtime=lagtime, dim=2, reduced=reduced, use_distances=use_distances)
        with open(tica_model_path, "wb") as f:
            pickle.dump(tica_model, f)
    
    # include replica generated trajectories
    replicas = glob.glob(f'{traj_ref_path[:-5]}*.xtc')
    for replica in replicas:
        if replica == traj_ref_path or replica.endswith('R4.xtc'):  # '{cath_domain}_R4.xtc' for reference
            continue
        traj_rep = load_traj(replica, top=top)
        traj_ref = traj_ref.join(traj_rep)

    # compute tica
    feat_func = tica_features if not reduced else reduced_tica_features
    features, contact_pairs = feat_func(traj_ref, use_distances=use_distances, contact_pairs=None)
    feat_model, _ = feat_func(traj_model, use_distances=use_distances, contact_pairs=contact_pairs)
    tics_ref = tica_model.transform(features)
    tics_model = tica_model.transform(feat_model)
    tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
    tic2d_js = compute_joint_js_distance(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1])

    # compute backbone torus
    torus_ref = compute_torus(traj_ref)
    torus_model = compute_torus(traj_model)
    torus_js = compute_js_distance(torus_ref, torus_model)

    # compute radius of gyration
    rg_ref = compute_radius_of_gyration(traj_ref)
    rg_model = compute_radius_of_gyration(traj_model)
    rg_js = compute_js_distance(rg_ref, rg_model)

    out["torus_js"] = torus_js
    out["rg_js"] = rg_js
    out["tic_js"] = tic_js
    out["tic2d_js"] = tic2d_js

    # compute TIC-0 decorrelation
    acovf_threshold = 0.5
    tic0_ref_mu, tic0_ref_sigma = np.mean(tics_ref[:, 0]), np.std(tics_ref[:, 0])
    tic0_ref_adj = (tics_ref[:, 0] - tic0_ref_mu) / tic0_ref_sigma
    tic0_model_adj = (tics_model[:, 0] - tic0_ref_mu) / tic0_ref_sigma
    autocorr_ref = acovf(tic0_ref_adj, nlag=len(tic0_ref_adj) - 1, adjusted=True, demean=False)
    autocorr_model = acovf(tic0_model_adj, nlag=len(tic0_model_adj) - 1, adjusted=True, demean=False)
    out["tic0_decorr_md"] = np.where(autocorr_ref < acovf_threshold)[0][0]
    out["tic0_decorr_bool"] = 0
    out["tic0_decorr"] = len(tic0_model_adj)
    if autocorr_model[0] > acovf_threshold:
        decorr_indices = np.where(autocorr_model < acovf_threshold)[0]
        if len(decorr_indices) > 0:
            out["tic0_decorr_bool"] = 1
            out["tic0_decorr"] = decorr_indices[0]

    # compute Val-CA
    val_ca, _ = compute_validity(traj_model)
    out["val_ca"] = val_ca

    # compute RMSE contact
    try:
        contact_ref, _ = compute_contact_map(traj_ref)
        contact_model, _ = compute_contact_map(traj_model)
        n_residues = contact_ref.shape[0]
        rmse_contact = np.sqrt(2 / (n_residues * (n_residues - 1)) * np.sum((contact_ref - contact_model) ** 2))
    except BaseException as e:
        print(f"[!] Errno when computing RMSE contact: {e}")
        rmse_contact = 0
    out["rmse_contact"] = rmse_contact

    # Markov state model stuff
    n_clusters = 100
    nstates = 10
    kmeans, ref_kmeans = get_kmeans(tics_ref, n_clusters=n_clusters, max_iter=100)
    try:
        msm, pcca = get_msm(ref_kmeans, lagtime=lagtime, nstates=nstates, n_clusters=n_clusters)
        # assign inactive microstates to the nearest active microstates
        kmeans_wrapper = KMeansWrapper(kmeans, msm.count_model.state_symbols)
        model_discrete = msm_discretize(tics_model, kmeans_wrapper, pcca)
        ref_discrete = msm_discretize(tics_ref, kmeans_wrapper, pcca)
        model_metastatble_probs = (model_discrete == np.arange(nstates)[:, None]).mean(1)
        ref_metastable_probs = (ref_discrete == np.arange(nstates)[:, None]).mean(1)
        msm_js = jensenshannon(model_metastatble_probs, ref_metastable_probs)
    except BaseException as e:
        print(f"[!] Errno when computing MSM: {e}")
        msm_js = 0
    out["msm_js"] = msm_js

    print("========== EVALUATION RESULT ==========")
    for k, v in out.items():
        print(f"{k}: {v:.4f}")

    ### ========= plot figures ========= ###
    if plot:
        pdb = 'anony'   # TODO: replace the name as you want
        name = 'PVB'
        # TIC
        plot_free_energy(tics_ref[:, 0], tics_model[:, 0], f"fig/{pdb}/free_energy_on_tic0.pdf", xlabel='TIC 0', name=name)
        plot_free_energy(tics_ref[:, 1], tics_model[:, 1], f"fig/{pdb}/free_energy_on_tic1.pdf", xlabel='TIC 1', name=name)
        # MSM
        plot_msm(ref_metastable_probs, model_metastatble_probs, name, f"fig/{pdb}/msm.pdf")

    return out


def parse():
    arg_parser = ArgumentParser(description='simulation')
    arg_parser.add_argument('--top', type=str, default=None, help='topology file path (.pdb)')
    arg_parser.add_argument('--ref', type=str, required=True, help='reference trajectory file path')
    arg_parser.add_argument('--model', type=str, required=True, help='model generated trajectory file path')
    arg_parser.add_argument('--lagtime', type=int, help='lagtime for performing TICA')
    arg_parser.add_argument('--reduced', action='store_true', default=False, help='only use backbone dihedrals as projected features')
    arg_parser.add_argument('--use_distances', action='store_true', help='[Optional] using pairwise distances as projected features')
    arg_parser.add_argument('--plot', action='store_true', help='[Optional] plot figures after evaluation')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.model, args.ref, top=args.top, reduced=args.reduced, use_distances=args.use_distances, plot=args.plot)


if __name__ == "__main__":
    main(parse())
