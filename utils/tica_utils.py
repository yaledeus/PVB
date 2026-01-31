import numpy as np
import deeptime as dt
import mdtraj as md
import emcee

SELECTION = "symbol == C or symbol == N or symbol == S"


def compute_distances(traj, cutoff=0.45, contact_pairs=None):
    ### for heavy atoms, the contact cutoff = 4.5/5.0 Angstrom
    ### ref: https://link.springer.com/article/10.1186/1471-2105-13-292
    if contact_pairs is None:
        top = traj.topology
        N = traj.n_atoms
        
        # build covalent neighbor map
        covalent_pairs = set()
        # 1-2 neighbors
        for bond in top.bonds:
            covalent_pairs.add((min(bond[0].index, bond[1].index),
                                max(bond[0].index, bond[1].index)))
        # build neighbor dict for 1-2
        neighbors = {atom.index: set() for atom in top.atoms}
        for i,j in covalent_pairs:
            neighbors[i].add(j)
            neighbors[j].add(i)
        # 1-3 pairs
        for atom in neighbors:
            for n1 in neighbors[atom]:
                second_neighbors = neighbors[n1] - {atom}
                for n2 in second_neighbors:
                    covalent_pairs.add((min(atom, n2), max(atom, n2)))
        # candidate pairs (all heavy atom pairs)
        pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N) if (i, j) not in covalent_pairs], dtype=np.compat.long)
        dist0 = md.compute_distances(traj[0], pairs)
        contact_mask = dist0[0] < cutoff
        contact_pairs = pairs[contact_mask]
    
    # compute distances
    distances = md.compute_distances(traj, contact_pairs)
    # distances = md.compute_distances(traj, [[i, j] for i in range(N) for j in range(N) if i < j])
    return distances, contact_pairs


def wrap(array):
    return np.sin(array), np.cos(array)


def reduced_tica_features(trajectory, use_distances=False, selection=SELECTION, contact_pairs=None):
    _, phi = md.compute_phi(trajectory)
    _, psi = md.compute_psi(trajectory)
    torus = np.concatenate([phi, psi], axis=1)
    dihedrals = np.concatenate([*wrap(torus)], axis=1)
    # dihedrals = np.concatenate([*wrap(phi), *wrap(psi)], axis=-1)
    return dihedrals, None


def tica_features(trajectory, use_dihedrals=True, use_distances=True, selection=SELECTION, contact_pairs=None):
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        # _, omega = md.compute_omega(trajectory)
        _, chi1 = md.compute_chi1(trajectory)
        _, chi2 = md.compute_chi2(trajectory)
        _, chi3 = md.compute_chi3(trajectory)
        _, chi4 = md.compute_chi4(trajectory)
        _, chi5 = md.compute_chi5(trajectory)
        torus = np.concatenate([phi, psi, chi1, chi2, chi3, chi4, chi5], axis=1)
        dihedrals = np.concatenate([*wrap(torus)], axis=1)
        # dihedrals = np.concatenate([*wrap(phi), *wrap(psi)], axis=-1)
    if use_distances:
        trajectory = trajectory.atom_slice(trajectory.top.select(selection))
        ca_distances, contact_pairs = compute_distances(trajectory, contact_pairs=contact_pairs)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1), contact_pairs
    elif use_distances:
        return ca_distances, contact_pairs
    elif use_dihedrals:
        return dihedrals, None
    else:
        return [], None


def run_tica(trajectory, lagtime=500, dim=40, reduced=False, use_distances=False, contact_pairs=None):
    if not reduced:
        feats, _ = tica_features(trajectory, use_distances=use_distances, contact_pairs=contact_pairs)
    else:
        feats, _ = reduced_tica_features(trajectory, use_distances=use_distances, contact_pairs=contact_pairs)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(feats).fetch_model()
    tica_model = tica.fit(feats, reweighting_model).fetch_model()
    return tica_model


def reweigh_by_free_energy(traj, tica_model, bins=100):
    features = tica_features(traj)
    tics_ref = tica_model.transform(features)[:, 0]
    feat_bins = np.linspace(tics_ref.min(), tics_ref.max(), bins)
    hist_ref, edges_ref = np.histogram(tics_ref, bins=feat_bins, density=True)
    weight = hist_ref.max() / hist_ref
    bin_indices = np.digitize(tics_ref, edges_ref[:-1], right=False) - 1
    return weight[bin_indices]


def get_vamp2(traj, lag):
    feats = tica_features(traj)
    vamp = dt.decomposition.VAMP(lag).fit_fetch(feats)
    vamp2_score = vamp.score(2)
    return vamp2_score


def ESS(TIC, axis=0):
    """
    :param TIC: TIC of T sampling steps, (T, tic_dim)
    :return: effective sample size
    """
    T = TIC.shape[0]
    tau = emcee.autocorr.integrated_time(TIC[:, axis], quiet=True)[0]
    return T / tau


def get_kmeans(feats, n_clusters=100, max_iter=100):
    kmeans = dt.clustering.KMeans(n_clusters=n_clusters, max_iter=max_iter, fixed_seed=137).fit_fetch(feats)
    return kmeans, kmeans.transform(feats)


class KMeansWrapper:
    def __init__(self, kmeans, active_states):
        self.kmeans = kmeans
        self.active_states = active_states
        self.microstate_map = self.assign_to_active_state()
    
    def assign_to_active_state(self) -> np.ndarray:
        cluster_centers = self.kmeans.cluster_centers   # (n_micro, dim)
        active_centers = cluster_centers[self.active_states]    # (n_active, dim)
        
        diff = cluster_centers[:, None, :] - active_centers[None, :, :]  # (n_micro, n_active, dim)
        dist_sq = np.sum(diff**2, axis=2) # (n_micro, n_active)
        nearest_rel_idx = np.argmin(dist_sq, axis=1)  # (n_micro,)
        
        return nearest_rel_idx

    def transform(self, feats):
        trans_feats = self.kmeans.transform(feats)
        return self.microstate_map[trans_feats]


def get_msm(feats, lagtime=1000, nstates=10, n_clusters=100):
    msm = dt.markov.msm.MaximumLikelihoodMSM(lagtime=lagtime).fit_fetch(feats)
    # msm = pyemma.msm.estimate_markov_model(feats, lag=lag)
    pcca = msm.pcca(nstates)
    print(f"pcca assignments: {len(pcca.assignments)}")
    # assert len(pcca.assignments) == n_clusters
    # cmsm = dt.markov.msm.MaximumLikelihoodMSM(lagtime=lagtime).fit_fetch(pcca.assignments[feats])
    # cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[feats], lag=lag)
    return msm, pcca


def msm_discretize(feats, kmeans, pcca):
    return pcca.assignments[kmeans.transform(feats)]