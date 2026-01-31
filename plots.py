import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
import numpy as np
import os


def create_save_dir(save_path):
    save_dir = os.path.split(save_path)[0]
    os.makedirs(save_dir, exist_ok=True)


def plot_tic2d_kde(tic0, tic1, save_path, tic_range=None, xlabel='TIC0', ylabel='TIC1', name=None):
    create_save_dir(save_path)

    if tic_range is None:
        tic_range = ((tic0.min(), tic0.max()), (tic1.min(), tic1.max()))

    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    sns.kdeplot(
        x=tic0, y=tic1,
        fill=True, cmap="OrRd", levels=50, thresh=0.05,
        ax=ax
    )
    ax.set_xlim(tic_range[0][0], tic_range[0][1])
    ax.set_ylim(tic_range[1][0], tic_range[1][1])
    ax.tick_params(axis='both', labelsize=30)
    ax.set_xlabel(xlabel, color='black', fontsize=50)
    ax.set_ylabel(ylabel, color='black', fontsize=50)
    plt.title(f"{name}", fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_free_energy(torsion_ref, torsion_model, save_path, xlabel='TIC 0', name='PVB'):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))
    feat_bins = np.linspace(torsion_ref.min(), torsion_ref.max(), 100)
    hist_ref, edges_ref = np.histogram(torsion_ref, bins=feat_bins, density=True)
    free_energy_ref = -np.log(hist_ref/hist_ref.max())
    centers_ref = 0.5*(edges_ref[1:] + edges_ref[:-1])
    plt.plot(centers_ref, free_energy_ref, linewidth=4, label='MD', linestyle='-')

    hist_model, edges_model = np.histogram(torsion_model, bins=feat_bins, density=True)
    free_energy_model = -np.log(hist_model/hist_model.max())
    centers_model = 0.5*(edges_model[1:] + edges_model[:-1])
    plt.plot(centers_model, free_energy_model, linewidth=4, label=name, linestyle='--')

    plt.legend(title='', loc='upper right', fontsize=30)

    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel(xlabel, color='black', fontsize=50)
    plt.ylabel("Free energy/$k_B$T", color='black', fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_free_energy_all(tics_list, method_list, save_path, xlabel='TIC 0'):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))
    feat_bins = np.linspace(tics_list[0].min(), tics_list[0].max(), 100)

    for tics, method in zip(tics_list, method_list):
        hist, edges = np.histogram(tics, bins=feat_bins, density=True)
        free_energy = -np.log(hist/hist.max())
        centers = 0.5*(edges[1:] + edges[:-1])
        plt.plot(centers, free_energy, linewidth=4, label=method, linestyle='-' if method == 'MD' else '--')

    plt.legend(title='', loc='upper right', fontsize=30)

    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel(xlabel, color='black', fontsize=50)
    plt.ylabel("Free energy/$k_B$T", color='black', fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_residue_dist_map(res_dist_matrix, save_path, name=None, label=None, reverse=True):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        res_dist_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues_r" if reverse else "Blues",
        cbar_kws={'label': label},
        annot_kws={'size': 25},
        square=True
    )

    # plt.title(name, color='black', fontsize=50)
    plt.xlabel("Residue ID", color='black', fontsize=50)
    plt.ylabel("Residue ID", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(label=label, fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_distance_distribution(adj_dist_list, method_list, save_path, bins=50):
    create_save_dir(save_path)

    sns.set_theme(style="white")
    plt.figure(figsize=(16, 12))

    colors = ['#D62728', '#1F77B4']

    # adj_dist_list[0] default to be reference trajectories
    min_val, max_val = np.min(adj_dist_list[0]), np.max(adj_dist_list[0])
    shared_bins = np.linspace(min_val, max_val, bins)

    for adj_dist, method, color in zip(adj_dist_list, method_list, colors):
        sns.histplot(adj_dist, bins=shared_bins, kde=False, color=color, stat="density", label=method)

    # plt.title("Pairwise distance", fontsize=30)
    plt.xlabel("Distance (Å)", color='black', fontsize=50)
    plt.ylabel("Population", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_metric_over_time(rmsd_list, method_list, save_path, dt=0.01, ylabel='RMSD'):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))

    colors = ['#D62728', '#1F77B4']

    sns.set_theme(style="white", context="talk", font_scale=1.4)
    for rmsd, method, color in zip(rmsd_list, method_list, colors):
        time = np.arange(len(rmsd)).astype(float) * dt # unit: ns
        df = pd.DataFrame({"Time (ns)": time, "rmsd": rmsd})
        sns.lineplot(data=df, x="Time (ns)", y="rmsd", color=color, linewidth=4, label=method)

    plt.xlabel("Time (ns)", color='black', fontsize=50)
    plt.ylabel(f"{ylabel} (Å)", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_ess(ess_list, method_list, save_path, screen=None):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))

    md_ess = []
    for ess, method in zip(ess_list, method_list):
        if method == 'MD':
            md_ess.append(ess)
    md_ess_median = np.median(np.array(md_ess, dtype=float))
    ess_list = np.array(ess_list, dtype=float)
    ess_list /= md_ess_median

    if screen is not None:
        screen_ess_list, screen_method_list = [], []
        for ess, method in zip(ess_list, method_list):
            if method in screen:
                screen_ess_list.append(ess)
                screen_method_list.append(method)
    else:
        screen_ess_list, screen_method_list = ess_list, method_list
    
    # flattern
    N = len(screen_ess_list[0])
    screen_ess_list = screen_ess_list.flatten()
    screen_method_list = [method for method in screen_method_list for _ in range(N)]

    data = {
        "ess": screen_ess_list,
        "method": screen_method_list
    }

    # reference line y = 1
    plt.axhline(y=1, color='#00008B', linestyle='--', linewidth=3)

    sns.boxplot(data=data, x='method', y='ess', showfliers=False,
                boxprops=dict(facecolor='#D2B48C', edgecolor='black'),
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3)
                )

    plt.xlabel("Method", color='black', fontsize=50)
    plt.ylabel("ESS/s", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_msm(msm_prob_ref, msm_prob_model, name, save_path=None, vmin=-0.2, vmax=0.2):
    create_save_dir(save_path)
    
    nstates = len(msm_prob_ref)
    diff = np.array(msm_prob_ref) - np.array(msm_prob_model)
    diff_matrix = np.zeros((nstates, nstates)) * np.nan
    np.fill_diagonal(diff_matrix, diff)

    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(diff_matrix, cmap="RdBu_r", center=0,
                     vmin=vmin, vmax=vmax,
                     xticklabels=[i for i in range(nstates)],
                     yticklabels=[i for i in range(nstates)])
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel(name, fontsize=50)
    plt.ylabel('MD', fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_infer_time():
    def _read(path, steps=1000):
        data = pd.read_csv(path)
        infer_time = data['TIME'].to_numpy()
        infer_time /= steps
        return infer_time
    models = ["PVB", "UniSim", "MDGEN", "ITO"]
    infer_time = [_read("baselines/infer_time_pvb.csv", steps=1000),
                  _read("baselines/infer_time_unisim.csv", steps=1000),
                  _read("baselines/infer_time_mdgen.csv", steps=10),
                  _read("baselines/infer_time_ito.csv", steps=10)
                ]

    data = []
    for model, times in zip(models, infer_time):
        for t in times:
            data.append({"Model": model, "Inference Time (s)": t})

    df = pd.DataFrame(data)

    plt.figure(figsize=(16, 12))
    ax = sns.boxplot(x="Model", y="Inference Time (s)", data=df, palette="muted", showfliers=False)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(int(ymin), int(ymax) + 1, 1))

    sns.despine()
    # plt.title("Inference Time Distribution per Model", fontsize=14)
    plt.ylabel("Inference Time per Step (s)", fontsize=50)
    plt.xlabel("Model", fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("fig/infer_time.pdf")
    plt.clf()


def plot_contact(contacts, save_path, dt=1, cutoff=1.2, name=None):
    create_save_dir(save_path)

    is_contact = contacts < cutoff
    Q = is_contact.sum(axis=1) / is_contact.shape[1]
    time = np.arange(len(Q)).astype(float) * dt # unit: $\mu$s

    df = pd.DataFrame({"Time (ns)": time, "Q": Q})
    sns.set_theme(style="white", context="talk", font_scale=1.4)
    plt.figure(figsize=(16, 12))
    sns.lineplot(data=df, x="Time (ns)", y="Q", color="darkred", linewidth=4)

    plt.fill_between(df["Time (ns)"], df["Q"], color="lightcoral", alpha=0.3)

    plt.ylim(0, 1.05)
    plt.title(name, fontsize=50)
    plt.xlabel("Time (ns)", fontsize=50)
    plt.ylabel("Contacts", fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    plot_infer_time()