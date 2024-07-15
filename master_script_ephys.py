## is V1 tickled by mouse wiggles?
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ephys_utils import *
from wheel_utils import get_extended_trial_windows
from curate_eids import curate_eids_neural
from decoder_utils import NN
from plot_utils import build_legend, set_figure_style
from prepare_wheelData_csv import prepare_wheel_data_single_csv
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def obtain_eid_data_frame(reg, eid, time_lower, time_upper, alignment_type):
    """Obtain wheel data as dataframe and dictionary for N = 1 session

    :param reg (str): brain region 
    :param eid (str): session
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :param T_BIN: bin size [sec] [int]
    :param alignment_type: "motionOnset" or "stimOnset" [string]

    :return df (pd dataframe): dataframe of session wheel data
    :return d (dict): dictionary of session wheel data

    """

    ## load DataFrame
    df = pd.read_csv(f"/nfs/gatsbystor/naureeng/{reg}/{eid}/{eid}_wheelData.csv")
    d, trial_issue = get_extended_trial_windows(eid, time_lower, time_upper, alignment_type)
    ## drop trials with issues in wheel data from DataFrame
    bad_df = df.index.isin(trial_issue)
    df = df[~bad_df]

    return df, d 

def plot_scatter(axs, data_x, data_y, cstring, title_suffix, xlabel, ylabel, n_trials_K):
    """Scatterplot of data sorted by #changes in wheel direction (K) for N = 1 session

    :param axs (handle): scatterplot axes
    :param data_x (arr): x-axis input
    :param data_y (arr): y-axis input
    :param cstring (list of str): color for each K [1 x 5]
    :param title_suffix (str): title
    :param xlabel (str): x-axis label
    :param ylabel (str): y-axis label
    :param n_trials_K (list): #trials for each K [1 x 5]

    """

    for idx, (x, y) in enumerate(zip(data_x, data_y)):
        axs[idx].scatter(x, y, s=20, color=cstring[idx])
        axs[idx].set_title(f"K = {idx} {title_suffix}", fontweight="bold", fontsize=14)
        axs[idx].set_xlabel(xlabel, fontsize=14)
        axs[idx].set_ylabel(ylabel, fontsize=14)
        axs[idx].set_xlim([0, 10])
        axs[idx].set_ylim([0, 10])
        axs[idx].set_aspect("equal")
        axs[idx].text(0, 0.05, f'{n_trials_K[idx]} trials, μ = {round(np.nanmean(x),2)}, σ = {round(np.nanmean(y),2)}', fontsize=14)
        for axis in ["top", "bottom", "left", "right"]:
            axs[idx].spines[axis].set_linewidth(2)
        

def obtain_neural_data_by_K(df, d, alignment_type, time_lower, time_upper, T_BIN, data_path, reg, eid):
    """Obtain neural data sorted by #changes in wheel direction (K) for N = 1 session

    :param df (pd dataframe): dataframe of session wheel data
    :param d (dict): dictionary of session wheel data
    :param alignment_type: "motionOnset" or "stimOnset" [string]
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :param T_BIN: bin size [sec] [int]
    :param data_path (str): path to store data files
    :param reg (str): brain region
    :param eid (str): session

    :return df (pd dataframe): dataframe of session wheel data
    :return d (dict): dictionary of session wheel data

    """

    ## sort trials by K and set up figure for subplots
    fig_trial, axs_trial = plt.subplots(5, 1, figsize=(15, 15))
    set_figure_style(font_family="Arial", tick_label_size=12, axes_linewidth=2)
    sns.despine(trim=False, offset=4)

    fig_cell, axs_cell = plt.subplots(5, 1, figsize=(15, 15))
    set_figure_style(font_family="Arial", tick_label_size=12, axes_linewidth=2)
    sns.despine(trim=False, offset=4)

    cstring = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    ## save results per K in a dictionary
    results = {}

    mu_data_K = []; std_data_K = []; n_trials_K = []
    mu_cell_K = []; std_cell_K = []; 
    mu_trial_K = []; std_trial_K = [];
    
    for idx, K in enumerate([0, 1, 2, 3, 4]):
        if K == 4:
            df_K = df.query(f"n_extrema>={K} and contrast!=0") ## for K = 4, use K >=4 trials for more data
        else:
            df_K = df.query(f"n_extrema=={K} and contrast!=0") ## obtain K trials

        trials_K = df_K.index.tolist() ## obtain indices of K trials
        dict_K = {key: d[key] for key in trials_K} ## filter dict for K trials

        ## plot psth
        Res = pair_firing_rate_contrast(eid, probe, dict_K, reg, time_lower, time_upper, T_BIN, alignment_type)
        n_units, n_bins, n_trials = Res.shape

        ## compute summary statistics 
        mu_cell = [np.nanmean(np.nanmean(Res[i,:,:], axis=1))/T_BIN for i in range(n_units)]
        st_cell = [np.nanstd(np.nanstd(Res[i,:,:], axis=1))/T_BIN for i in range(n_units)]

        mu_trial = [np.nanmean(np.nanmean(Res[:,:,i], axis=0))/T_BIN for i in range(n_trials)]
        st_trial = [np.nanstd(np.nanstd(Res[:,:,i], axis=0))/T_BIN for i in range(n_trials)]

        mu_cell_K.append(mu_cell); std_cell_K.append(st_cell);
        mu_trial_K.append(mu_trial); std_trial_K.append(st_trial);

        mu_data, std_data, n_trials_data = obtain_psth(Res, time_lower, time_upper, T_BIN)
        mu_data_K.append(mu_data); std_data_K.append(std_data); n_trials_K.append(n_trials_data)

        ## store results in dictionary
        results[K] = {
                "mu_cell": np.nanmean(mu_cell),
                "st_cell": np.nanmean(st_cell),
                "mu_trial": np.nanmean(mu_trial),
                "st_trial": np.nanmean(st_trial),
                "n_units": n_units, 
                "n_trials": n_trials
                }

    ## trial plot
    plot_scatter(axs_trial, mu_trial_K, std_trial_K, cstring, f"(N = {n_units} units)", r'$\mu_{trial}$', r'$\sigma_{trial}$', n_trials_K)
    fig_trial.tight_layout()
    plot_path_trial = Path(data_path) / reg / f"{eid}/{eid}_K_trial.png"
    plot_path_trial.parent.mkdir(parents=True, exist_ok=True)
    fig_trial.savefig(plot_path_trial, dpi=300, bbox_inches="tight")
    print(f"{reg}, {eid}: scatterplot by K per trial saved")

    ## cell plot
    plot_scatter(axs_cell, mu_cell_K, std_cell_K, cstring, f"(N = {n_units} units)", r'$\mu_{cell}$', r'$\sigma_{cell}$', n_trials_K)
    fig_cell.tight_layout()
    plot_path_cell = Path(data_path) / reg / f"{eid}/{eid}_K_cell.png"
    plot_path_cell.parent.mkdir(parents=True, exist_ok=True)
    fig_cell.savefig(plot_path_cell, dpi=300, bbox_inches="tight")
    print(f"{reg}, {eid}: scatterplot by K per cell saved")

    ## save results
    with open(Path(data_path) / reg / f'{eid}/results_K_trial.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"{reg}, {eid}: results by K per trial pickled")

    return mu_data_K, std_data_K, n_trials_K, n_units


def perform_decoder_analysis(eid, probe, df, d, reg, time_lower, time_upper, T_BIN, alignment_type):
    """Decoding for N = 1 session
    
    :param eid (str): session
    :param probe (str): probe # ["probe00" or "probe01"]
    :param df (pd dataframe): dataframe of session wheel data
    :param d (dict): dictionary of session wheel data
    :param reg (str): brain region
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :param T_BIN: bin size [sec] [int]
    :param alignment_type: "motionOnset" or "stimOnset" [string]

    :return acs [list]: decoder accuracies

    """

    ## obtain decoder training input
    Res = pair_firing_rate_contrast(eid, probe, d, reg, time_lower, time_upper, T_BIN, alignment_type)
    n_units, n_bins, n_trials = Res.shape
    Res_single_trial = np.nanstd(Res, axis=0)
    print(Res_single_trial.shape)

    ## obtain decoder training output
    ## stimulus side: contrast sign
    y0 = np.sign(df["contrast"].values)
    print(y0.shape)
    num_right = np.sum(y0 == -1.0)
    num_left  = np.sum(y0 == 1.0)
    num_zero  = np.sum(y0 == 0.0)
    print(f"N = {len(df)} trials: {num_right} right side, {num_left} left side, {num_zero} zero contrast")

    ## decoder for N = 1 session
    ## input: [-1, 1] sec VISp neural data aligned to motionOnset
    ## output: P(stimulus side) for each trial

    acs = NN(Res_single_trial.T, y0.T, "LR", shuf=True)

    return acs


def plot_decoder_analysis(fold_accu, cts, n_units, total_trials, data_path, reg, eid):
    """Plot decoding for N = 1 session (such that one data point is one fold)

    :param fold_accu (list): decoder accuracies in 6 folds [1 x 6] 
    :param cts (float): #trials per change in wheel direction (K)
    :param n_units (float): #units
    :param total_trials (float): total #trials 
    :param data_path (str): path to store data files
    :param reg (str): brain region
    :param eid (str): session

    """

    m1 = [np.mean(fold_accu[i]) for i in range(len(fold_accu))]
    st1 = [np.std(fold_accu[i]) for i in range(len(fold_accu))]

    ## plot of decoding results with P(stimulus side) for each trial
    svfg = plt.figure(figsize=(10,8))
    set_figure_style()
    sns.boxplot(data=fold_accu, showfliers=False, linewidth=3)
    sns.stripplot(data=fold_accu, jitter=True, edgecolor="gray", linewidth=1)
    build_legend(m1, st1, cts)
    plt.ylabel("decoder accuracy", fontsize=28)
    plt.xlabel("#changes in wheel direction", fontsize=28)
    plt.text(0.0, 1.0, f"N = {n_units} units, {total_trials} trials, 1 session", fontsize=28, ha="left", fontweight="bold")
    plt.ylim([0,1])
    sns.despine(trim=False, offset=8)
    plt.tight_layout()

    ## save plot
    plt.savefig(Path(data_path) / reg / f"{eid}/{eid}_decoder_motionOnset.png", dpi=300)
    print(f"decoder analysis saved")


## parameters
reg = "VISp"
time_lower = -1 ## (sec) pre
time_upper = 1 ## (sec)  post
T_BIN = 0.02 ## (sec) 20 msec bin size
alignment_type = "motionOnset" ## align to (time 0)

## get sessions
data_path = "/nfs/gatsbystor/naureeng/"
eids, probes = sessions_with_region(reg, one=one)
print(len(eids), "sessions")

#eids_final = []
#probes_final = []

for i in range(0,1):
    ## compute drift
    eid = eids[i]
    probe = probes[i]

    ## compute drift
    #max_drift = compute_maximum_drift(eid, probe)
    #print(f"{eid}: {max_drift} microns of maximum drift")

    ## build csv
    #prepare_wheel_data_single_csv(reg, eid, data_path)

    ## obtain dataframe
    df_motionOnset, d_motionOnset = obtain_eid_data_frame(reg, eid, time_lower, time_upper, "motionOnset")
    df_stimOnset, d_stimOnset = obtain_eid_data_frame(reg, eid, time_lower, time_upper, "stimOnset")

    ## plot psth
    mu_data_K, std_data_K, n_trials_K, n_units = obtain_neural_data_by_K(df_stimOnset, d_stimOnset, "stimOnset", time_lower, time_upper, T_BIN, data_path, reg, eid)
    #plot_psth(mu_data_K, std_data_K, n_trials_K, time_lower, time_upper, T_BIN, n_units, "stimOnset", data_path, reg, eid)
    
    mu_data_K, std_data_K, n_trials_K, n_units = obtain_neural_data_by_K(df_motionOnset, d_motionOnset, "motionOnset", time_lower, time_upper, T_BIN, data_path, reg, eid)
    #plot_psth(mu_data_K, std_data_K, n_trials_K, time_lower, time_upper, T_BIN, n_units, "motionOnset", data_path, reg, eid)

    ## decoding

    fold_outputs = perform_decoder_analysis(eid, probe, df_motionOnset, d_motionOnset, reg, time_lower, time_upper, T_BIN, "motionOnset")
    df_fold_outputs = pd.DataFrame(fold_outputs)
    df_fold_outputs["trial_no"] = np.arange(0, len(df_fold_outputs), 1)
    y0 = np.sign(df_motionOnset["contrast"].values)
    df_motionOnset["stimulus_side"] = y0

    df_final = pd.merge(df_motionOnset, df_fold_outputs)

    fold_accu = []
    cts = []
    results = {}

    for K in [0, 1, 2, 3, 4]:
        if K == 4: 
            df_K = df_final.query(f"n_extrema>={K} and contrast!=0")
        else:
            df_K = df_final.query(f"n_extrema=={K} and contrast!=0")

        fold_accu.append([np.mean(df_K["stimulus_side"] == df_K[i]) for i in np.arange(0,5,1)])
        cts.append(len(df_K))

        ## store results in dictionary
        results[K] = {
                "mu_decoder": np.nanmean(fold_accu),
                "st_decoder": np.nanstd(fold_accu),
                "n_units": n_units, 
                "n_trials": len(df_K)
                }

    plot_decoder_analysis(fold_accu, cts, n_units, sum(cts), data_path, reg, eid)

    ## save results
    with open(Path(data_path) / reg / f'{eid}/results_K_decoder.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"{reg}, {eid}: decoder results by K pickled")

    ## save eid and probe
    #eids_final.append(eid)
    #probes_final.append(probe)


## store eids and probes
#np.save(Path(data_path) / reg / f"{reg}_eids_final.npy", eids_final)
#np.save(Path(data_path) / reg / f"{reg}_probes_final.npy", probes_final)
#print(f"{reg}: {len(eids_final)} sessions included in decoding analysis")
