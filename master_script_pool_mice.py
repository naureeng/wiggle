## import dependencies
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from analysis_utils import compute_pearsonr_wheel_accu, update_csv_glm_hmm_engagement 
import pickle
from plot_utils import plot_boxplot, save_plot, set_figure_style
from scipy.stats import mannwhitneyu, skew
from master_script_per_mouse import *
import sys

## obtain mouse names
pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
pth_res = Path(pth_dir, 'results')
pth_res.mkdir(parents=True, exist_ok=True)
mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) ## load mouse_names

sys.path.append(Path(pth_dir, "wiggle/plotting/"))
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
from preprocess.build_analysis_per_mouse import *
from plotting.plot_analysis_per_mouse import *

def compute_pearsonr_across_mice(mouse_names, data_path):
    """Compute Pearson r correlation coefficent between # of wheel direction changes and proportion correct across mice

    Args:
        mouse_names (list): list of strings of mouse names

    """
    mice_pearsonr_wheel_accu = []
    for subject_name in mouse_names:
        try:
            r = compute_pearsonr_wheel_accu(subject_name, data_path)
            mice_pearsonr_wheel_accu.append(r) 
        except Exception as e:
            print(f"Error processing mouse: {str(e)}")

    np.save(Path(data_path).joinpath(f"mice_pearsonr_wheel_accu.npy"), mice_pearsonr_wheel_accu)
    print(f"pearson r correlation cofficients saved for N = {len(mouse_names)} mice")

def sort_mice(mouse_names, data_path):
    """Sort mice by Pearson r correlation coefficient between # of wheel direction changes and proportion correct across mice
    "good wigglers" (positive correlation), "neutral wigglers" (negligible correlation), "bad wigglers" (negative correlation)
    
    Args:
        mouse_names (list): list of strings of mouse names
        data_path (str): data path to store files

    """
    mice_pearsonr_wheel_accu = np.load(Path(data_path).joinpath(f"mice_pearsonr_wheel_accu.npy"))
    pos_idx = np.where(mice_pearsonr_wheel_accu >= 0.5)[0]
    neu_idx = np.where( (mice_pearsonr_wheel_accu < 0.5) & (mice_pearsonr_wheel_accu > -0.3))[0]
    neg_idx = np.where(mice_pearsonr_wheel_accu <= -0.3)[0]

    pos_mouse_names = [mouse_names[i] for i in pos_idx]
    neu_mouse_names = [mouse_names[i] for i in neu_idx]
    neg_mouse_names = [mouse_names[i] for i in neg_idx]

    print(len(pos_mouse_names), "good wigglers")
    print(len(neu_mouse_names), "neutral wigglers")
    print(len(neg_mouse_names), "bad wigglers")

    ## save data
    np.save(Path(data_path).joinpath(f"good_wigglers.npy"), pos_mouse_names)
    np.save(Path(data_path).joinpath(f"neu_wigglers.npy"), neu_mouse_names)
    np.save(Path(data_path).joinpath(f"bad_wigglers.npy"), neg_mouse_names)

def plot_K_feedbackType_group(mouse_names, data_path, img_name):
    """Plot # of wheel direction changes vs proportion correct in group of mice

    Args:
        mouse_names (list): list of strings of mouse names
        data_path (str): data path to store files
        img_name (str): file name to add to data_path

    """

    bwm_data = []
    for i in range(len(mouse_names)):
        subject_name = mouse_names[i]
        with open (Path(data_path).joinpath(f"results/{subject_name}.k_groups_feedbackType"), "rb") as f:
            x = pickle.load(f)
            data = x[6:11] ## indices of low contrast values
            bwm_data.append(data)

    final_data = pd.DataFrame(bwm_data)
    final_data = final_data.dropna()

    bwm_df = pd.read_csv(Path(data_path).joinpath("final_eids/mouse_names.csv"))
    final_data["Count"] = bwm_df["n_sessions"] ## to weigh boxplot data by #sessions

    plot_boxplot(final_data, f"N = {len(mouse_names)} mice", "# of wheel direction changes", "proportion correct", [0,1,2,3,4], "Mann-Whitney", figure_size=(10,8))
    save_plot("/nfs/gatsbystor/naureeng", f"{img_name}_low_contrast_feedbackType.png")

def save_group_color_plot(mouse_names, data_path, file_name, group_name, fig_dim, color_map_lim):
    """Plot scaled color plot in group of mice

    Args:
        mouse_names (list): list of strings of mouse names
        data_path (str): data path to store files
        group_name (str): file name to add to data_path
        fig_dim (arr): (width, height) of figure
        color_map_lim (list): [lower bound, upper bound] of color_map

    """

    data_extrema = []
    for i in range(len(mouse_names)):
        subject_name = mouse_names[i]
        avg_mouse_data = np.load(Path(pth_dir).joinpath(f"{subject_name}/{subject_name}_{file_name}.npy"))
        if len(avg_mouse_data)==9:
            data_extrema.append(avg_mouse_data)
    
    group_data = np.mean(data_extrema[1:], axis=0) ## remove data not appended with remainder of matrices
    std_data = np.std(data_extrema[1:], axis=0)
    avg_group = pd.DataFrame(group_data)
    plot_color_plot("", avg_group.T, "coolwarm", data_path, fig_dim, color_map_lim, group_name)
    plt.close("all")
    
    return np.squeeze(group_data), np.squeeze(std_data), data_extrema[1:]

def perform_stats_color_plot(data_path, fig_dim, grp_colors):
    """Perform statistical analysis on color-scaled plot analysis across groups

    Args:
        data_path (str): data path to store files
        fig_dim (arr): (width, height) of figure
        grp_colors (list of str): hex code color palette for wiggle groups

    """

    wiggle_groups = ["good", "neutral", "bad"]

    grp_total_data = []; grp_std_data = []; grp_data_85 = []; grp_std_85 = []; grp_raw_total_data = []; grp_raw_data_85 = []; 
    for group in wiggle_groups:
        with open(Path(data_path).joinpath("results", f"{group}.wigglers"), "rb") as f:
            data = pickle.load(f)
            [total_data, std_data, data_85, std_85, raw_total_data, raw_data_85] = data
            grp_total_data.append(total_data); grp_std_data.append(std_data); grp_data_85.append(data_85); grp_std_85.append(std_85); 
            grp_raw_total_data.append(raw_total_data); grp_raw_data_85.append(raw_data_85)

    svfg = plt.figure(figsize=fig_dim)
    set_figure_style()

    for i in range(len(grp_total_data)):
        xval = np.arange(0,len(grp_total_data[i]),1)
        plt.plot(xval, grp_data_85[i], lw=3, label=f"{wiggle_groups[i]} wigglers 85% sessions (N = {len(grp_raw_data_85[i])} mice)", color=grp_colors[i])
        plt.scatter(xval, grp_data_85[i], 60, c=grp_colors[i])
        plt.errorbar(xval, grp_data_85[i], yerr=grp_std_85[i], alpha=0.3, fmt="o", color=grp_colors[i], ecolor=grp_colors[i], elinewidth=3, capsize=0)

    ## plot styling
    stimulus_contrast = ["-100", "-25", "-12.5", "-6.25", "0.0", "6.25", "12.5", "25", "100"]
    plt.ylabel("mean # of wheel direction changes", fontsize=28)
    plt.xlabel("visual stimulus contrast [%]", fontsize=28)
    plt.xticks(np.arange(0,9,1), stimulus_contrast, fontsize=24)
    plt.legend(fontsize=18, loc="upper left")
    sns.despine(trim=False, offset=8)

    ## stats
    print(mannwhitneyu(grp_total_data[0], grp_total_data[2]), "total data")
    print(mannwhitneyu(grp_data_85[0], grp_data_85[2]), "85% data only")
    print(f"good wigglers-- mean: {np.mean(grp_data_85[0])}, median: {np.median(grp_data_85[2])}, std: {np.std(grp_data_85[2])}")
    print(f"bad wigglers-- mean: {np.mean(grp_data_85[1])}, median: {np.median(grp_data_85[2])}, std: {np.std(grp_data_85[2])}")

    ## stats per contrast group
    for j in range(len(stimulus_contrast)):
        contrast_good = [grp_raw_data_85[0][i][j] for i in range(len(grp_raw_data_85[0]))]
        contrast_bad = [grp_raw_data_85[2][i][j] for i in range(len(grp_raw_data_85[2]))]
        print(mannwhitneyu(contrast_good, contrast_bad), f"85% data at {stimulus_contrast[j]}% contrast")

    plt.ylim(bottom = 0)
    plt.tight_layout()
    plt.savefig(Path(data_path).joinpath(f"wiggle/results/stats_wiggle_color_plots.png"), dpi=300)


## main script
if __name__=="__main__":

    #[per_mouse_analysis(subject_name) for subject_name in mouse_names]
    #update_csv_glm_hmm_engagement(pth_dir)

    #compute_pearsonr_across_mice(mouse_names, pth_dir)
    #sort_mice(mouse_names, pth_dir)

    bwm_data = [compute_K_data(mouse_names[i], pth_dir) for i in range(50)]
    print(bwm_data)
    #np.save(Path(pth_dir).joinpath("bwm_K_data.npy"), bwm_data)

    #bwm_data = np.load(Path(pth_dir).joinpath("bwm_K_data.npy"))
    #print(bwm_data)
    #plot_boxplot(pd.DataFrame(bwm_data), "speed", "#changes in wheel direction", "visual speed vs accuracy correlation (r)", [0, 1, 2], "Mann-Whitney", figure_size=(10,8))

"""
    grp_colors = ["#F8766D", "#00BA38", "#619CFF"]
    grp_wiggle = ["good", "neutral", "bad"]

    for i in range(len(grp_wiggle)):
        group = grp_wiggle[i] 
        plot_glm_hmm_engagement(group, pth_dir, grp_colors[i])
        wigglers = np.load(Path(pth_dir, f"{group}_wigglers.npy"))
        plot_K_feedbackType_group(wigglers, pth_dir, f"{group}_wigglers")
        total_data, std_data, raw_total_data = save_group_color_plot(wigglers, pth_dir, "avg_prop_wiggle", f"{group}_wigglers", (3,4), (0,3))
        data_85, std_85, raw_data_85 = save_group_color_plot(wigglers, pth_dir, "avg_prop_wiggle_85", f"{group}_85_wigglers", (3,4), (0,3))
        data = [total_data, std_data, data_85, std_85, raw_total_data, raw_data_85]
        with open(Path(pth_dir).joinpath("results", f"{group}.wigglers"), "wb") as f:
            pickle.dump(data, f)
            print(f"{group} wigglers data pickled")

    perform_stats_color_plot(pth_dir, (8,8), grp_colors)
"""
