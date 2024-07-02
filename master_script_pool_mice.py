## import dependencies
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from analysis_utils import update_csv_glm_hmm_engagement 
import pickle
from plot_utils import plot_boxplot, save_plot, set_figure_style
from scipy.stats import mannwhitneyu, skew
from master_script_per_mouse import *
import sys

## obtain mouse names
pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
pth_res = Path(pth_dir, 'results')
pth_res.mkdir(parents=True, exist_ok=True)
#mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) 

sys.path.append(Path(pth_dir, "wiggle/plotting/"))
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
from preprocess.build_analysis_per_mouse import *
from plotting.plot_analysis_per_mouse import *


def sort_mice(mouse_names, data_path):
    """Sort mice based on proportion correct vs # of wheel direction changes per mouse
    The classifications are good, neutral, and bad wigglers.
    
    Args:
        mouse_names (list): list of strings of mouse names
        data_path (str): data path to store files

    """

    wiggler_groups = [classify_mouse_wiggler(subject_name, data_path) for subject_name in mouse_names]

    ## initialize dictionary to store mouse names by category
    wiggler_dict = {"good": [], "neutral": [], "bad": []}

    ## populate dictionary with mouse names based on their classification
    for mouse, group in zip(mouse_names, wiggler_groups):
        wiggler_dict[group].append(mouse)

    ## print #mice per classification
    for category in wiggler_dict:
        print(len(wiggler_dict[category]))

    ## save classified mouse names to .npy files
    for category, names in wiggler_dict.items():
        np.save(Path(data_path).joinpath(f"{category}.npy"), names)

    return wiggler_groups

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

    plot_boxplot(final_data, f"N = {len(mouse_names)} mice", "# of wheel direction changes", "proportion correct", [0,1,2,3,4], "Mann-Whitney", figure_size=(8,8))

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

    grp_total_data = []; grp_std_data = []; grp_raw_total_data = []
    for group in wiggle_groups:
        with open(Path(data_path).joinpath("results", f"{group}.wigglers"), "rb") as f:
            data = pickle.load(f)
            [total_data, std_data, raw_total_data] = data
            grp_total_data.append(total_data); grp_std_data.append(std_data); grp_raw_total_data.append(raw_total_data)

    svfg = plt.figure(figsize=fig_dim)
    set_figure_style()

    xval_shift = [0, 0.2, 0.4]

    for i in range(len(grp_total_data)):
        xval = np.arange(0,len(grp_total_data[i]),1)
        #plt.plot(xval, grp_total_data[i], lw=3, label=f"{wiggle_groups[i]} wigglers (N = {len(grp_raw_total_data[i])} mice)", color=grp_colors[i])
        plt.scatter(xval+xval_shift[i], grp_total_data[i], 60, c=grp_colors[i])
        plt.errorbar(xval+xval_shift[i], grp_total_data[i], yerr=(1.96*grp_std_data[i])/np.sqrt(len(grp_std_data[i])), alpha=1.0, fmt="o", color=grp_colors[i], ecolor=grp_colors[i], elinewidth=3, capsize=0)

    ## plot styling
    stimulus_contrast = ["-100", "-25", "-12.5", "-6.25", "0.0", "6.25", "12.5", "25", "100"]
    plt.ylabel("mean # of wheel direction changes", fontsize=28)
    plt.xlabel("visual stimulus contrast [%]", fontsize=28)
    plt.xticks(np.arange(0,9,1), stimulus_contrast, fontsize=24)
    plt.legend(fontsize=18, loc="upper left")
    sns.despine(trim=False, offset=8)

    ## stats
    print(mannwhitneyu(grp_total_data[0], grp_total_data[2]), "total data")
    print(f"good wigglers-- mean: {np.mean(grp_total_data[0])}, median: {np.median(grp_total_data[2])}, std: {np.std(grp_total_data[2])}")
    print(f"bad wigglers-- mean: {np.mean(grp_total_data[1])}, median: {np.median(grp_total_data[2])}, std: {np.std(grp_total_data[2])}")

    ## stats per contrast group
    for j in range(len(stimulus_contrast)):
        contrast_good = [grp_raw_total_data[0][i][j] for i in range(len(grp_raw_total_data[0]))]
        contrast_bad = [grp_raw_total_data[2][i][j] for i in range(len(grp_raw_total_data[2]))]
        print(mannwhitneyu(contrast_good, contrast_bad), f"data at {stimulus_contrast[j]}% contrast")

    plt.ylim(bottom = 0)
    plt.tight_layout()
    plt.savefig(Path(data_path).joinpath(f"wiggle/results/stats_wiggle_color_plots.png"), dpi=300)


## main script
if __name__=="__main__":

    ## load csv
    df = pd.read_csv(Path(pth_dir, f"glm_hmm_analysis.csv")) #f"glm_hmm_analysis.csv"))
    mouse_names = df["mouse_name"].tolist() 

    ## analysis of mice with glm-hmm data
    #[per_mouse_analysis(subject_name) for subject_name in mouse_names]
    
    ## update csv with wiggler groups
    #wiggler_groups = sort_mice(mouse_names, pth_dir)
    #df["wiggler_group"] = wiggler_groups
    #df.to_csv(Path(pth_dir, f"glm_hmm_analysis.csv")) #f"glm_hmm_analysis.csv"))
   
    ## update csv with glm-hmm states
    #update_csv_glm_hmm_engagement(pth_dir)

    ## wiggler group analysis
    grp_colors = ["#F8766D", "#00BA38", "#619CFF"]
    grp_wiggle = ["good", "neutral", "bad"]

    for i in range(len(grp_wiggle)):
        group = grp_wiggle[i] 
        #plot_glm_hmm_engagement(group, pth_dir, grp_colors[i])
        wigglers = np.load(Path(pth_dir, f"{group}_wigglers.npy"))
        plot_K_feedbackType_group(wigglers, pth_dir, f"{group}_wigglers")
        #total_data, std_data, raw_total_data = save_group_color_plot(wigglers, pth_dir, "avg_prop_wiggle", f"{group}_wigglers", (3,4), (0,3))
        #data = [total_data, std_data, raw_total_data]
        #with open(Path(pth_dir).joinpath("results", f"{group}.wigglers"), "wb") as f:
            #pickle.dump(data, f)
            #print(f"{group} wigglers data pickled")

    #perform_stats_color_plot(pth_dir, (8,8), grp_colors)
