import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from numpy import nan
from scipy.stats import pearsonr
from itertools import chain
import pickle

pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle"))
from plot_utils import *

def plot_xval_vs_yval(subject_name, data_path, xval, yval, xstr, ystr, tstr, pstr):
    """Obtains scatterplot of two variables with line of best fit and Pearson correlation coefficient

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files
        xval (list): independent variable values
        yval (list): dependent variables values
        xstr (str): x-axis label
        ystr (str): y-axis label
        tstr (str): plot title
        pstr (str): path folder name
    
    """
    svfg = plt.figure(figsize=(10,8))
    set_figure_style()

    ## convert inputs to numpy arrays
    xval = np.array(xval)
    yval = np.array(yval)

    ## scatter plot
    plt.scatter(xval, yval, 200, c="dimgray", alpha=0.5)

    ## Pearson correlation
    r, p_val = pearsonr(xval, yval)
    a, b = np.polyfit(xval, yval, 1)
    plt.plot(xval, xval*a + b, lw=2, c="black")
    plt.xlabel(xstr, fontsize=28)

    ## plot labels and title
    plt.ylabel(ystr, fontsize=28)
    plt.title(f"{tstr} (r = {round(r,2)}, p-val = {round(p_val,2)})", fontweight="bold", fontsize=28)

    ## adjust plot limits
    plt.xlim(left=0)
    plt.ylim([0.0,1.0])

    ## remove spines
    sns.despine(trim=False, offset=8)

    ## tight layout
    plt.tight_layout()

    ## save plot
    pth_res = Path(data_path, f"wiggle/results/{pstr}/")
    pth_res.mkdir(parents=True, exist_ok=True)
    svfg.savefig(Path(pth_res, f"{subject_name}_{pstr}.png"), dpi=300)


def plot_color_plot(subject_name, data_n_extrema_mouse, color_map, data_path, fig_dim, color_map_lim, img_name):
    """Plot scaled colorplot of data

    Args:
        subject_name (str): mouse name
        data_n_extrema_mouse (list): sessions x stimulus contrast on mean # of changes in wheel direction
        color_map (str): heatmap for data
        data_path (str): data path to store files 
        fig_dim (arr): (width, height) of figure
        color_map_lim (list): [lower bound, upper bound] of color_map
        img_name (str): image name to include in stored data_path

    """
    svfg = plt.figure(figsize=fig_dim)
    plt.imshow(data_n_extrema_mouse, cmap=color_map)
    plt.colorbar(orientation="vertical").set_label(label="mean # of wheel direction changes", size=18)
    plt.clim(color_map_lim)
    plt.axis("off")
    plt.tight_layout()

    ## save plot
    pth_res = Path(data_path).joinpath(f"wiggle/results/{img_name}")
    pth_res.mkdir(parents=True, exist_ok=True)
    svfg.savefig(Path(pth_res).joinpath(f"{subject_name}_{img_name}.png"), dpi=300)
    print(f"{subject_name}: {img_name} colorplot saved")


def plot_glm_hmm_data(subject_name, data_path):

    ## load data
    with open(f"{subject_name}_glm_hmm.pkl", "rb") as f:
        data = pickle.load(f)
        [state_1, state_2, state_3] = data
    
    ## set colorscheme
    states = [state_1, state_2, state_3]
    colors = ["tab:orange", "tab:green", "tab:blue"]

    ## make DataFrame 

    for i in range(len(states)):
        data = pd.DataFrame(states[i])

        ## define x-axis range
        xval = np.arange(0,5,1)

        ## plot data
        svfg = plt.figure(figsize=(10,8))
        set_figure_style()
        ax = sns.boxplot(data = data.T, boxprops=dict(facecolor=colors[i], color=colors[i], alpha=0.5), linewidth=3, showfliers=False)
        sns.stripplot(data = data.T, jitter=True, edgecolor=colors[i], ax=ax, linewidth=3, color=colors[i])

        ## plot labels and title
        plt.xticks(xval, ["k = 0", "k = 1", "k = 2", "k = 3", "k >= 4"], fontsize=28)
        plt.xlabel("# of wheel direction changes", fontsize=28)
        plt.ylabel(f"proportion of data in state {i+1}", fontsize=28)
        plt.title(f"3-state GLM-HMM {subject_name}: state {i+1}", fontsize=28, fontweight="bold")

        ## adjust plot limits
        plt.ylim([-0.05,1.05])

        ## remove spines
        sns.despine(trim=False, offset=8)

        ## tight layout
        plt.tight_layout()

        ## save plot
        pth_res = Path(data_path, f"wiggle/results/glm_hmm/state_{i}/")
        pth_res.mkdir(parents=True, exist_ok=True)
        svfg.savefig(Path(pth_res, f"{subject_name}_state_{i}.png"), dpi=300)

