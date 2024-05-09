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
import ternary

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
    """Save ternary plot of 3-state GLM-HMM probabilities per mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files

    """

    ## obtain 3-D matrix of 3-state GLM-HMM probabilities
    eids = np.load(Path(data_path).joinpath(f"{subject_name}/{subject_name}_eids_glm_hmm.npy"), allow_pickle=True)
    points_mouse = np.load(Path(data_path).joinpath(f"{subject_name}/{subject_name}_points_mouse.npy"), allow_pickle=True)
    ## compute L2 norm to color-code data 
    dist = [np.sqrt(points_mouse[i][0]**2 + points_mouse[i][1]**2 + points_mouse[i][2]**2) for i in range(len(points_mouse))]

    ## scatter plot
    scale = 1
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(12,10)
    set_figure_style()
    ## plot points
    tax.scatter(points_mouse, marker=".", vmin=0.0, vmax=1.0, colormap=plt.cm.viridis, colorbar=True, c=dist, cmap=plt.cm.viridis)

    ## set axis labels and Title
    fontsize = 24
    offset = 0.14
    tax.left_axis_label("P(state 3)", fontsize=fontsize, offset=offset)
    tax.right_axis_label("P(state 2)", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("P(state 1)", fontsize=fontsize, offset=offset)

    ## prettify ternary plot
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.ticks(axis='lbr', linewidth=0.25, multiple=0.1, tick_formats="%.1f", fontsize=18)
    tax.boundary(linewidth=3)
    tax.gridlines(color="black", multiple=0.1, linewidth=0.50)
    tax.set_title(f"{subject_name} (N = {len(eids)} sessions; {len(points_mouse):,} trials)", fontsize=28, fontweight="bold")
    tax.gridlines(multiple=0.2, color="black")

    ## save ternary plots
    pth_res = Path(data_path).joinpath(f"wiggle/results/ternary_plot/")
    pth_res.mkdir(parents=True, exist_ok=True)
    tax.savefig(Path(pth_res).joinpath(f"{subject_name}_ternary_plot.png"), dpi=300)





