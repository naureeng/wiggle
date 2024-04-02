import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from numpy import nan
from scipy.stats import pearsonr

pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle"))
from plot_utils import *

def plot_xval_vs_yval(subject_name, data_path, xval, yval, xstr, ystr, tstr):
    """Obtains scatterplot of two variables with line of best fit and Pearson correlation coefficient

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files
        xval (list): independent variable values
        yval (list): dependent variables values
        xstr (str): x-axis label
        ystr (str): y-axis label
        tstr (str): plot title
    
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

    ## tight layout and display plot
    plt.tight_layout()
    plt.show()

def plot_imagesc(subject_name, pth_dir):
    """Displays data as scaled color plot 

    Args:
        subject_name (str): mouse name
        pth_dir (str): path to data files 

    """

    data_n_extrema_mouse = []

    eids = np.load(Path(pth_dir, subject_name, f"{subject_name}_eids_wheel.npy"))
    for eid in eids:
        df_eid = pd.read_csv(Path(pth_dir, subject_name, eid, f"{eid}_wheelData.csv"))
        threshold = 2
        df_eid['wiggle'] = df_eid['n_extrema'].gt(threshold).astype(int)
        contrast_values = df_eid["contrast"].nunique()
        if contrast_values == 9:
            data_n_extrema = df_eid.groupby("contrast")["n_extrema"].mean().tolist()
            data_n_extrema_mouse.append(data_n_extrema)

    gb = pd.DataFrame(data_n_extrema_mouse)
    plt.figure(figsize=(10,8))
    set_figure_style()
    im = plt.imshow(gb, cmap="coolwarm")
    cb = plt.colorbar(im, orientation='vertical').set_label(label='mean # of wheel direction changes', size=22)
    plt.clim([0,5])
    plt.tight_layout()
    plt.axis("off")
    plt.show()
