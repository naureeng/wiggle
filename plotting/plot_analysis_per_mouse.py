import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from numpy import nan

pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle"))
from plot_utils import *

def plot_xval_vs_yval(subject_name, data_path, xval, yval, xstr, ystr, tstr):
    svfg = plt.figure(figsize=(10,8))
    set_figure_style()
    plt.scatter(xval, yval, 60)
    plt.xlabel(xstr, fontsize=28)
    plt.ylabel(ystr, fontsize=28)
    plt.title(tstr, fontsize=28, fontweight="bold")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    sns.despine(trim=False, offset=8)
    plt.tight_layout()
    plt.show()

def plot_imagesc(subject_name, pth_dir):

    data_n_extrema_mouse = []

    eids = np.load(Path(pth_dir, subject_name, f"{subject_name}_eids_wheel.npy"))
    for eid in eids:
        df_eid = pd.read_csv(Path(pth_dir, f"{subject_name}/{eid}/{eid}_wheelData.csv"))
        threshold = 2
        df_eid['wiggle'] = df_eid['n_extrema'].gt(threshold).astype(int)
        contrast_values = df_eid["contrast"].nunique()
        if contrast_values == 9:
            data_n_extrema = df_eid.groupby("contrast")["n_extrema"].mean().tolist()
            data_n_extrema_mouse.append(data_n_extrema)

    gb = pd.DataFrame(data_n_extrema_mouse)
    plt.figure(figsize=(10,8))
    set_figure_style()
    plt.imshow(gb, cmap="coolwarm")
    plt.colorbar(label="mean # of wheel direction changes")
    plt.clim([0,5])
    plt.tight_layout()
    plt.axis("off")
    plt.show()
