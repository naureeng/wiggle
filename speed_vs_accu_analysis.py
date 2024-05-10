## master script for speed vs accu correlation boxplot
from plot_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle
from numpy import nan
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator

## import subdirectories
pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
sys.path.append(Path(pth_dir, "wiggle/plotting/"))
from preprocess.build_analysis_per_mouse import *
from preprocess.integrate_GLM_HMM import *
from plotting.plot_analysis_per_mouse import *

def compute_K_data(subject_name, data_path):
    """Compute speed vs accuracy correlation coefficient for fixed K in each mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files

    Returns:
        data (list): speed vs accuracy correlation coefficient in K = [2,3,4] for one mouse

    """
    data = []
    for K in [2,3,4]:
        low_contrast_speed_K, low_contrast_accu_K = build_fix_K_speed_accu(subject_name, data_path, K)
        if len(low_contrast_speed_K) >=10:
            try:
                m = pearsonr(low_contrast_speed_K, low_contrast_accu_K).statistic
            except:
                m = np.nan
            data.append(m)
        else:
            print(f"{subject_name} has < 10 data points to compute r")
            data.append(np.nan)

    return data

def obtain_bwm_K_data(data_path):
    """Obtain speed vs accuracy correlation coefficients across all mice

    Args:
        data_path (str): path to store data files
    
    Returns:
        bwm_data (list of lists): speed vs accuracy correlation coefficient in K = [2,3,4] across multiple mice

    """

    ## load mouse names
    df = pd.read_csv(Path(data_path).joinpath("final_eids/mouse_names.csv"))
    mouse_names = df["mouse_names"].tolist()

    ## obtain speed vs accu correlation data per mouse
    bwm_data = [compute_K_data(mouse_names[i], data_path) for i in range(len(mouse_names))]

    ## store and save data
    final_data = pd.DataFrame(bwm_data)
    final_data.to_csv(Path(data_path).joinpath("bwm_K_data.csv"), index=False)
    print(f"bwm speed vs accu K analysis csv saved for N = {len(mouse_names)} mice")

    return bwm_data

def boxplot_bwm_K_data(data_path):
    """Boxplot of speed vs accuracy correlation coefficients across all mice for fixed K

    Args:
        data_path (str): path to store data files

    """

    ## load data
    final_data = pd.read_csv(Path(pth_dir).joinpath("bwm_K_data.csv"))

    ## obtain summary stats for legend
    cts = final_data.count()
    m1 = final_data.mean()
    st1 = final_data.std()
    
    ## boxplot data
    svfg = plt.figure(figsize=(12,12))
    set_figure_style()
    ax = sns.boxplot(data=final_data, linewidth=3, palette=["tab:green", "tab:red", "tab:purple"])
    sns.stripplot(data=final_data, jitter=True, edgecolor="gray", linewidth=3, palette=["tab:green", "tab:red", "tab:purple"])
    plt.ylabel("visual speed vs accuracy correlation (r)", fontsize=28)
    plt.xticks([0,1,2], ["2", "3", "4"], fontsize=28)
    plt.xlabel("# of wheel direction changes", fontsize=28)

    ## stats annotations
    pairs=[("0", "1"), ("1", "2"), ("0", "2")]
    annotator = Annotator(ax, pairs, data=final_data)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', fontsize=28, line_width=4, comparisons_correction="Bonferroni")
    annotator.apply_and_annotate()
    ax = plt.gca()
    sns.despine(trim=False, offset=8)
    plt.ylim([-1.05,1.05])

    ## build legend
    legend_elements = [Patch(facecolor='tab:green', edgecolor='k', label='Color Patch', linewidth=2),
            Patch(facecolor='tab:red', edgecolor='k', label='Color Patch', linewidth=2),
            Patch(facecolor='tab:purple', edgecolor='k', label='Color Patch', linewidth=2)]

    ax.legend(handles=legend_elements, labels=[f"N = {cts[0]} mice, {round(m1[0],2)} ± {round(st1[0],2)}", f"N = {cts[1]} mice, {round(m1[1],2)} ± {round(st1[1],2)}", f"N = {cts[2]} mice, {round(m1[2],2)} ± {round(st1[2],2)}"], frameon=False, fontsize=20, loc='upper left', bbox_to_anchor=(0.95, 1))

    plt.tight_layout()
    plt.savefig(Path(data_path).joinpath("bwm_speed_accu_K.png"), bbox_inches="tight")
    print(f"N = {len(final_data)} mice plot for speed vs accu saved")


if __name__=="__main__":
    obtain_bwm_K_data(pth_dir)
    boxplot_bwm_K_data(pth_dir)
    
