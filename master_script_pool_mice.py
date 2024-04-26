## import dependencies
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from analysis_utils import compute_pearsonr_wheel_accu
import pickle
from plot_utils import plot_boxplot, save_plot
import sys

## obtain mouse names
pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
pth_res = Path(pth_dir, 'results')
pth_res.mkdir(parents=True, exist_ok=True)
mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) ## load mouse_names

sys.path.append(Path(pth_dir, "wiggle/plotting/"))
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
    
    Args:
        mouse_names (list): list of strings of mouse names
        data_path (str): data path to store files

    """
    mice_pearsonr_wheel_accu = np.load(Path(data_path).joinpath(f"mice_pearsonr_wheel_accu.npy"))
    pos_idx = np.where(mice_pearsonr_wheel_accu > 0)[0]
    neg_idx = np.where(mice_pearsonr_wheel_accu <= 0)[0]

    pos_mouse_names = [mouse_names[i] for i in pos_idx]
    neg_mouse_names = [mouse_names[i] for i in neg_idx]

    print(len(pos_mouse_names), "good wigglers")
    print(len(neg_mouse_names), "bad wigglers")

    ## save data
    np.save(Path(data_path).joinpath(f"good_wigglers.npy"), pos_mouse_names)
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

def save_group_color_plot(mouse_names, data_path, file_name, group_name):
    """Plot scaled color plot in group of mice

    Args:

    """
    data_extrema = []
    for i in range(len(mouse_names)):
        subject_name = mouse_names[i]
        avg_mouse_data = np.load(Path(pth_dir).joinpath(f"{subject_name}/{subject_name}_{file_name}.npy"))
        if len(avg_mouse_data)==9:
            data_extrema.append(avg_mouse_data)
    
    group_data = np.mean(data_extrema[1:], axis=0) ## remove data not appended with remainder of matrices
    avg_group = pd.DataFrame(group_data)
    plot_color_plot("", avg_group.T, "coolwarm", data_path, (3,5), [1,3], group_name)

## main script
if __name__=="__main__":
    #compute_pearsonr_across_mice(mouse_names, pth_dir)
    #sort_mice(mouse_names, pth_dir)

    for group in ["good", "bad"]:
        wigglers = np.load(Path(pth_dir, f"{group}_wigglers.npy"))
        plot_K_feedbackType_group(wigglers, pth_dir, f"{group}_wigglers")
        save_group_color_plot(wigglers, pth_dir, "avg_prop_wiggle", f"{group}_wigglers")
        save_group_color_plot(wigglers, pth_dir, "avg_prop_wiggle_90", f"{group}_90_wigglers")

