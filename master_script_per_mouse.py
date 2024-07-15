## master script for individual mouse analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle
from analysis_utils import load_data, save_average_data, compute_glm_hmm_engagement, convert_wheel_deg_to_visual_deg, classify_mouse_wiggler 
from curate_eids import curate_eids_mouse
from wheel_utils import plot_ballistic_movement

## import subdirectories
pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
sys.path.append(Path(pth_dir, "wiggle/plotting/"))
from preprocess.build_analysis_per_mouse import *
from preprocess.integrate_GLM_HMM import *
from plotting.plot_analysis_per_mouse import *

def per_mouse_analysis(subject_name):
    """Preprocess one mouse

    :param subject_name (str): mouse name

    """

    ## preprocess one mouse

    ## plot wiggle speed vs accuracy for given K
    for K in [2,3,4]: ## K = 2, 3, >=4
        try:
            low_contrast_speed_K, low_contrast_accu_K = build_fix_K_speed_accu(subject_name, pth_dir, K)
            plot_xval_vs_yval(subject_name, pth_dir, low_contrast_speed_K, low_contrast_accu_K, "mean visual speed [deg/sec]", "proportion correct", f"K = {K}: speed vs accuracy", f"{K}_speed_vs_accu")
        except Exception as e:
            print(f"K = {K} does not have enough data in {subject_name}")

    ## compute prop wiggle vs accuracy
    low_contrast_accu, low_contrast_wiggle = build_prop_wiggle_vs_accuracy(subject_name, pth_dir)
    try:
        plot_xval_vs_yval(subject_name, pth_dir, low_contrast_wiggle, low_contrast_accu, "mean # of wheel direction changes", "proportion correct", "prop wiggle vs accuracy", "prop_wiggle_vs_accu")
    except Exception as e:
        print(f"not enough sessions to do Pearson correlation in {subject_name}")

    ## colorplot of prop wiggle
    try:
        data_n_extrema_mouse, accu_mouse = load_data(subject_name, pth_dir)
        save_average_data(subject_name, data_n_extrema_mouse, accu_mouse, pth_dir)
        plot_color_plot(subject_name, data_n_extrema_mouse, "coolwarm", pth_dir, (3,8), [0,5], "imagesc")

        ## colorplot of avg prop wiggle across sessions
        avg_mouse_data = pd.DataFrame(data_n_extrema_mouse).mean()
        plot_color_plot(subject_name, pd.DataFrame(avg_mouse_data).T, "coolwarm", pth_dir, (3,4), [1,3], "avg_imagesc")

    except Exception as e:
        print(f"not enough data to do scaled color plot in {subject_name}")

    ## convert wheel degrees to visual degrees
    try:
        convert_wheel_deg_to_visual_deg(subject_name, pth_dir)
        build_mouse_wheel_csv(subject_name, "wheelData", "wheel", pth_dir)
    except Exception as e:
        print(f"issues in wheel data of {subject_name}")

    ## GLM-HMM state vs wiggles
    try:
        build_mouse_GLM_HMM_csv(subject_name, pth_dir)
        build_mouse_wheel_csv(subject_name, "glm_hmm", "glm_hmm", pth_dir)
        build_wiggle_GLM_HMM_analysis(subject_name, pth_dir)

        ## plot data
        plot_glm_hmm_data(subject_name, pth_dir)
        compute_glm_hmm_engagement(subject_name, pth_dir)

    except Exception as e:
        print(f"no GLM-HMM data for {subject_name}")

if __name__=="__main__":
    subject_name = "CSHL_003"
    curate_eids_mouse(subject_name, pth_dir)
    plot_ballistic_movement(subject_name, pth_dir)

    #classify_mouse_wiggler(subject_name, pth_dir)
    #per_mouse_analysis(subject_name)

