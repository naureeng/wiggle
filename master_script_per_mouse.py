## master script for individual mouse analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle
from analysis_utils import load_data, save_average_data

## import subdirectories
pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
sys.path.append(Path(pth_dir, "wiggle/plotting/"))
from preprocess.build_analysis_per_mouse import *
from preprocess.integrate_GLM_HMM import *
from plotting.plot_analysis_per_mouse import *


def per_mouse_analysis(subject_name):
    """Preprocess one mouse

    Args:
        subject_name (str): mouse name

    """
    
    ## preprocess one mouse

    ## plot speed vs accuracy for given K
    for K in np.arange(0,5,1): ## K = 0, 1, 2, 3, >=4
        try:
            low_contrast_speed_K, low_contrast_accu_K = build_fix_K_speed_accu(subject_name, pth_dir, K)
            plot_xval_vs_yval(subject_name, pth_dir, low_contrast_speed_K, low_contrast_accu_K, "mean speed [deg/sec]", "proportion correct", f"K = {K}: speed vs accuracy", f"{K}_speed_vs_accu")
        except Exception as e:
            print(f"K = {K} does not have enough data")

    ## compute prop wiggle vs accuracy
    low_contrast_accu, low_contrast_wiggle = build_prop_wiggle_vs_accuracy(subject_name, pth_dir)
    try:
        plot_xval_vs_yval(subject_name, pth_dir, low_contrast_wiggle, low_contrast_accu, "mean # of wheel direction changes", "proportion correct", "prop wiggle vs accuracy", "prop_wiggle_vs_accu")
    except Exception as e:
        print(f"not enough sessions to do Pearson correlation")

    ## colorplot of prop wiggle
    data_n_extrema_mouse, accu_mouse = load_data(subject_name, pth_dir)
    save_average_data(subject_name, data_n_extrema_mouse, accu_mouse, pth_dir)
    plot_color_plot(subject_name, data_n_extrema_mouse, "coolwarm", pth_dir, (3,8), [0,5], "imagesc")

    ## colorplot of avg prop wiggle across sessions
    avg_mouse_data = pd.DataFrame(data_n_extrema_mouse).mean()
    plot_color_plot(subject_name, pd.DataFrame(avg_mouse_data).T, "coolwarm", pth_dir, (3,4), [1,3], "avg_imagesc")

    ## colorplot of avg prop wiggle >85% sessions only
    avg_mouse_data_85 = np.load(Path(pth_dir).joinpath(f"{subject_name}/{subject_name}_avg_prop_wiggle_85.npy"))
    plot_color_plot(subject_name, pd.DataFrame(avg_mouse_data_85).T, "coolwarm", pth_dir, (3,4), [1,3], "avg_imagesc_85")

    ## GLM-HMM state vs wiggles
    #build_mouse_GLM_HMM_csv(subject_name, pth_dir)
    #build_mouse_wheel_csv(subject_name, pth_dir)

    #state_1 = [build_wiggle_GLM_HMM_analysis(subject_name, pth_dir, K)[0] for K in np.arange(0,5,1)]
    #state_2 = [build_wiggle_GLM_HMM_analysis(subject_name, pth_dir, K)[1] for K in np.arange(0,5,1)]
    #state_3 = [build_wiggle_GLM_HMM_analysis(subject_name, pth_dir, K)[2] for K in np.arange(0,5,1)]

    # pickle file
    #with open(f"{subject_name}_glm_hmm.pkl", "wb") as f:
        #pickle.dump([state_1, state_2, state_3], f)
        #print(f"{subject_name} file pickled")

    #colors = ["tab:orange", "tab:green", "tab:blue"]
    
    ## plot data
    #plot_glm_hmm_data(subject_name, pth_dir)

if __name__=="__main__":
    subject_name = "CSHL_003"
    per_mouse_analysis(subject_name)
