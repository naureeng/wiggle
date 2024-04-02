## master script for individual mouse analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
sys.path.append(Path(pth_dir, "wiggle/plotting/"))
from preprocess.build_analysis_per_mouse import *
from plotting.plot_analysis_per_mouse import *

if __name__=="__main__":
    subject_name = "PL028"
    low_contrast_accu, low_contrast_wiggle = build_prop_wiggle_vs_accuracy(subject_name, pth_dir)
    #build_mouse_wheel_csv(subject_name, pth_dir)
    #plot_xval_vs_yval(subject_name, pth_dir, low_contrast_wiggle, low_contrast_accu, "mean # of wheel direction changes", "proportion correct", "% wiggle vs accuracy")
    #plot_imagesc(subject_name, pth_dir)

    for K in np.arange(0,4,1):
        low_contrast_speed_K, low_contrast_accu_K = build_fix_K_speed_accu(subject_name, pth_dir, K)
        plot_xval_vs_yval(subject_name, pth_dir, low_contrast_speed_K, low_contrast_accu_K, "mean speed [deg/sec]", "proportion correct", f"K = {K}: speed vs accuracy")
        

