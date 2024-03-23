## master script 
## author: naureen ghani

## import dependencies
from one.api import ONE
from curate_eids import *
from analysis_utils import *
from plot_utils import * 

## example mouse
subject_name = "NYU-04"

## [1] curate eids with csv files per mouse
#curate_eids_mouse(subject_name)
eids = np.load(f"/nfs/gatsbystor/naureeng/{subject_name}/{subject_name}_eids_wheel.npy")
print(f"{subject_name}: {len(eids)} wheel sessions")

## [2] analysis across eids per mouse

## accuracy across contrast groups
k0_hi, k1_hi, k2_hi, k3_hi, k4_hi, n_trials_hi = compute_wiggle_var_by_grp(subject_name, eids, 1.0, "feedbackType") ## high
k0_lo, k1_lo, k2_lo, k3_lo, k4_lo, n_trials_lo = compute_wiggle_var_by_grp(subject_name, eids, 0.0625, "feedbackType") ## low
k0_ze, k1_ze, k2_ze, k3_ze, k4_ze, n_trials_ze = compute_wiggle_var_by_grp(subject_name, eids, 0.0, "feedbackType") ## zero

## [3] plotting across eids per mouse
suffix = "K_four_groups"
x0_str = "k = 0"
x1_str = "k = 1"
x2_str = "k = 2"
x3_str = "k = 3"
x4_str = "k >= 4"

## high contrast
plot_four_boxplot_grps(n_trials_hi, k0_hi, k1_hi, k2_hi, k3_hi, k4_hi, f"{subject_name} high contrast", f"high_contrast_{suffix}", subject_name, x0_str, x1_str, x2_str, x3_str, x4_str, "proportion correct")
## low contrast
plot_four_boxplot_grps(n_trials_lo, k0_lo, k1_lo, k2_lo, k3_lo, k4_lo, f"{subject_name} low contrast", f"low_contrast_{suffix}", subject_name, x0_str, x1_str, x2_str, x3_str, x4_str, "proportion correct")
## zero contrast
plot_four_boxplot_grps(n_trials_ze, k0_ze, k1_ze, k2_ze, k3_ze, k4_ze, f"{subject_name} zero contrast", f"zero_contrast_{suffix}", subject_name, x0_str, x1_str, x2_str, x3_str, x4_str, "proportion correct")

