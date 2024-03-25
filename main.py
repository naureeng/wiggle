## master script 
## author: naureen ghani

## import dependencies
from one.api import ONE
from curate_eids import *
from analysis_utils import *
from plot_utils import * 
import pickle

## per mouse analysis

def preprocess_per_mouse(subject_name):
    """
    PREPROCESS_PER_MOUSE computes csv with  wheel Data "{eid}_wheelData.csv" for N = 1 mouse
    :param subject_name: mouse name [string]
    :return csv per eid for N = 1 mouse
    """

    ## curate eids with csv files per mouse
    curate_eids_mouse(subject_name)

def build_k_groups_per_mouse(subject_name, yname, y_str):
    """
    BUILD_K_GROUPS_PER_MOUSE outputs analysis and plot of k groups for N = 1 mouse 

    :param subject_name: mouse name [string]
    :param yname: variable of interest [string]
    :param ystring: y-axis label for yname [string]
    :return /{subject_name}_{yname}.png [image]
    :return /{subject_name}.k_groups_{yname] [pickle]

    """

    eids = np.load(f"/nfs/gatsbystor/naureeng/{subject_name}/{subject_name}_eids_wheel.npy")
    print(f"{subject_name}: {len(eids)} wheel sessions")

    ## [2] analysis across eids per mouse

    ## accuracy across contrast groups
    k0_hi, k1_hi, k2_hi, k3_hi, k4_hi, n_trials_hi = compute_wiggle_var_by_grp(subject_name, eids, 1.0, f"{yname}") ## high
    k0_lo, k1_lo, k2_lo, k3_lo, k4_lo, n_trials_lo = compute_wiggle_var_by_grp(subject_name, eids, 0.0625, f"{yname}") ## low
    k0_ze, k1_ze, k2_ze, k3_ze, k4_ze, n_trials_ze = compute_wiggle_var_by_grp(subject_name, eids, 0.0, f"{yname}") ## zero

    ## [3] plotting across eids per mouse
    suffix = f"K_four_groups_{yname}"
    x0_str = "k = 0" 
    x1_str = "k = 1"
    x2_str = "k = 2"
    x3_str = "k = 3"
    x4_str = "k >= 4"

    ## high contrast
    plot_four_boxplot_grps(n_trials_hi, k0_hi, k1_hi, k2_hi, k3_hi, k4_hi, f"{subject_name} high contrast", f"high_contrast_{suffix}", subject_name, x0_str, x1_str, x2_str, x3_str, x4_str, y_str)
    ## low contrast
    plot_four_boxplot_grps(n_trials_lo, k0_lo, k1_lo, k2_lo, k3_lo, k4_lo, f"{subject_name} low contrast", f"low_contrast_{suffix}", subject_name, x0_str, x1_str, x2_str, x3_str, x4_str, y_str)
    ## zero contrast
    plot_four_boxplot_grps(n_trials_ze, k0_ze, k1_ze, k2_ze, k3_ze, k4_ze, f"{subject_name} zero contrast", f"zero_contrast_{suffix}", subject_name, x0_str, x1_str, x2_str, x3_str, x4_str, y_str)

    ## [4] save data per mouse
    f = open(f"/nfs/gatsbystor/naureeng/{subject_name}.k_groups_{yname}", 'wb')
    data = [np.median(k0_hi), np.median(k1_hi), np.median(k2_hi), np.median(k3_hi), np.median(k4_hi), np.median(n_trials_hi), np.median(k0_lo), np.median(k1_lo), np.median(k2_lo), np.median(k3_lo), np.median(k4_lo), np.median(n_trials_lo), np.median(k0_ze), np.median(k1_ze), np.median(k2_ze), np.median(k3_ze), np.median(k4_ze), np.median(n_trials_ze)]
    print(data)
    pickle.dump(data, f)
    f.close()
    print("data pickled")

def compute_across_mice(mouse_names, yname, ystring):
    """
    COMPUTE_ACROSS_MICE outputs analysis and plot of k groups for multiple mice

    :param mouse_names: mouse names [list]
    :param yname: variable of interest [string]
    :param ystring: y-axis label for yname [string]
    :return /{}_{yname}.png [image]

    """

    ## initialize variables for across mice data
    bwm_n_trials_hi = []; bwm_k0_hi = []; bwm_k1_hi = []; bwm_k2_hi = []; bwm_k3_hi = []; bwm_k4_hi = [] ## high contrast
    bwm_n_trials_lo = []; bwm_k0_lo = []; bwm_k1_lo = []; bwm_k2_lo = []; bwm_k3_lo = []; bwm_k4_lo = [] ## low contrast
    bwm_n_trials_ze = []; bwm_k0_ze = []; bwm_k1_ze = []; bwm_k2_ze = []; bwm_k3_ze = []; bwm_k4_ze = [] ## zero contrast
    
    ## process each individual mouse
    for i in range(len(mouse_names)):
        subject_name = mouse_names[i]
        #preprocess_per_mouse(subject_name)
        build_k_groups_per_mouse(subject_name, yname, ystring)
        print(f"{subject_name}: done")

        ## store variables for across mice data 
        f = open(f"/nfs/gatsbystor/naureeng/{subject_name}.k_groups_{yname}", 'rb')
        data = pickle.load(f)
        [k0_hi, k1_hi, k2_hi, k3_hi, k4_hi, n_trials_hi, k0_lo, k1_lo, k2_lo, k3_lo, k4_lo, n_trials_lo, k0_ze, k1_ze, k2_ze, k3_ze, k4_ze, n_trials_ze] = data
        bwm_k0_hi.append(k0_hi); bwm_k1_hi.append(k1_hi); bwm_k2_hi.append(k2_hi); bwm_k3_hi.append(k3_hi); bwm_k4_hi.append(k4_hi); bwm_n_trials_hi.append(n_trials_hi)
        bwm_k0_lo.append(k0_lo); bwm_k1_lo.append(k1_lo); bwm_k2_lo.append(k2_lo); bwm_k3_lo.append(k3_lo); bwm_k4_lo.append(k4_lo); bwm_n_trials_lo.append(n_trials_lo)
        bwm_k0_ze.append(k0_ze); bwm_k1_ze.append(k1_ze); bwm_k2_ze.append(k2_ze); bwm_k3_ze.append(k3_ze); bwm_k4_ze.append(k4_ze); bwm_n_trials_ze.append(n_trials_ze)

    ## plot all mice
    plot_four_boxplot_grps(bwm_n_trials_hi, bwm_k0_hi, bwm_k1_hi, bwm_k2_hi, bwm_k3_hi, bwm_k4_hi, f"high contrast (N = {len(bwm_k1_hi)} mice)", f"high_contrast_K_four_groups_{yname}", "", "k = 0", "k = 1", "k = 2", "k = 3", "k >= 4", ystring)
    plot_four_boxplot_grps(bwm_n_trials_lo, bwm_k0_lo, bwm_k1_lo, bwm_k2_lo, bwm_k3_lo, bwm_k4_lo, f"low contrast (N = {len(bwm_k1_lo)} mice)", f"low_contrast_K_four_groups_{yname}", "", "k = 0", "k = 1", "k = 2", "k = 3", "k >= 4", ystring)
    plot_four_boxplot_grps(bwm_n_trials_ze, bwm_k0_ze, bwm_k1_ze, bwm_k2_ze, bwm_k3_ze, bwm_k4_ze, f"zero contrast (N = {len(bwm_k1_ze)} mice)", f"zero_contrast_K_four_groups_{yname}", "", "k = 0", "k = 1", "k = 2", "k = 3", "k >= 4", ystring)


if __name__=="__main__":

    ## [1] load data 
    mouse_names = np.load(f"/nfs/gatsbystor/naureeng/mouse_names.npy", allow_pickle=True)
    print(len(mouse_names), "mice")

    ## [2] process all mice
    compute_across_mice(mouse_names, "feedbackType", "proportion correct")
    compute_across_mice(mouse_names, "rms", "RMS [deg/sec]")
    compute_across_mice(mouse_names, "speed", "speed [deg/sec]")
    compute_across_mice(mouse_names, "duration", "duration [sec]")
