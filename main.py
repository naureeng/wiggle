## master script 
## author: naureen ghani

## import dependencies
from one.api import ONE
from curate_eids import *
from analysis_utils import *
from plot_utils import * 
import pickle
import seaborn as sns
from pathlib import Path

#one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          #password='international', silent=True)

one = ONE()

## per mouse analysis

def preprocess_per_mouse(subject_name, data_path):
    """Pre-process one mouse

    Computes csv with  wheel Data "{eid}_wheelData.csv" for N = 1 mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files 

    """

    ## curate eids with csv files per mouse
    curate_eids_mouse(subject_name, data_path)


def build_k_groups_per_mouse(subject_name, yname, ystr, data_path):
    """Process one mouse

    Outputs analysis and plot of groups by #extrema for N = 1 mouse 

    Args:
        subject_name (str): mouse name
        yname (str): variable of interest
        ystr (str): y-axis label for yname 

    """

    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_wheel.npy"))
    print(f"{subject_name}: {len(eids)} wheel sessions")

    contrasts = [1.0, 0.0625, 0.0]  # high, low, zero contrasts
    contrast_labels = ["high", "low", "zero"] ## contrast labels
    data = [] # initialize data

    ## plot labels
    x_labels = ["0", "1", "2", "3", "4"]
    xstr  = "#changes in wheel direction"

    for i in range(len(contrasts)):
        contrast = contrasts[i]
        results, n_trials, n_trials_K = compute_wiggle_var_by_grp(subject_name, eids, contrast, yname, data_path)
        
        data.extend([np.median(result) for result in results])
        data.append(int(np.median(n_trials))) ## integer necessary for weighing DataFrame

        plot_four_boxplot_grps(n_trials, results, x_labels, subject_name, xstr, ystr, yname, Path(data_path, f"{contrast_labels[i]}_contrast_{yname}"), "Wilcoxon", figure_size=(10,8))

        ## build legend
        m1_mouse  =  [np.nanmean(results[i]) for i in range(len(results))]
        st1_mouse = [np.nanstd(results[i]) for i in range(len(results))]
        cts_mouse = [np.nansum(n_trials_K[i]) for i in range(len(results))]

        build_legend(m1_mouse, st1_mouse, cts_mouse)

        ## save plot
        save_plot(Path(data_path, f"{contrast_labels[i]}_contrast_{yname}"), f"{subject_name}_{yname}.png")
        
        ## save trial counts
        np.save(Path(data_path, f"{subject_name}", f"{subject_name}_n_trials_K_{contrast_labels[i]}.npy"), n_trials_K)

    ## save data
    with open(Path(data_path, "results", f"{subject_name}.k_groups_{yname}"), "wb") as f:
        pickle.dump(data, f)


def compute_across_mice(mouse_names, yname, ystr, data_path):
    """Process multiple mice

    Outputs analysis and plot of groups by #extrema for N > 1 mouse

    Args:
        mouse_names (list): mouse names
        yname (str): variable of interest
        ystring (str): y-axis label for yname

    """

    bwm_data = [[] for _ in range(18)]  # Initialize list for all data points

    for subject_name in mouse_names:
        with open(Path(data_path, "results", f"{subject_name}.k_groups_{yname}"), "rb") as f:
            data = pickle.load(f)
            for i, val in enumerate(data):
                bwm_data[i].append(val)
    
    ## plotting for high, low, and zero contrast across all mice
    contrast_labels = ["high", "low", "zero"] 
    idx = np.arange(0,18,6)

    for i in range(len(contrast_labels)):
        start = idx[i]
        plot_four_boxplot_grps(bwm_data[start:start+6][-1], bwm_data[start:start+6], ["0", "1", "2", "3", "4"], f"N = {len(mouse_names)} mice", "#changes in wheel direction", ystr, yname, Path(data_path, f"{contrast_labels[i]}_contrast_{yname}"), "Wilcoxon", figure_size=(10,8))

        ## compute summary stats
        m1_final  = np.nanmean(bwm_data[start:start+6], axis=1)
        st1_final = np.nanstd(bwm_data[start:start+6], axis=1)

        ## compute trial counts
        _, _, cts_final = compute_n_trials_per_K(pth_dir, contrast_labels[i])
        
        ## build legend
        build_legend(m1_final, st1_final, cts_final)
       
        ## save plot
        save_plot(Path(data_path, f"{contrast_labels[i]}_contrast_{yname}"), f"N = {len(mouse_names)} mice_{yname}.png")


if __name__=="__main__":

    ## analysis per mouse
    pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
    pth_res = Path(pth_dir, 'results')
    pth_res.mkdir(parents=True, exist_ok=True)
    mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) ## load mouse_names

    ## pre-process per mouse
    #[preprocess_per_mouse(subject_name, pth_dir) for subject_name in mouse_names]
    #[convert_wheel_deg_to_visual_deg(subject_name, pth_dir) for subject_name in mouse_names]

    ## build plots
    for yname, ystr in [("feedbackType", "proportion correct")]: #[("feedbackType", "proportion correct"), ("rms", "RMS [deg/sec]"), ("visual_speed", "visual speed [visual deg/sec]")]:
        #[build_k_groups_per_mouse(subject_name, yname, ystr, pth_dir) for subject_name in mouse_names] ## plot per mouse
        compute_across_mice(mouse_names, yname, ystr, pth_dir) ## plot across mice
