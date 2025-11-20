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

    :param subject_name (str): mouse name
    :param data_path (str): path to data files 

    """

    ## curate eids with csv files per mouse
    curate_eids_mouse(subject_name, data_path)


def build_k_groups_per_mouse(subject_name, yname, ystr, data_path, time_window, suffix):
    """Process one mouse

    Outputs analysis and plot of groups by #extrema for N = 1 mouse 

    :param subject_name (str): mouse name
    :param yname (str): variable of interest
    :param ystr (str): y-axis label for yname 
    :param time_window (str): time window of analysis
    :param suffix (str): suffix of analysis

    """

    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_data.npy"))
    print(f"{subject_name}: {len(eids)} wheel sessions")

    contrasts = [1.0, 0.25, 0.125, 0.0625, 0.0]  # high, low, zero contrasts
    contrast_labels = ["1", "025", "0125", "00625", "0"] ## contrast labels
    data = [] # initialize data

    ## plot labels
    x_labels = ["0", "1", "2", "3", "4"]
    xstr  = "#changes in wheel direction"

    for i in range(len(contrasts)):
        contrast = contrasts[i]
        print(contrast)
        results, n_trials, n_trials_K = compute_wiggle_var_by_grp(subject_name, eids, contrast, yname, data_path, time_window, suffix)
        
        data.extend([np.median(result) for result in results])
        data.append(int(np.median(n_trials))) ## integer necessary for weighing DataFrame

        m1_mouse, st1_mouse = plot_four_boxplot_grps(n_trials, results, x_labels, subject_name, xstr, ystr, yname, Path(data_path, f"{contrast_labels[i]}_contrast_{yname}"), "Wilcoxon", figure_size=(10,8))

        ## build legend
        #m1_mouse  =  [np.nanmean(results[i]) for i in range(len(results))]
        #st1_mouse = [np.nanstd(results[i]) for i in range(len(results))]
        cts_mouse = [np.nansum(n_trials_K[i]) for i in range(len(results))]

        #build_legend(m1_mouse, st1_mouse, cts_mouse)

        ## save plot
        save_plot(Path(data_path, f"{contrast_labels[i]}_contrast_{yname}_{time_window}_{suffix}"), f"{subject_name}_{yname}.svg")
        print(Path(data_path, f"{contrast_labels[i]}_contrast_{yname}_{time_window}_{suffix}"))
        print(f"{subject_name}: boxplot for {contrast_labels[i]} {yname} saved")
        
        ## save trial counts
        np.save(Path(data_path, f"{subject_name}", f"{subject_name}_n_trials_K_{contrast_labels[i]}_{time_window}_{suffix}.npy"), n_trials_K)

    ## save median data
    with open(Path(data_path, "results", f"{subject_name}.k_groups_{yname}_{time_window}_{suffix}"), "wb") as f:
        pickle.dump(data, f)


def compute_across_mice(mouse_names, yname, ystr, data_path, time_window, suffix):
    """Process multiple mice

    Outputs analysis and plot of groups by #extrema for N > 1 mouse

    :param mouse_names (list): mouse names
    :param yname (str): variable of interest
    :param ystring (str): y-axis label for yname
    :param time_window (str): time window of analysis
    :param suffix (str): suffix of analysis 

    """
    bwm_data = [[] for _ in range(30)]  # Initialize list for all data points

    for subject_name in mouse_names:
        try:
            with open(Path(data_path, "results", f"{subject_name}.k_groups_{yname}_{time_window}_{suffix}"), "rb") as f:
                data = pickle.load(f)
                for i, val in enumerate(data):
                    bwm_data[i].append(val)
        except:
            pass
            print(f"no data {subject_name}")
    
    ## plotting for high, low, and zero contrast across all mice
    #contrast_labels = ["100", "25", "12", "6", "0"] 
    contrasts = [1.0, 0.25, 0.125, 0.0625, 0.0]  # high, low, zero contrasts
    contrast_labels = ["1", "025", "0125", "00625", "0"] ## contrast labels
    idx = np.arange(0,30,6)

    for i in range(len(contrast_labels)):
        start = idx[i]

        tstr = f"N = {len(mouse_names)} mice" ## title string

        plot_four_boxplot_grps(bwm_data[start:start+6][-1], bwm_data[start:start+6], ["0", "1", "2", "3", "4"], tstr, "#changes in wheel direction", ystr, yname, Path(data_path, f"{contrast_labels[i]}_contrast_{yname}_{time_window}_{suffix}"), "Wilcoxon", figure_size=(10,8))

        ## compute summary stats
        m1_final  = np.nanmean(bwm_data[start:start+6], axis=1)
        st1_final = np.nanstd(bwm_data[start:start+6], axis=1)
    
        ## compute trial counts
        cts_final = compute_cts_final(mouse_names, contrasts[i], time_window, suffix)
        #_, _, cts_final = compute_n_trials_per_K(mouse_names, pth_dir, contrast_labels[i], time_window, suffix)
        
        ## build legend
        build_legend(m1_final, st1_final, cts_final)
        
        ## save plot
        
        print(Path(data_path, f"{contrast_labels[i]}_contrast_{yname}_{time_window}_{suffix}"))
        save_plot(Path(data_path, f"{contrast_labels[i]}_contrast_{yname}_{time_window}_{suffix}"), f"N = {len(mouse_names)} mice_{yname}.svg")


if __name__=="__main__":

    ## analysis per mouse
    pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
    pth_res = Path(pth_dir, 'results')
    pth_res.mkdir(parents=True, exist_ok=True)

    mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) ## load mouse_names
    print(len(mouse_names), "mice")
    time_window = "total"
    suffix = ""

    ## pre-process per mouse
    [preprocess_per_mouse(subject_name, pth_dir) for subject_name in mouse_names]
    [convert_wheel_deg_to_visual_deg(subject_name, pth_dir, time_window) for subject_name in mouse_names]

    ## build plots
     for yname, ystr in [("feedbackType", "proportion correct"), ("rms", "RMS [deg/sec]"), ("visual_speed", "visual speed [visual deg/sec]")]:
        for i in range(len(mouse_names)):
            try:
                subject_name = mouse_names[i]
                build_k_groups_per_mouse(subject_name, yname, ystr, pth_dir, time_window, suffix)
            except:
                pass
                print(f"issue in {subject_name}")
        
        compute_across_mice(mouse_names, yname, ystr, pth_dir, time_window, suffix) ## plot across mice


