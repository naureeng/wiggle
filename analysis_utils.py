## analysis functions

## import dependencies
from pathlib import Path
import numpy as np
import pandas as pd

def compute_wiggle_var_by_grp(subject_name, eids, contrast_value, yname, data_path):
    """ Perform wiggle analysis per mouse. 

    Compute data per group by k and trial counts for N = 1 mouse.

    Args:
        subject_name (str): mouse name
        eids (list): sessions
        contrast_value (float): contrast value [±1, ±0.25, ±0.125, ±0.0625, 0]
        data_path (str): path to data files

    Returns:
        Data for groups k = 0, 1, 2, 3, 4, and trial counts (tuple of lists)

    """

    group_data = [[] for _ in range(5)] ## initialize lists for groups k = 0 to 4
    trial_counts = [] ## initialize list for trial counts

    for eid in eids:

        try:
            df_eid = pd.read_csv(f"{data_path}/{subject_name}/{eid}/{eid}_wheelData.csv")
        except FileNotFoundError:
            print(f"File not found for session {eid}. Skipping...")
            continue

        df_eid["feedbackType"] = df_eid["feedbackType"].replace(-1,0) ## to compute average
        df_data = df_eid.query(f"abs(contrast)=={contrast_value} and abs(goCueRT-stimOnRT)<=0.05") ## to query trials by contrast
        trial_counts.append(len(df_data)) 

        for k in range(5):
            df_k = df_data.query(f"n_extrema == {k} and duration <= 1") ## to query trials by #extrema and duration [sec] 
            if len(df_k) >= 1: ## minimum of one trial
                group_mean = df_k[yname].mean()
                group_data[k].append(group_mean)

    return tuple(group_data), trial_counts


def load_data(subject_name, data_path):
    """Load data for a given subject
    
    Args:
        subject_name (str): mouse name
        data_path (str): data path to store files
    
    Returns:
        data_n_extrema_mouse (list): sessions x stimulus contrast data on mean # of changes in wheel direction
        accu_mouse (list): accuracies across sessions

    """

    eids = np.load(Path(data_path).joinpath(subject_name, f"{subject_name}_eids_wheel.npy"))
    data_n_extrema_mouse = []; accu_mouse = []
    for eid in eids:
        df_eid = pd.read_csv(Path(data_path).joinpath(subject_name, eid, f"{eid}_wheelData.csv")) 
        threshold = 2 ## definition of wiggle is >=2 changes in wheel direction
        df_eid["wiggle"] = df_eid["n_extrema"].gt(threshold).astype(int) ## filter dataframe by threshold
        contrast_values = df_eid["contrast"].nunique()
        df_eid["feedbackType"] = df_eid["feedbackType"].replace(-1,0) ## to compute accuracy by mean

        if contrast_values == 9: ## full stimulus set presentation
            data_n_extrema = df_eid.groupby("contrast")["n_extrema"].mean().tolist()
            data_n_extrema_mouse.append(data_n_extrema)
            accu_mouse.append(df_eid["feedbackType"].mean().tolist())

    return data_n_extrema_mouse, accu_mouse


def save_average_data(subject_name, data_n_extrema_mouse, accu_mouse, data_path):
    """Save average data for a given subject as npy file

    Args:
        subject_name (str): mouse name
        data_n_extrema_mouse (list): sessions x stimulus contrast data on mean # of changes in wheel direction
        accu_mouse (list): accuracies across sessions
        data_path (str): data path to store files

    """
    avg_data = pd.DataFrame(data_n_extrema_mouse).mean()
    np.save(Path(data_path).joinpath(f"{subject_name}/{subject_name}_avg_prop_wiggle.npy"), avg_data.values.tolist())

    ## obtain >90% performance sessions
    perf_90 = np.where(np.array(accu_mouse) >= 0.90)[0] ## indices of sessions
    data_90 = [data_n_extrema_mouse[i] for i in perf_90]
    avg_mouse_data_90 = pd.DataFrame(data_90).mean()
    np.save(Path(data_path).joinpath(f"{subject_name}/{subject_name}_avg_prop_wiggle_90.npy"), avg_mouse_data_90.values.tolist())
