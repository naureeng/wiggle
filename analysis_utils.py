## import dependencies
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
