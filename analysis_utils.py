## analysis functions

## import dependencies
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr

def classify_mouse_wiggler(subject_name, data_path):
    """Classify mice as good, neutral, or bad wiggler

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files

    Returns:
        mouse_wiggler_group (str): classification of the mouse as "good wiggler", "neutral wiggler", or "bad wiggler"

    """

    k = np.arange(0,5,1)
    ## read pickle file
    pth_data = Path(data_path).joinpath(f"results/{subject_name}.k_groups_feedbackType")
    with open (pth_data, "rb") as f:
        x = pickle.load(f)
        ## low contrast data
        non_wiggle = x[6:8] ## k = [0,1]
        wiggle = x[8:11] ## k = [2,3,4]
        res = pearsonr(k, np.nan_to_num(x[6:11])) 
        s = pd.Series([np.mean(non_wiggle), np.mean(wiggle)])
        if s.is_monotonic_increasing==True and res.statistic >=0.5:
            mouse_wiggler_group = "good"
        else:
            if res.statistic <= -0.5:
                mouse_wiggler_group = "bad"
            else:
                mouse_wiggler_group = "neutral"
    
    print(f"{subject_name}: {mouse_wiggler_group}")

    return mouse_wiggler_group


def convert_wheel_deg_to_visual_deg(subject_name, data_path):
    """Convert speed analysis done in wheel degrees to visual degrees

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files
        
    """

    eids = np.load(os.path.join(data_path, subject_name, f"{subject_name}_eids_wheel.npy"))
    wheel_conversion_factor = 35 / np.rad2deg(0.3) ## 35 visual degrees = 0.3 wheel radians

    for eid in eids:
        try:
            file_path = os.path.join(data_path, subject_name, eid, f"{eid}_wheelData.csv")
            df_eid = pd.read_csv(file_path)
            wheel_speed = df_eid["speed"].values
            K = df_eid["n_extrema"].values
           
            ## compute visual speed
            visual_speed = [wheel_speed[i] * wheel_conversion_factor if K[i] != 0 else 0 for i in range(len(K))]

            ## store data
            df_eid["visual_speed"] = np.nan_to_num(visual_speed, posinf=0, neginf=0) ## remove infs
            df_eid.to_csv(file_path, index=False)

        except FileNotFoundError:
            print(f"File not found for session {eid}. Skipping...")

    print(f"{subject_name}: speed data converted from wheel deg/sec to visual deg/sec for {len(eids)} sessions")


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
    n_trials = [[] for _ in range(5)] ## initialize lists for groups k = 0 to 4
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
            df_k = df_data.query(f"n_extrema == {k} and duration <=1") ## to query trials by #extrema and duration [sec] 
            if len(df_k) >= 1: ## minimum of one trial
                group_mean = df_k[yname].mean()
                group_data[k].append(group_mean)
                n_trials[k].append(len(df_k))

    return tuple(group_data), trial_counts, n_trials


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


def compute_glm_hmm_engagement(subject_name, data_path):
    """Update csv files with manual annotation of 3-state GLM-HMM model
    Args:
        subject_name (str): mouse name
        data_path (str): data path to store files

    """
    glm_hmm_df = pd.read_csv(Path(data_path).joinpath("glm_hmm_analysis.csv")) ## manual annotation csv
    idx = glm_hmm_df.index[glm_hmm_df["mouse_name"] == f"{subject_name}"][0]
    states = dict(glm_hmm_df.iloc[idx])

    ## load glm-hmm eids
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_glm_hmm.npy"))
    print(f"{subject_name}: {len(eids)} glm-hmm eids")

    ## update glm-hmm csv
    for eid in eids:
        df_eid = pd.read_csv(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv"))
        df_eid['state_binary'] = df_eid['state_glm_hmm_3'].map(states)
        ## save updated csv
        df_eid["feedbackType"] = df_eid["feedbackType"].replace(-1,0)
        df_eid.to_csv(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv"))

    ## concatenate all glm-hmm csvs
    csv_learn = []
    [csv_learn.append(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv")) for eid in eids]

    ## concatenate csvs
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_learn], ignore_index=True)
    df_csv_concat.to_csv(Path(data_path, subject_name, f"{subject_name}_glm_hmm_classification.csv"), index=False)
    print(f"csv saved for {subject_name}: {len(eids)} sessions")

def load_mouse_wiggle_data(subject_name, data_path):
    """Load mouse wiggle data 

    Args:
        subject_name (str): mouse name
        data_path (str): data path to store files
    """

    try:
        df_mouse = pd.read_csv(Path(data_path, subject_name, f"{subject_name}_glm_hmm_classification.csv"))
        wiggle = df_mouse.query("abs(goCueRT-stimOnRT) <=0.05 and n_extrema >=2 and duration <=1")
        return wiggle
    except Exception as e:
        print(f"{subject_name} has no sessions")
        return None

def compute_wiggle_statistics(wiggle):
    """Compute mouse wiggle statistics

    Args:
        wiggle (df): dataframe of mouse wiggle data
    """

    stats = {}
    wiggle["feedbackType"] = wiggle["feedbackType"].replace(-1,0)
    stats["mean_K"] = wiggle["n_extrema"].mean()
    stats["median_K"] = wiggle["n_extrema"].median()

    total_trials = len(wiggle)

    for state in ["engaged", "biased", "disengaged"]:
        state_wiggle = wiggle.query(f"state_binary == '{state}'")
        state_wiggle_early = state_wiggle.query("goCueRT <=0.08 and goCueRT >=-0.2")
        state_wiggle_normal = state_wiggle.query("goCueRT >0.08 and goCueRT <1.2")

        ## proportion state
        stats[f"P{state}"] = len(state_wiggle) / total_trials if total_trials > 0 else 0 
        stats[f"P{state}_early"] = len(state_wiggle_early) / total_trials if total_trials > 0 else 0
        stats[f"P{state}_normal"] = len(state_wiggle_normal) / total_trials if total_trials > 0 else 0

        ## proportion correct
        stats[f"P{state}_corr"] = state_wiggle["feedbackType"].mean()
        stats[f"P{state}_early_corr"] = state_wiggle_early["feedbackType"].mean()
        stats[f"P{state}_normal_corr"] = state_wiggle_normal["feedbackType"].mean()

    return stats

def update_csv_glm_hmm_engagement(data_path):
    """Update glm_hmm_classification csv with mouse wiggle statistics

    Args:
        data_path (str): data path to store files

    """

    df = pd.read_csv(Path(data_path, "glm_hmm_analysis.csv"), index_col=0)
    print(df)
    mouse_names = df["mouse_name"].tolist()

    ## list to hold results for each subject
    results = []

    for subject_name in mouse_names:
        wiggle = load_mouse_wiggle_data(subject_name, data_path)
        stats = compute_wiggle_statistics(wiggle)
        results.append(stats)

    ## convert results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df["mouse_name"] = mouse_names

    ## concatenate new results with the original DataFrame
    updated_df = df.merge(results_df, how="right", on="mouse_name")
    print(updated_df)

    ## save updated DataFrame to csv
    updated_df.to_csv(Path(data_path, "glm_hmm_analysis.csv"), index=False)
    print("glm-hmm analysis updated")


