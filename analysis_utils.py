## analysis functions

## import dependencies
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr

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
            visual_speed = [(wheel_speed[i] / (K[i]+0.000001 / 2)) * wheel_conversion_factor for i in range(len(K))]

            ## store data
            df_eid["visual_speed"] = visual_speed
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

    ## obtain >85% performance sessions
    perf_85 = np.where(np.array(accu_mouse) >= 0.85)[0] ## indices of sessions
    data_85 = [data_n_extrema_mouse[i] for i in perf_85]
    avg_mouse_data_85 = pd.DataFrame(data_85).mean()
    np.save(Path(data_path).joinpath(f"{subject_name}/{subject_name}_avg_prop_wiggle_85.npy"), avg_mouse_data_85.values.tolist())


def compute_pearsonr_wheel_accu(subject_name, data_path):
    """Compute Pearson r correlation coefficient between # of wheel direction changes vs proportion correct

    Args:
        subject_name (str): mouse name

    """
    k = np.arange(0,5,1)
    ## read pickle file
    pth_data = Path(data_path).joinpath(f"results/{subject_name}.k_groups_feedbackType")
    with open (pth_data, "rb") as f:
        x = pickle.load(f)
        data = x[6:11] ## indices of low contrast data
        r = pearsonr(k, data).statistic

    return r

def compute_glm_hmm_engagement(subject_name, data_path):
    """Compute P(engagement) based on manual annotation of 3-state GLM-HMM model
    Args:
        subject_name (str): mouse name
        data_path (str): data path to store files

    """
    glm_hmm_df = pd.read_csv(Path(data_path).joinpath("glm_hmm_analysis.csv")) ## manual annotation csv
    idx = glm_hmm_df.index[glm_hmm_df["mouse_name"] == f"{subject_name}"][0]
    states = dict(glm_hmm_df.iloc[idx])

    ## eids
    ## load glm-hmm eids
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_glm_hmm.npy"))
    print(f"{subject_name}: {len(eids)} glm-hmm eids")
    eids_80 = []

    ## compute P(engagement)
    for eid in eids:
        df_eid = pd.read_csv(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv"))
        df_eid['state_binary'] = df_eid['state_glm_hmm_3'].map(states)
        ## save updated csv
        df_eid.to_csv(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv"))
        ## identify sessions with >=85% accuracy
        df_eid["feedbackType"] = df_eid["feedbackType"].replace(-1,0)
        session_accu = df_eid["feedbackType"].mean()
        if session_accu >= 0.80:
            eids_80.append(eid)

    ## concatenate all glm-hmm csvs
    csv_learn = []
    [csv_learn.append(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv")) for eid in eids_80]

    ## concatenate csvs
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_learn], ignore_index=True)
    df_csv_concat.to_csv(Path(data_path, subject_name, f"{subject_name}_glm_hmm_classification_80.csv"), index=False)
    print(f"csv saved for {subject_name}: {len(eids)} sessions")

def update_csv_glm_hmm_engagement(data_path):
    """Update glm_hmm_classification csv with num_engaged and num_disengaged for wiggles

    Args:
        data_path (str): data path to store files

    """
    df = pd.read_csv(Path(data_path, "glm_hmm_analysis.csv"))
    
    ## initialize
    num_engaged = []; num_disengaged = []; num_biased = []
    mean_K = []; median_K = []; 
    Pengaged = []; Pbiased = []; Pdisengaged = [] 

    ## compute #engaged, #disengaged, # biased wiggles
    for subject_name in df["mouse_name"]:
        try:
            df_mouse = pd.read_csv(Path(data_path, subject_name, f"{subject_name}_glm_hmm_classification_80.csv"))
            wiggle = df_mouse.query("n_extrema >=2 and duration <=1")

            ## compute statistics
            mean_K_mouse = wiggle["n_extrema"].mean()
            median_K_mouse = wiggle["n_extrema"].median()

            ## split by state
            engage_wiggle = wiggle.query("state_binary == 'engaged'")
            disengage_wiggle = wiggle.query("state_binary == 'disengaged'")
            bias_wiggle = wiggle.query("state_binary == 'biased'")

            ## compute Pengaged
            total_trials = len(engage_wiggle) + len(disengage_wiggle) + len(bias_wiggle)
            Pengaged_mouse = len(engage_wiggle) / total_trials
            Pbiased_mouse = len(bias_wiggle) / total_trials
            Pdisengaged_mouse = len(disengage_wiggle) / total_trials

            ## save results
            num_engaged.append(len(engage_wiggle)); num_disengaged.append(len(disengage_wiggle)); num_biased.append(len(bias_wiggle))
            mean_K.append(mean_K_mouse); median_K.append(median_K_mouse); Pengaged.append(Pengaged_mouse); Pbiased.append(Pbiased_mouse); Pdisengaged.append(Pdisengaged_mouse)

        except Exception as e:
            print(f"{subject_name} has no sesions")
            num_engaged.append([]); num_disengaged.append([]); num_biased.append([])
            mean_K.append([]); median_K.append([]); Pengaged.append([]); Pbiased.append([]); Pdisengaged.append([])

    df["num_engaged_wiggles"] = num_engaged; df["num_disengaged_wiggles"] = num_disengaged; df["num_biased_wiggles"] = num_biased
    df["mean_K"] = mean_K; df["median_K"] = median_K; df["Pengaged"] = Pengaged; df["Pbiased"] = Pbiased; df["Pdisengaged"] = Pdisengaged

    ## save csv
    df.to_csv(Path(data_path, f"glm_hmm_analysis.csv"), index=False)
    print("glm-hmm analysis updated")


