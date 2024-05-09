import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from pathlib import Path

def build_mouse_wheel_csv(subject_name, suffix, eid_str, data_path):
    """Obtains csv across sessions

    Concatenates csv files for N = 1 mouse
    
    Args:
        subject_name (str): mouse name
        suffix (str): suffix of eid csv files
        eid_str (str): suffix of eids npy file
        data_path (str): path to store data files

    """
    
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_{eid_str}.npy"))
    print(f"{subject_name}: {len(eids)} sessions")
    csv_files = [Path(data_path, subject_name, f"{eid}/{eid}_{suffix}.csv") for eid in eids]

    ## concatenate csvs
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    df_csv_concat.to_csv(Path(data_path, subject_name, f"{subject_name}_{suffix}.csv"), index=False)
    print("dataframe saved to csv")


def build_prop_wiggle_vs_accuracy(subject_name, data_path): 
    """Obtains % wiggles vs accuracy

    Compute proportion wiggles vs accuracy in low visual contrast trials in N = 1 mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files
    
    Returns:
        low_contrast_accu_mouse (list): mean accuracy per session for N = 1 mouse 
        low_contrast_wiggle_mouse (list): mean proportion wiggles per session for N = 1 mouse 

    """
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_wheel.npy"))

    low_contrast_accu_mouse = []; low_contrast_wiggle_mouse = []

    for eid in eids:
        try:
            df_eid = pd.read_csv(Path(data_path, subject_name, f"{eid}/{eid}_wheelData.csv"))
            df_eid["feedbackType"] = df_eid["feedbackType"].replace(-1,0)
            low_contrast_data = df_eid.query("abs(contrast)==0.0625 and duration<=1") ## query for low contrast data  
            low_contrast_accu = low_contrast_data["feedbackType"].mean() ## accuracy
            low_contrast_wiggle = low_contrast_data["n_extrema"].mean() ## prop wiggle

            if len(low_contrast_data)>=1: ## check that there is at least one trial
                ## save data
                low_contrast_accu_mouse.append(low_contrast_accu); low_contrast_wiggle_mouse.append(low_contrast_wiggle)
        except Exception as e:
            print(f"Error processing session {eid}: {str(e)}")

    return low_contrast_accu_mouse, low_contrast_wiggle_mouse


def build_fix_K_speed_accu(subject_name, data_path, K, min_trials=30, bin_width=10):
    """Computes speed vs accuracy for a given #extrema

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files 
        K (int): # extrema
        min_trials (int): threshold for #trials per bin
        bin_width (int): speed bin size [deg/sec]

    Returns:
        low_contrast_speed_K (list): mean speed across bins per mouse
        low_contrast_accu_K (list): mean accuracy across bins per mouse

    """
    df_mouse = pd.read_csv(Path(data_path, subject_name, f"{subject_name}_wheelData.csv"))
    df_mouse["feedbackType"] = df_mouse["feedbackType"].replace(-1,0)
   
    ## use K >= 4 to increase fraction of data
    if K == 4:
        low_contrast_data = df_mouse.query(f"duration<=1 and n_extrema>={K}")
    else:
        low_contrast_data = df_mouse.query(f"duration<=1 and n_extrema=={K}")

    ## compute speed and accuracy per speed bin
    speed_bins = np.arange(0, 300, bin_width)
    low_contrast_speed_K = []
    low_contrast_accu_K = []

    for i in range(len(speed_bins) - 1):
        start_bin = speed_bins[i]
        end_bin = speed_bins[i + 1]
        bin_mask = (low_contrast_data["speed"] >= start_bin) & (low_contrast_data["speed"] < end_bin)
        bin_data = low_contrast_data[bin_mask]

        if len(bin_data) >= min_trials: ## filter data by #trials
            accu_bin = bin_data["feedbackType"].mean()
            speed_bin = bin_data["visual_speed"].mean()
            low_contrast_speed_K.append(speed_bin)
            low_contrast_accu_K.append(accu_bin)

    return low_contrast_speed_K, low_contrast_accu_K


def compute_K_data(subject_name, data_path):
    """Compute visual speed [visual deg/sec] vs accuracy for a fixed # changes in wheel direction (K) 

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files

    Returns:
        data (list): Pearson correlation coefficient (r) between visual speed [visual deg/sec] and accuracy

    """

    data = []
    for K in [2,3,4]:
        low_contrast_speed_K, low_contrast_accu_K = build_fix_K_speed_accu(subject_name, data_path, K)
        if len(low_contrast_speed_K) >=10: ## minimum of 10 sessions required per mouse to do Pearson correlation
            try:
                m = pearsonr(low_contrast_speed_K, low_contrast_accu_K).statistic
            except:
                m = np.nan
            data.append(m)
        else:
            print(f"{subject_name} has < 10 data points to compute r")
            data.append(np.nan)

    return data



