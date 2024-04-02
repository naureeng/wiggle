import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from pathlib import Path

def build_mouse_wheel_csv(subject_name, data_path):
    """Obtains csv across sessions

    Concatenates csv files for N = 1 mouse
    
    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files

    """
    
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_wheel.npy"))
    print(f"{subject_name}: {len(eids)} sessions")
    csv_files = [Path(data_path, subject_name, f"{eid}/{eid}_wheelData.csv") for eid in eids]

    ## concatenate csvs
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    print(df_csv_concat)
    df_csv_concat.to_csv(Path(data_path, subject_name, f"{subject_name}_total.csv"), index=False)
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
            low_contrast_data = df_eid.query("abs(goCueRT-stimOnRT)<=0.05 and abs(contrast)==0.0625 and duration<=1") ## query for low contrast data  
            low_contrast_accu = low_contrast_data["feedbackType"].mean() ## accuracy
            low_contrast_wiggle = low_contrast_data["n_extrema"].mean() ## prop wiggle

            if len(low_contrast_data)>=1: ## check that there is at least one trial
                ## save data
                low_contrast_accu_mouse.append(low_contrast_accu); low_contrast_wiggle_mouse.append(low_contrast_wiggle)
        except Exception as e:
            print(f"Error processing session {eid}: {str(e)}")

    return low_contrast_accu_mouse, low_contrast_wiggle_mouse


def build_fix_K_speed_accu(subject_name, data_path, K):
    df_mouse = pd.read_csv(Path(data_path, subject_name, f"{subject_name}_total.csv"))
    df_mouse["feedbackType"] = df_mouse["feedbackType"].replace(-1,0)
    low_contrast_data = df_mouse.query(f"abs(goCueRT-stimOnRT)<=0.05 and abs(contrast)==0.0625 and duration<=1 and n_extrema=={K}")

    speed_bins = np.arange(0,300,10)
    low_contrast_speed_K = []; low_contrast_accu_K = []
    for i in range(len(speed_bins)-1):
        start_bin = speed_bins[i]
        end_bin = speed_bins[i+1]
        low_contrast_bin = low_contrast_data.query(f"speed>={start_bin} and speed<{end_bin}")

        if len(low_contrast_bin)>=10: ## check that there is at least ten trials
            print(len(low_contrast_bin), "#trials in bin")
            low_contrast_accu_bin = low_contrast_bin["feedbackType"].mean()
            low_contrast_speed_bin = low_contrast_bin["speed"].mean()
            low_contrast_speed_K.append(low_contrast_speed_bin); low_contrast_accu_K.append(low_contrast_accu_bin)

    return low_contrast_speed_K, low_contrast_accu_K

