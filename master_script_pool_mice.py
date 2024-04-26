## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from analysis_utils import compute_pearsonr_wheel_accu

## obtain mouse names
pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
pth_res = Path(pth_dir, 'results')
pth_res.mkdir(parents=True, exist_ok=True)
mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) ## load mouse_names

def compute_pearsonr_across_mice(mouse_names, data_path):
    """Compute Pearson r correlation coefficent between # of wheel direction changes and proportion correct across mice

    Args:
        mouse_names (list): list of strings of mouse names

    """
    mice_pearsonr_wheel_accu = []
    for subject_name in mouse_names:
        try:
            r = compute_pearsonr_wheel_accu(subject_name, data_path)
            mice_pearsonr_wheel_accu.append(r) 
        except Exception as e:
            print(f"Error processing mouse: {str(e)}")

    np.save(Path(data_path).joinpath(f"mice_pearsonr_wheel_accu.npy"), mice_pearsonr_wheel_accu)
    print(f"pearson r correlation cofficients saved for N = {len(mouse_names)} mice")

def sort_mice(mouse_names, data_path):
    """Sort mice by Pearson r correlation coefficient between # of wheel direction changes and proportion correct across mice
    
    Args:
        mouse_names (list): list of strings of mouse names
        data_path (str): data path to store files

    """
    mice_pearsonr_wheel_accu = np.load(Path(data_path).joinpath(f"mice_pearsonr_wheel_accu.npy"))
    pos_idx = np.where(mice_pearsonr_wheel_accu > 0)[0]
    neg_idx = np.where(mice_pearsonr_wheel_accu <= 0)[0]

    pos_mouse_names = [mouse_names[i] for i in pos_idx]
    neg_mouse_names = [mouse_names[i] for i in neg_idx]

    print(len(pos_mouse_names), "good wigglers")
    print(len(neg_mouse_names), "bad wigglers")

    ## save data
    np.save(Path(data_path).joinpath(f"good_wigglers.npy"), pos_mouse_names)
    np.save(Path(data_path).joinpath(f"bad_wigglers.npy"), neg_mouse_names)

## main script
if __name__=="__main__":
    #compute_pearsonr_across_mice(mouse_names, pth_dir)
    sort_mice(mouse_names, pth_dir)

