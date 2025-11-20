## analysis of ortiz et al, 2020
## closed loop, 1.25% and 4% contrast, 0.5 cpd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from scipy.io import loadmat
from pathlib import Path
from scipy.stats import mannwhitneyu
from plot_utils import set_figure_style
from statannotations.Annotator import Annotator

base = Path("/nfs/gatsbystor/naureeng/ortiz_2020/Closed Loop-20251022T142034Z-1-001/Closed Loop")

def process_file(mat_file):
    data = loadmat(mat_file)
    y = data['y']

    def unwrap_safe(i):
        if len(y) > i and y[i] is not None and len(y[i]) > 0:
            return np.array(y[i][0]).squeeze()
        return None

    # Unpack signals safely
    tone        = unwrap_safe(0)
    licks       = unwrap_safe(1)
    startSignal = unwrap_safe(2)
    rotation    = unwrap_safe(3)
    stimulus    = unwrap_safe(4)
    rewards     = unwrap_safe(5)
    contrast    = unwrap_safe(6)
    trialType   = unwrap_safe(7)

    # Check that necessary signals exist
    if startSignal is None or rotation is None or contrast is None:
        print(f"⚠️  Skipping {mat_file.name}: missing signal(s)")
        return pd.DataFrame()  # return empty frame so concat still works

    # Convert to int and ensure it's a flat array
    stimulus = np.ravel(stimulus).astype(int)

    # Identify transitions: 0→1 (onset), 1→0 (offset)
    stim_on_indices  = np.where(np.diff(stimulus) == 1)[0] + 1
    stim_off_indices = np.where(np.diff(stimulus) == -1)[0] + 1

    # Handle edge cases: if stimulus starts ON or ends ON
    if stimulus[0] == 1:
        stim_on_indices = np.insert(stim_on_indices, 0, 0)
    if stimulus[-1] == 1:
        stim_off_indices = np.append(stim_off_indices, len(stimulus))

    # Identify trial starts and ends
    trial_starts = np.where(np.diff(startSignal.astype(int)) == 1)[0] + 1
    trial_ends = np.append(trial_starts[1:], len(startSignal))
    
    trial_data = []
    for i, (start, end) in enumerate(zip(stim_on_indices, stim_off_indices)):
        sampling_interval = 1 / 1000  # seconds per sample
        duration_samples = (end - start) * sampling_interval
        rot = rotation[start:end]
        c = contrast[start]
        drot = np.diff(rot)
        drot[drot == 0] = np.nan
        n_changes = np.nansum(np.diff(np.sign(drot)) != 0)
        # Normalize number of changes per second or per sample
        k_norm = n_changes / duration_samples if duration_samples > 0 else np.nan
        trial_data.append({'trial': i, 'contrast': c, 'n_changes': n_changes, 'changes_per_second': k_norm})

    df = pd.DataFrame(trial_data)
    if df.empty:
        print(f"⚠️  {mat_file.name}: no valid trials found")
        return df

    per_contrast = df.groupby('contrast')['changes_per_second'].mean().reset_index()
    per_contrast['subject'] = mat_file.stem
    return per_contrast

# Load all .mat files and concatenate
all_files = list(base.glob("*_data.mat"))
results = pd.concat([process_file(f) for f in all_files], ignore_index=True)

# Focus on contrast == 2 and 64
subset = results[results['contrast'].isin([2, 64])]
print(subset)
subset.to_csv(base / "pupil_contrast_analysis.csv", index=False)
print("csv saved")

