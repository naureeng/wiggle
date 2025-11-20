## analysis of ortiz et al, 2020
## closed loop, 1.25% and 4% contrast, 0.5 cpd

import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from pathlib import Path

# Path to your folder
folder = "/nfs/gatsbystor/naureeng/ortiz_2020/Open Loop-20251022T142612Z-1-001/Open Loop"

def count_direction_changes(trace):
    # Compute sign of velocity
    sign = np.sign(trace).astype(float)  # convert to float so we can use NaN
    # Remove zeros (avoid spurious sign flips)
    sign[sign == 0] = np.nan
    # Count sign changes (ignoring NaNs)
    return np.nansum(np.diff(sign) != 0)

# Loop over all .mat files in folder
results = []
for file in os.listdir(folder):
    if file.endswith(".mat"):
        path = os.path.join(folder, file)
        data = loadmat(path, squeeze_me=True)

        # Extract signals (mimic MATLAB code)
        rotation = data.get("dispRotation", None)
        stimulus = data.get("digitalDiode", None)
        rewards = data.get("valve", None)
        igor_duration = data.get("igorDuration", 1/10000)

        if rotation is None or stimulus is None:
            continue

        downsample_const = 10
        rotation = rotation[::downsample_const]
        stimulus = stimulus[::downsample_const] * -1  # make same convention as MATLAB
        rewards = rewards[::downsample_const] if rewards is not None else None

        # Determine trial boundaries (stimulus on/off transitions)
        stim_on = np.where(np.diff(stimulus) != 0)[0]
        trial_starts = stim_on[::2] if len(stim_on) % 2 == 0 else stim_on[:-1:2]
        trial_ends = stim_on[1::2] if len(stim_on) % 2 == 0 else stim_on[1::2]

        # Compute number of direction changes (k) for each trial
        for start, end in zip(trial_starts, trial_ends):
            # Compute trial duration (in seconds)
            sampling_interval = 1 / 1000  # seconds per sample (after downsampling)
            duration_samples = (end - start) * sampling_interval
            trace = rotation[start:end]
            k = count_direction_changes(trace)
            # Normalize number of changes per second or per sample
            k_norm = k / duration_samples if duration_samples > 0 else np.nan
            results.append({
                "file": file,
                "trial_start": start,
                "trial_end": end,
                "num_direction_changes": k,
                "trial_duration": duration_samples,
                "changes_per_second": k_norm
                })

# Convert to DataFrame
df = pd.DataFrame(results)
df.to_csv(Path(folder) / "open_loop_high_contrast_n_changes.csv", index=False)
print("csv saved for 64% open-loop trials")
