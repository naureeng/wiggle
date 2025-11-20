import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from itertools import chain
import ast

## obtain all mouse names
data_path = "/nfs/gatsbystor/naureeng/"
df = pd.read_csv(Path(data_path, f"bwm_mouse_stats.csv"))
mouse_names = df["mouse_names"].tolist()
print(len(mouse_names), "mice")

def build_wheel_csv_per_mouse(subject_name, data_path):
    eids = np.load(Path(data_path) / subject_name / f"{subject_name}_eids_data.npy")
    print(f"{subject_name}: {len(eids)} sessions")
    csv_learn = []
    for i in range(len(eids)):
        eid = eids[i]
        path = Path(data_path) / subject_name / f"{eid}/{eid}_wheelData_total.csv"
        if os.path.exists(path):
            csv_learn.append(path)

    ## concatenate csvs
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_learn], ignore_index=True)
    df_csv_concat.to_csv(Path(data_path) / subject_name / f"{subject_name}_wheel_sum.csv", index=False)
    print(f"{subject_name}: wheel_sum csv saved")

bwm_control_gain = pd.DataFrame([], columns=["subject_name", "median_amplitude", "mean_amplitude", "mean_duration", "gain", "count"])

for i in range(len(mouse_names)):
    try:
        subject_name = mouse_names[i]
        df_mouse = pd.read_csv(Path(data_path) / subject_name / f"{subject_name}_wheel_sum.csv", index_col=0) 

        ## convert string to list
        df_mouse["extrema_heights"] = df_mouse["extrema_heights"].apply(ast.literal_eval)

        ## absolute value of list
        df_mouse["extrema_heights"] = df_mouse["extrema_heights"].apply(
            lambda x: [abs(v) for v in x] if isinstance(x, list) else x)

        ## query groups
        df_low_gain = df_mouse.query("abs(contrast)==0.0625 and n_extrema>=2 and feedbackType==1")
        df_high_gain = df_mouse.query("abs(contrast)==0.50 and n_extrema>=2 and feedbackType==1")

        ## flatten list
        flattened_extrema_low_gain  = list(chain.from_iterable(df_low_gain["extrema_heights"]))
        flattened_extrema_high_gain = list(chain.from_iterable(df_high_gain["extrema_heights"]))

        # Add rows for low and high gain
        if len(flattened_extrema_low_gain) > 0:
            bwm_control_gain.loc[len(bwm_control_gain)] = {
                "subject_name": subject_name,
                "median_amplitude": np.median(flattened_extrema_low_gain),
                "mean_amplitude": np.mean(flattened_extrema_low_gain),
                "mean_duration": df_low_gain["jitter_duration"].mean(),
                "gain": "low",
                "count": len(df_low_gain)
            }

        if len(flattened_extrema_high_gain) > 0:
            bwm_control_gain.loc[len(bwm_control_gain)] = {
                "subject_name": subject_name,
                "median_amplitude": np.median(flattened_extrema_high_gain),
                "mean_amplitude": np.mean(flattened_extrema_high_gain),
                "mean_duration": df_high_gain["jitter_duration"].mean(),
                "gain": "high", 
                "count": len(df_high_gain)
            }
    except:
        pass
        print(f"no data: {subject_name}")

## save df as csv
csv_path = Path(data_path) / "bwm_control_gain.csv"
bwm_control_gain.to_csv(csv_path, index=False)
print(bwm_control_gain)
print(f"{str(csv_path)} saved")

