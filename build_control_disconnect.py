## analysis of baruchin et al, 2023
## wiggle occurrence in period between stimOnset and goCue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wheel_utils import find_nearest

base = Path("/nfs/gatsbystor/naureeng/baruchin_2023/ephys/")

total_mean_movement_occurrence_pre_goCue = []
total_mean_movement_occurrence_post_goCue = []
total_n_trials = []

for subject_path in base.glob("SS*"):  # or adjust glob pattern to match all mice
    subject_name = subject_path.name

    # Loop through all date folders under the subject directory
    for date in sorted(subject_path.glob("20*")):  # YYYY-MM-DD format

        df = pd.DataFrame([], columns=["stimOn_times", "goCue_times", "stimOff_times", "contrastLeft", "contrastRight"])

        ## time between stimOnset and goCue
        stimOn_intervals = np.load(Path(base / subject_name / date / "_ss_trials.stimOn_intervals.npy"))
        df["stimOn_times"] = stimOn_intervals[:,0]
        df["stimOff_times"] = stimOn_intervals[:,1]
        n_trials = len(stimOn_intervals)
        print(len(stimOn_intervals), "trials")

        goCue_times = np.load(Path(base / subject_name / date / "_ss_trials.goCue_times.npy"))
        df["goCue_times"] = goCue_times

        df["pre_goCue_duration"] = df["goCue_times"] - df["stimOn_times"]
        df["duration"] = df["stimOff_times"] - df["goCue_times"]

        ## query for 25% contrast trials
        contrastLeft = np.load(Path(base / subject_name / date / "_ss_trials.contrastLeft.npy"))
        contrastRight = np.load(Path(base / subject_name / date / "_ss_trials.contrastRight.npy"))
        df["contrastLeft"] = contrastLeft
        df["contrastRight"] = contrastRight

        ## array of wheel movements [start, stop] (sec)
        wheelMoves = np.load(Path(base / subject_name / date / "_ss_wheelMoves.intervals.npy"))
        wheel_start = wheelMoves[:,0]
        wheel_stop = wheelMoves[:,1]
        print(len(wheelMoves), "wheel movements")

        # count wheel moves in pre_goCue time per trial
        wheel_counts = []
        for start, end in zip(df["stimOn_times"], df["goCue_times"]):
            # count how many wheel moves overlap this trial interval
            count = np.sum((wheel_stop > start) & (wheel_start < end))
            wheel_counts.append(count)

        # count wheel moves in total time per trial
        wheel_total_counts = []
        for start, end in zip(df["goCue_times"], df["stimOff_times"]):
            # count how many wheel moves overlap this trial interval
            count = np.sum((wheel_stop > start) & (wheel_start < end))
            wheel_total_counts.append(count)

        df["n_wheelMoves"] = wheel_counts
        df["n_wheelMoves_total"] = wheel_total_counts
        print(df["n_wheelMoves"].sum(), "wheel counts in pre-goCue period")
        print(df["n_wheelMoves_total"].sum(), "wheel counts in goCue period")

        df_25 = df.query("contrastLeft==0.25 or contrastRight==0.25")
        n_wheelMoves = df_25["n_wheelMoves"].sum()
        total_pre_goCue_time = df_25["pre_goCue_duration"].sum()

        n_wheelMoves_post_goCue = df_25["n_wheelMoves_total"].sum()
        total_post_goCue_time = df_25["duration"].sum()

        mean_movement_occurrence_pre_goCue = n_wheelMoves / n_trials
        mean_movement_occurrence_post_goCue = n_wheelMoves_post_goCue / n_trials 

        print(f"25% contrast, pre-goCue: {mean_movement_occurrence_pre_goCue}")
        print(f"25% contrast, post-goCue: {mean_movement_occurrence_post_goCue}")
        
        total_mean_movement_occurrence_pre_goCue.append(mean_movement_occurrence_pre_goCue)
        total_mean_movement_occurrence_post_goCue.append(mean_movement_occurrence_post_goCue)
        total_n_trials.append(n_trials)

## save data
np.save(Path(base / "total_pre_goCue.npy"), total_mean_movement_occurrence_pre_goCue)
np.save(Path(base / "total_post_goCue.npy"), total_mean_movement_occurrence_post_goCue)
np.save(Path(base / "total_n_trials.npy"), total_n_trials)
print("mean movement occurrence saved")
