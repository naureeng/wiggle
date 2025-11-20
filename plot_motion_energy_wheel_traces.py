import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def load_aligned_wheel_trace(data_path, subject_name, eid, trial_no, time_window=(-0.1, 0.6)):
    wheel_path = Path(data_path) / subject_name / eid / f"{eid}.wheel"

    with open(wheel_path, 'rb') as f:
        t_eid, p_eid, v_eid, motionOnset_idx, stimOnset_idx, goCueOnset_idx, responseTime_idx = pickle.load(f)

    t = np.array(t_eid[trial_no])
    p = np.array(p_eid[trial_no])
    stim_idx = stimOnset_idx[trial_no]
    stim_time = t[stim_idx]

    # Align to stim onset
    t_aligned = t - stim_time
    mask = (t_aligned >= time_window[0]) & (t_aligned <= time_window[1])
    return t_aligned[mask], p[mask]

def plot_example_trials_by_motion_bin(data_path, subject_name, time_window=(-0.1, 1.0)):

    # Load sessions
    eids = np.load(Path(data_path, f"{subject_name}/{subject_name}_eids_data.npy"))
    print(f"{subject_name}: {len(eids)} sessions")

    all_df = []
    for eid in eids:
        csv_path = Path(data_path) / subject_name / eid / f"{eid}_wheelData_total.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["eid"] = eid
        bins = np.arange(500,90500,10000)
        labels = [f'({bins[j]},{bins[j+1]}]' for j in range(len(bins)-1)]
        df["motion_bin"] = pd.cut(df["motion_energy_norm"], bins=bins, labels=labels, right=True)  # Binned data by motion energy
        df["feedbackType"] = df["feedbackType"].replace(-1, 0)  # To do mean calculation

        # restrict to trials with good RT alignment
        df_filt = df.query("abs(goCueRT - stimOnRT) <= 0.05 and duration<=1")
        all_df.append(df_filt)

    if len(all_df) == 0:
        print("No usable files found.")
        return

    df_all = pd.concat(all_df, ignore_index=True)
    df_all = df_all.dropna(subset=["motion_bin"])

    print(f"Loaded {len(df_all)} trials with motion energy")


    representative_trials = {}

    for b in labels:
        df_bin = df_all[df_all["motion_bin"] == b]
        if len(df_bin) == 0:
            representative_trials[b] = None
        else:
            representative_trials[b] = df_bin.iloc[-1]   # select example


    fig, axs = plt.subplots(8, 1, figsize=(12, 25), sharex=True)
    fig.suptitle(f"{subject_name}: Example wheel traces by motion-energy bin", fontsize=22)

    for i, b in enumerate(labels):
        ax = axs[i]
        trial = representative_trials[b]

        if trial is None:
            ax.text(0.5, 0.5, f"No trials in {b}", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
            continue

        eid = trial["eid"]
        trial_no = int(trial["trial_no"])

        # Load aligned wheel trace
        t_aligned, p_aligned = load_aligned_wheel_trace(data_path, subject_name, eid, trial_no, time_window)

        # Plot
        ax.plot(t_aligned, p_aligned, color="k", linewidth=2)
        ax.axvline(0, color="r", linestyle="--")

        ax.set_ylabel(b, rotation=0, labelpad=40, fontsize=12)
        ax.set_title(f"Motion bin {b} — eid={eid}, trial={trial_no}", fontsize=14)
        ax.grid(True)

    axs[-1].set_xlabel("Time from stimulus onset (s)", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("/nfs/gatsbystor/naureeng/test_motion_energy_traces.svg")

if __name__=="__main__":

    mouse_name = "ibl_witten_16"
    time_window = "total"
    data_path = Path("/nfs/gatsbystor/naureeng/")

    bins = np.arange(500, 90500, 10000)
    labels = [f'({bins[j]},{bins[j+1]}]' for j in range(len(bins)-1)]
    order = labels[:]  # 8 bins in increasing order

    eids = np.load(Path(data_path, f"{mouse_name}/{mouse_name}_eids_data.npy"))
    print(f"{mouse_name}: {len(eids)} sessions")
    plot_example_trials_by_motion_bin(data_path, mouse_name, time_window=(-0.1, 1.0))
