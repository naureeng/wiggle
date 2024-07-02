## is V1 tickled by mouse wiggles?
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ephys_utils import *
from wheel_utils import get_extended_trial_windows
from curate_eids import curate_eids_neural
from decoder_utils import NN
from plot_utils import build_legend, set_figure_style

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

## VISp sessions
reg = "VISp"
data_path = "/nfs/gatsbystor/naureeng/"

eids, probes = sessions_with_region(reg, one=one)
print(len(eids), "sessions")

## example 
eid = eids[0]
print(eid)
probe = probes[0]

## compute drift
max_drift = compute_maximum_drift(eid, probe)

## parameters
alignment_type = "motionOnset" ## align at t=0
time_lower = -0.4 ## (sec) pre-400 msec 
time_upper = 0.4 ## (sec)  post-400 msec 
T_BIN = 0.02 ## (sec) 20 msec bin size

## load DataFrame
df = pd.read_csv(f"/nfs/gatsbystor/naureeng/VISp/{eid}/{eid}.csv")
d, trial_issue = get_extended_trial_windows(eid, time_lower, time_upper, alignment_type)
## drop trials with issues in wheel data from DataFrame
bad_df = df.index.isin(trial_issue)
df = df[~bad_df]
print(df)

"""
## sort trials by K
mu_data_K = []; std_data_K = []; n_trials_K = []
for K in [0, 1, 2, 3, 4]:
    if K == 4:
        df_K = df.query(f"abs(goCueRT-stimOnRT)<=0.05 and n_extrema>={K}") ## for K = 4, use K >=4 trials for more data
    else:
        df_K = df.query(f"abs(goCueRT-stimOnRT)<=0.05 and n_extrema=={K}") ## obtain K trials

    trials_K = df_K.index.tolist() ## obtain indices of K trials
    dict_K = {key: d[key] for key in trials_K} ## filter dict for K trials

    ## plot psth
    Res = pair_firing_rate_contrast(eid, probe, dict_K, reg, time_lower, time_upper, T_BIN, alignment_type)
    mu_data, std_data, n_trials_data, n_units = obtain_psth(Res, time_lower, time_upper, T_BIN)
    mu_data_K.append(mu_data); std_data_K.append(std_data); n_trials_K.append(n_trials_data)

plot_psth(mu_data_K, std_data_K, n_trials_K, time_lower, time_upper, T_BIN, n_units)
"""

## obtain decoder training input
## period 1: -0.4 sec from motionOnset
## period 2: +0.4 sec from motionOnset
Res = pair_firing_rate_contrast(eid, probe, d, reg, time_lower, time_upper, T_BIN, alignment_type)
n_units, n_bins, n_trials = Res.shape
Res_single_trial = np.nanmean(Res, axis=0)
print(Res_single_trial.shape)

## obtain decoder training output
## stimulus side: contrast sign
y0 = np.sign(df["contrast"].values)
print(y0.shape)
num_right = np.sum(y0 == -1.0)
num_left  = np.sum(y0 == 1.0)
num_zero  = np.sum(y0 == 0.0)
print(f"N = {len(df)} trials: {num_right} right side, {num_left} left side, {num_zero} zero contrast")

## decoder for N = 1 session
acs, P_stimulus = NN(Res_single_trial.T, y0.T, shuf=False)
df["P_stimulus"] = P_stimulus

df_plot = df.copy()
df_plot['n_extrema_grouped'] = df_plot['n_extrema'].clip(upper=4)
m1 = df_plot.groupby(["n_extrema_grouped"])["P_stimulus"].mean().values.tolist()
st1 = df_plot.groupby(["n_extrema_grouped"])["P_stimulus"].std().values.tolist()
cts = df_plot["n_extrema_grouped"].value_counts().sort_index().values.tolist()

svfg = plt.figure(figsize=(10,8))
set_figure_style()
sns.boxplot(data=df_plot, x="n_extrema_grouped", y="P_stimulus", order=[0,1,2,3,4], showfliers=False, linewidth=3)
sns.stripplot(data=df_plot, x="n_extrema_grouped", y="P_stimulus", jitter=True, edgecolor="gray", linewidth=1)
build_legend(m1, st1, cts)
plt.ylabel("P(stimulus side)", fontsize=28)
plt.xlabel("#changes in wheel direction", fontsize=28)
plt.text(0.0, 1.0, f"N = {n_units} units, {n_trials} trials, 1 session", fontsize=28, ha="left", fontweight="bold") 
plt.ylim([0,1])
sns.despine(trim=False, offset=8)
plt.tight_layout()
plt.show()

