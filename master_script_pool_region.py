## session-wide analysis for brain region

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
from plot_utils import set_figure_style, build_legend

## load data
data_path = "/nfs/gatsbystor/naureeng/"
reg = "VISp"

eids_final = np.load(Path(data_path) / reg / f"{reg}_eids_final.npy")
print(len(eids_final), f"{reg} sessions")


## distribution of decoder accuracy

results = {K: {"fano_factor": [], "cov": []} for K in range(5)}

for i in range(0,1):
    eid = eids_final[i]
    fano_factor_across_session = []
    cov_across_session = []

    with open(Path(data_path) / reg / f'{eid}/results_K_decoder.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)

"""

        for K in [0, 1, 2, 3, 4]:
            mu_trial = data[K]["mu_cell"]
            st_trial = data[K]["st_cell"]
            fano_factor = st_trial**2 / mu_trial
            cov = st_trial / mu_trial
            fano_factor_across_session.append(fano_factor)
            cov_across_session.append(cov)

            results[K]["fano_factor"].append(fano_factor)
            results[K]["cov"].append(cov)

# converting results to a DataFrame for each K value
df_fano_factor = pd.DataFrame({K: results[K]["fano_factor"] for K in results})
df_cov = pd.DataFrame({K: results[K]["cov"] for K in results})

## plot data
data = df_fano_factor

plt.figure(figsize=(10,8))
set_figure_style()
sns.boxplot(data=data, linewidth=3, showfliers=False)
sns.stripplot(data=data, jitter=True, edgecolor="gray", linewidth=1)
plt.ylabel(r"Fano Factor ($\sigma_{cell} ^2 / \mu_{cell}$)", fontsize=28)
plt.xlabel("#changes in wheel direction", fontsize=28)
plt.title(f"N = {len(eids_final)} {reg} sessions", fontsize=28, fontweight="bold")
plt.ylim(bottom=0)

## build legend
m1 = data.mean(axis=0).values.tolist()
st1 = data.std(axis=0).values.tolist()
cts = np.repeat(len(eids_final), 5)
build_legend(m1, st1, cts)

# set y-axis limit to the maximum count
max_count = max(data.max())
plt.ylim(0, round(max_count,-1))

sns.despine(trim=False, offset=8)
plt.tight_layout()

plt.show()
"""
