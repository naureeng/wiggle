import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon
from plot_utils import set_figure_style
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator

data_path = "/nfs/gatsbystor/naureeng/"
bwm_control_gain = pd.read_csv(Path(data_path, "bwm_control_gain.csv"))

# Convert from wheel deg to visual deg
bwm_control_gain["median_amplitude"] = bwm_control_gain["median_amplitude"] * (35/np.rad2deg(0.3))
bwm_control_gain["mean_amplitude"] = bwm_control_gain["mean_amplitude"] * (35/np.rad2deg(0.3))

# Keep only subjects with both low & high gain
paired_subjects = (
    bwm_control_gain.groupby("subject_name")["gain"].nunique() == 2
)
paired_subjects = paired_subjects[paired_subjects].index
bwm_paired = bwm_control_gain[bwm_control_gain["subject_name"].isin(paired_subjects)]

# Weighted median amplitude per subject × gain
weighted = (
    bwm_paired.groupby(["subject_name", "gain"])
    .apply(lambda x: np.average(x["mean_amplitude"], weights=x["count"]))
    .reset_index(name="weighted_mean_amplitude")
)

## test run
svfg_test = plt.figure(figsize=(8,8))
set_figure_style()
plt.close()

## prettify plot
svfg = plt.figure(figsize=(8,8))

ax = sns.boxplot(data=weighted, x="gain", y="weighted_mean_amplitude", palette=["#3b8bc2", "#f45a5a"],showfliers=False, width=0.5, saturation=1, linewidth=1.5)

m1 = weighted.groupby(["gain"])["weighted_mean_amplitude"].median().values.tolist()
st1 = weighted.groupby(["gain"])["weighted_mean_amplitude"].std().values.tolist()
cts = bwm_control_gain.groupby(["gain"])["count"].sum().values.tolist() 

## put legend
palette = ["#3b8bc2", "#f45a5a"]

legend_elements = [
            Patch(facecolor=palette[0], label='Color Patch'),
            Patch(facecolor=palette[1], label='Color Patch')]

ax.legend(handles=legend_elements, labels=[
        f"N = {int(cts[0]):,} ({round(m1[0],2)} ± {round(st1[0],2)})",
        f"N = {int(cts[1]):,} ({round(m1[1],2)} ± {round(st1[1],2)})"],
        frameon=False, fontsize=20, title="# wiggles", title_fontsize=20, bbox_to_anchor=(0.95,1.0))

## put stats annotations 
pairs=[("high", "low")]
order = ["high", "low"]

# Paired Wilcoxon test (low vs high)
annotator = Annotator(ax, pairs, data=weighted, x="gain", y="weighted_mean_amplitude", order=order)
annotator.configure(test='Mann-Whitney', text_format='full', loc='outside', fontsize=20)
annotator.apply_and_annotate()

## labels
plt.xlabel("visuo-motor gain", fontsize=28)
plt.ylabel("mean wiggle amplitude (visual deg)", fontsize=28)

set_figure_style()
sns.despine(trim=False, offset=4)
plt.tight_layout()

plt.savefig(Path(data_path, "gain_control.svg"))
print("boxplot saved")
