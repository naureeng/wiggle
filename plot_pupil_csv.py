import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu
from plot_utils import set_figure_style
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator

df_all_trials = pd.read_csv("/nfs/gatsbystor/naureeng/lightning_pose_wiggle_200msec.csv")
print(len(df_all_trials["subject"].unique()))

## test run
svfg_test = plt.figure(figsize=(8,8))
set_figure_style()
plt.close()

## prettify plot
svfg = plt.figure(figsize=(8,8))

ax = sns.boxplot(data=df_all_trials, x="trial_type", y="pupil_diameter", palette=["#3b8bc2", "#f45a5a"],showfliers=False, width=0.5, saturation=1, linewidth=1.5)

m1 = df_all_trials.groupby(["trial_type"])["pupil_diameter"].median().values.tolist()
st1 = df_all_trials.groupby(["trial_type"])["pupil_diameter"].std().values.tolist()
cts = [1472, 1472]
print(cts)

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
pairs=[("wiggle", "nonwiggle")]
order = ["wiggle", "nonwiggle"]

# Paired Wilcoxon test (low vs high)
annotator = Annotator(ax, pairs, data=df_all_trials, x="trial_type", y="pupil_diameter", order=order)
annotator.configure(test='Mann-Whitney', text_format='full', loc='outside', fontsize=20)
annotator.apply_and_annotate()

## labels
plt.xlabel("trial type", fontsize=28)
plt.ylabel("mean pupil diameter (a.u.)", fontsize=28)

set_figure_style()
plt.ylim(bottom=0)
sns.despine(trim=False, offset=4)
plt.tight_layout()

plt.savefig("/nfs/gatsbystor/naureeng/pupil_control.svg")
print("boxplot saved")

