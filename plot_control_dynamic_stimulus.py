import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon
from plot_utils import set_figure_style
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator

open_path = "/nfs/gatsbystor/naureeng/ortiz_2020/Open Loop-20251022T142612Z-1-001/Open Loop"
open_df   = pd.read_csv(Path(open_path) / "open_loop_high_contrast_n_changes.csv")

close_path= "/nfs/gatsbystor/naureeng/ortiz_2020/Closed Loop-20251022T142034Z-1-001/Closed Loop"
close_df  = pd.read_csv(Path(close_path) / "pupil_contrast_analysis.csv")

open_data = open_df.groupby(["file"])["changes_per_second"].mean().values.tolist()
close_data = close_df.query("contrast == 64")["changes_per_second"].tolist()

# Combine open and closed loop data into one DataFrame
open_df_summary = pd.DataFrame({
    "condition": "dynamic",
    "num_changes": open_data
})

close_df_summary = pd.DataFrame({
    "condition": "static",
    "num_changes": close_data
})

combined_df = pd.concat([open_df_summary, close_df_summary], ignore_index=True)


## test run
svfg_test = plt.figure(figsize=(8,8))
set_figure_style()
plt.close()

## prettify plot
svfg = plt.figure(figsize=(8,8))

ax = sns.boxplot(data=combined_df, x="condition", y="num_changes", palette=["#3b8bc2", "#f45a5a"], showfliers=False, width=0.4, saturation=1, linewidth=1.5)

m1  = combined_df.groupby(["condition"])["num_changes"].median().values.tolist()
st1 = combined_df.groupby(["condition"])["num_changes"].std().values.tolist()
cts = combined_df.groupby(["condition"])["num_changes"].count().values.tolist() 

## put legend
palette = ["#3b8bc2", "#f45a5a"]

legend_elements = [
            Patch(facecolor=palette[0], label='Color Patch'),
            Patch(facecolor=palette[1], label='Color Patch')]

ax.legend(handles=legend_elements, labels=[
        f"N = {int(cts[0]):,} ({round(m1[0],2)} ± {round(st1[0],2)})",
        f"N = {int(cts[1]):,} ({round(m1[1],2)} ± {round(st1[1],2)})"],
        frameon=False, fontsize=20, title="# sessions", title_fontsize=20, bbox_to_anchor=(0.95,1.0))

## put stats annotations 
pairs=[("dynamic", "static")]
order = ["dynamic", "static"]

# Paired Wilcoxon test (low vs high)
annotator = Annotator(ax, pairs, data=combined_df, x="condition", y="num_changes", order=order)
annotator.configure(test='Mann-Whitney', text_format='full', loc='outside', fontsize=20)
annotator.apply_and_annotate()

## labels
plt.xlabel("stimulus and loop type", fontsize=28)
plt.ylabel("mean # movements per sec", fontsize=28)

set_figure_style()
sns.despine(trim=False, offset=4)
plt.ylim(bottom=0)
plt.tight_layout()

plt.savefig(Path(open_path, "dynamic_stimulus_control.svg"))
print("boxplot saved")
