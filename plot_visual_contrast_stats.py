import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon
from plot_utils import set_figure_style
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator
import os
import ast
from motion_energy_utils import generate_distinct_colors
import statsmodels.api as sm
import statistics

## analysis per mouse
pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
df = pd.read_csv(Path(pth_dir) / "bwm_mean_jitter_duration.csv")
df_counts = pd.read_csv(Path(pth_dir) / "bwm_mean_jitter_duration_counts.csv")
df = df.dropna(axis=1, how="any")
print(df)
#df = df * (35/np.rad2deg(0.3))
df_counts = df_counts.dropna(axis=1, how="any")
weights = df_counts.sum().values.tolist()

## test run
svfg_test = plt.figure(figsize=(8,8))
set_figure_style()
plt.close()

## prettify plot
svfg = plt.figure(figsize=(8,8))
palette = generate_distinct_colors("Grays", 5)

ax = sns.boxplot(data=df,showfliers=False, width=0.5, saturation=1, linewidth=1.5, palette=palette)

m1 = df.median()
st1 = df.std()
cts = df.count()

## put legend

legend_elements = [
            Patch(facecolor=palette[0], label='Color Patch'),
            Patch(facecolor=palette[1], label='Color Patch'),
            Patch(facecolor=palette[2], label='Color Patch'),
            Patch(facecolor=palette[3], label='Color Patch'),
            Patch(facecolor=palette[4], label='Color Patch')]

ax.legend(handles=legend_elements, labels=[
        f"N = {int(cts[0]):,} ({round(m1[0],2)} ± {round(st1[0],2)})",
        f"N = {int(cts[1]):,} ({round(m1[1],2)} ± {round(st1[1],2)})",
        f"N = {int(cts[2]):,} ({round(m1[2],2)} ± {round(st1[2],2)})",
        f"N = {int(cts[3]):,} ({round(m1[3],2)} ± {round(st1[3],2)})",
        f"N = {int(cts[4]):,} ({round(m1[4],2)} ± {round(st1[4],2)})"
        ],
        frameon=False, fontsize=20, title="# mice", title_fontsize=20, bbox_to_anchor=(0.95,1.0))

## fit line of best-fit
medians = m1.values.tolist()
x = np.arange(0, len(medians))
y = np.array(medians)
weights = np.array(weights)

# Add intercept term for regression
X = sm.add_constant(x)

model = sm.WLS(y, X, weights=weights)
results = model.fit()
print(results.summary())

# --- Extract statistics ---
slope = results.params[1]
intercept = results.params[0]
r_squared = results.rsquared
p_value = results.pvalues[1]

# --- Plot regression line ---
plt.plot(x, results.fittedvalues, color="k", lw=2)

# --- Annotate title with R² and p-value ---
plt.title(f"R² = {r_squared:.3f}, p = {p_value:.3e}", fontsize=30, fontweight="bold")

## labels
plt.xticks([0, 1, 2, 3, 4], ["0", "6.25", "12.5", "25", "100"], fontsize=28)
plt.xlabel("visual contrast (%)", fontsize=28)
plt.ylabel("mean wiggle occurrence", fontsize=28)
set_figure_style()
plt.ylim(bottom=0)
sns.despine(trim=False, offset=4)
plt.tight_layout()

contrasts = df.columns.astype(float)  # [0.0, 0.0625, 0.125, 0.25, 1.0]

# reshape both into long form
df_long = df.melt(var_name='contrast', value_name='value')
counts_long = df_counts.melt(var_name='contrast', value_name='ntrials')

# verify they align
assert df_long.shape[0] == counts_long.shape[0], "Mismatch after melting"

# prepare model inputs
y = df_long['value']
X = df_long['contrast'].astype(float)
X = sm.add_constant(X)
weights = counts_long['ntrials'].astype(float)

# fit weighted regression
model = sm.WLS(y, X, weights=weights)
results = model.fit()

print(results.summary())

##plt.savefig(Path(pth_dir, "wiggle_occurrence_contrast.svg"))
##print("boxplot saved")
