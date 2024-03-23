## functions to plot variable of interest across eids per mouse

## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator


def reindex_df(df, weight_col):
    """expand the dataframe to prepare for resampling
    result is 1 row per count per sample"""
    df = df.reindex(df.index.repeat(df[weight_col]))
    df.reset_index(drop=True, inplace=True)
    return(df)


def figure_style():
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams["axes.linewidth"] = 2


def plot_four_boxplot_grps(n_trials, x0, x1, x2, x3, x4, tstring, fname, mouse, x0_string, x1_string, x2_string, x3_string, x4_string, ystring):

    ## open figure
    svfg = plt.figure(figsize=(10,8))
    figure_style()
    
    ## prepare dataframe from inputs
    grp0_df = pd.DataFrame(x0, columns=[f"{x0_string}"])
    grp1_df = pd.DataFrame(x1, columns=[f"{x1_string}"])
    grp2_df = pd.DataFrame(x2, columns=[f"{x2_string}"])
    grp3_df = pd.DataFrame(x3, columns=[f"{x3_string}"])
    grp4_df = pd.DataFrame(x4, columns=[f"{x4_string}"])
    counts_df = pd.DataFrame(n_trials, columns=["Count"])
    overall_df = pd.concat([grp0_df, grp1_df, grp2_df, grp3_df, grp4_df, counts_df], axis=1)
    overall_df = overall_df.dropna() ## drop any rows with nans
    overall_df_weighted = reindex_df(overall_df, weight_col="Count") ## perform weighted average of data based on #trials

    ## plot data
    plt.xticks([0,1,2,3,4], labels=[f"{x0_string}", f"{x1_string}", f"{x2_string}", f"{x3_string}", f"{x4_string}"])
    ax = sns.boxplot(data=overall_df_weighted, order=[f"{x0_string}", f"{x1_string}", f"{x2_string}", f"{x3_string}", f"{x4_string}"], linewidth=3, showfliers=False)
    sns.stripplot(data=overall_df, order=[f"{x0_string}", f"{x1_string}", f"{x2_string}", f"{x3_string}", f"{x4_string}"], jitter=True, edgecolor="gray", ax=ax, linewidth=3)
    plt.ylabel(f"{ystring}", fontsize=28)
    plt.xlabel(f"{tstring}", fontsize=28, fontweight="bold")
    plt.ylim(bottom=-0.1)

    ## annotations
    pairs=[(f"{x0_string}", f"{x4_string}"), (f"{x0_string}", f"{x2_string}"),  (f"{x2_string}", f"{x4_string}")]
    order = [f"{x0_string}", f"{x1_string}", f"{x2_string}", f"{x3_string}", f"{x4_string}"]

    annotator = Annotator(ax, pairs, data=overall_df, order=order)
    annotator.configure(test='Wilcoxon', text_format='star', loc='outside', line_width="4", fontsize=28) # Wilcoxon Signed-Rank test for pairwise comparisons of same sessions within one mouse
    annotator.apply_and_annotate()

    ## despine
    sns.despine(trim=False, offset=8)
    plt.tight_layout()
    plt.show()
