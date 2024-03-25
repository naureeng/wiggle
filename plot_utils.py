## functions to plot variable of interest across eids per mouse

## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
    """
    PLOT_FOUR_BOXPLOT_GRPS outputs boxplot for N = >1 session, 1 mouse

    :param n_trials: #trials [np.array]
    :param x0: group0 data [list] (k = 0)
    :param x1: group1 data [list] (k = 1)
    :param x2: group2 data [list] (k = 2)
    :param x3: group3 data [list] (k = 3)
    :param x4: group4 data [list] (k >= 4)
    :param tstring: title [string]
    :param fname: folder name [string]
    :param mouse: subject name [string]
    :param x0_string: group0 label [string]
    :param x1_string: group1 label [string]
    :param x2_string: group2 label [string]
    :param x3_string: group3 label [string]
    :param x4_string: group4 label [string]
    :param ystring: yaxis label [string]
    :return boxplot [png file]

    """

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
    
    ## try/except clause to handle cases where the data is all zeros
    try:
        annotator = Annotator(ax, pairs, data=overall_df, order=order)

        if mouse == "": ## data analysis across multiple mice
            annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', line_width="4", fontsize=28, comparisons_correction="bonferroni") # U-test across mice with Bonferroni correction
        else: ## data analysis across sessions for one mouse
            annotator.configure(test='Wilcoxon', text_format='star', loc='outside', line_width="4", fontsize=28) # Wilcoxon Signed-Rank test for pairwise comparisons of same sessions within one mouse

        annotator.apply_and_annotate()
    except:
        pass

    ## despine
    sns.despine(trim=False, offset=8)
    plt.tight_layout()

    ## save plot
    pathname = f"/nfs/gatsbystor/naureeng/{fname}/"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    svfg.savefig(f"/nfs/gatsbystor/naureeng/{fname}/{mouse}_{fname}.png", dpi=300)
    print("boxplot img saved")
    plt.close()

