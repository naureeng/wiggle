## functions to plot data per mouse

## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
from scipy.stats import pearsonr
from numpy import nan
from matplotlib.patches import Patch
import pickle

def compute_n_trials_per_K(data_path, contrast_label):
    """Compute mean #trials across mice, across sessions

    :param data_path (str): data path to store files
    :param contrast_label (str): "high", "low", "zero" to specify contrast group

    :return m1_final   (list): mean #trials for k = [0, 1, 2, 3, 4]
    :return st1_final  (list): std #trials for k = [0, 1, 2, 3, 4]
    :return cts_final  (list): total #trials for k = [0, 1, 2, 3, 4]

    """

    mouse_names = np.load(Path(data_path, "mouse_names.npy"), allow_pickle=True)

    m1_overall, st1_overall, counts = [], [], []

    for subject_name in mouse_names:
        data = np.load(Path(data_path, subject_name, f"{subject_name}_n_trials_K_{contrast_label}.npy"), allow_pickle=True)
        
        ## compute average across sessions in one mouse
        m1  = [np.nanmean(data[i]) for i in range(len(data))]
        st1 = [np.nanstd(data[i]) for i in range(len(data))]
        ct1 = [np.nansum(data[i]) for i in range(len(data))]

        m1_overall.append(m1); st1_overall.append(st1); counts.append(ct1)

    ## compute average across sessions in all mice
    m1_final  = np.nanmean(m1_overall, axis=0)
    st1_final = np.nanmean(st1_overall, axis=0)
    cts_final = np.nansum(counts, axis=0)

    return m1_final, st1_final, cts_final

def build_legend(m1, st1, cts):
    ax = plt.gca()
    legend_elements = [
            Patch(facecolor='tab:blue', edgecolor='k', label='Color Patch', linewidth=2),
            Patch(facecolor='tab:orange', edgecolor='k', label='Color Patch', linewidth=2),
            Patch(facecolor='tab:green', edgecolor='k', label='Color Patch', linewidth=2),
            Patch(facecolor='tab:red', edgecolor='k', label='Color Patch', linewidth=2),
            Patch(facecolor='tab:purple', edgecolor='k', label='Color Patch', linewidth=2)
            ]

    ax.legend(handles=legend_elements, labels=[
        f"N = {int(cts[0]):,} ({round(m1[0],2)} ± {round(st1[0],2)})", 
        f"N = {int(cts[1]):,} ({round(m1[1],2)} ± {round(st1[1],2)})", 
        f"N = {int(cts[2]):,} ({round(m1[2],2)} ± {round(st1[2],2)})", 
        f"N = {int(cts[3]):,} ({round(m1[3],2)} ± {round(st1[3],2)})", 
        f"N = {int(cts[4]):,} ({round(m1[4],2)} ± {round(st1[4],2)})"],
        frameon=False, fontsize=20, title="# trials", title_fontsize=20, bbox_to_anchor=(1.05,1))

    plt.tight_layout()


def reindex_df(df, repeat_col):
    """ Expands dataframe for resampling.

    Repeats rows in the DataFrame based on the values in the specified column

    :param df (DataFrame): input DataFrame
    :param repeat_col (str): DataFrame column containing the count of repetitions for each row 

    :return expanded DataFrame with one row per count per sample

    """

    if repeat_col not in df.columns:
        raise ValueError(f"Column {repeat_col} not found in the DataFrame")

    if not df[repeat_col].dtype.kind in "iu": ## check if column contains non-negative integers
        raise ValueError(f"Column {repeat_col} must contain non-negative integers")
    
    expanded_df = df.reindex(df.index.repeat(df[repeat_col]))
    expanded_df.reset_index(drop=True, inplace=True)

    return expanded_df.copy() ## return a copy of the DataFrame to avoid modifying the original 


def set_figure_style(font_family="Arial", tick_label_size=24, axes_linewidth=2):
    """Prettify plots.

    Set default styling options for matplotlib figures.

    :param font_family (str): font family for sans-serif font (default is "Arial")
    :param tick_label_size (float): tick label sizes (default is 24)
    :param axes_linewidth (float): axes linewidth (default is 2)

    """

    plt.rcParams['font.sans-serif'] = font_family
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['xtick.labelsize'] = tick_label_size
    plt.rcParams['ytick.labelsize'] = tick_label_size
    plt.rcParams["axes.linewidth"] = axes_linewidth

def prepare_data_for_boxplot(x_data, x_labels, trial_counts):
    """Prepare data for boxplot.

    Stratify data by #extrema in wheel data for boxplot.

    :param x_data (list of array-like): data arrays for each group
    :param x_labels (list of str): labels for each group
    :param trial_counts (list): #trials for each group

    :return data (list): data values with corresponding group labels

    """

    ## prepare dataframe from inputs
    dfs = [pd.DataFrame(data, columns=[label]) for data, label in zip(x_data, x_labels)] 
    counts_df = pd.DataFrame(trial_counts, columns=["Count"])
    overall_df = pd.concat(dfs + [counts_df], axis=1)
    
    return overall_df.dropna() ## drop rows with NaN values


def add_statistical_annotations(ax, pairs, overall_df, order, statistical_test):
    """ Perform statistical analysis

    Add statistical annotations to plot

    :param ax (Axes): Axes object
    :param pairs (list of tuple): group label pairs for statistical testing
    :param overall_df (DataFrame): data as DataFrame
    :param order (list): group label order on x-axis

    Raises:
        Exception: if any error occurs during annotation 

    """

    try:
        annotator = Annotator(ax, pairs, data=overall_df, order=order)
        annotator.configure(test = statistical_test, text_format="star", loc="outside", line_width="2", fontsize=20)
        annotator.apply_and_annotate()
    except Exception as e:
        raise e

def save_plot(directory, filename):
    """
    Save plot

    :param directory (str): directory path where plot should be saved
    :param filename (str): file name to save plot as

    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    ## save plot
    plt.savefig(Path(directory) / filename, dpi=300)
    print("boxplot image saved")
    plt.close()


def plot_boxplot(data, tstring, xstring, ystring, order, statistical_test, figure_size=(10,8)):
    """Generate boxplot

    Plot data by #extrema in wheel data as boxplot

    :param data (list): data values
    :param tstring (str): title
    :param xstring (str): x-axis label
    :param ystring (str): y-axis label
    :param figure_size (tuple, optional): figure size (width, height)

    """
    
    data_weighted = reindex_df(data, "Count") ## perform weighted average of data based on #trials
    
    ## plot
    plt.figure(figsize=figure_size)
    set_figure_style()
    sns.boxplot(data=data_weighted, order=order, linewidth=3, showfliers=False)
    sns.stripplot(data=data, order=order, jitter=True, edgecolor="gray", linewidth=1)
    plt.ylabel(ystring, fontsize=28)
    plt.xlabel(xstring, fontsize=28)
    plt.text(0, 0.05, tstring, fontsize=28, ha="left", fontweight="bold") ## put title as text to prevent overlap with stat annotations
    ## stats
    ax = plt.gca()
    midpt = int(len(order) / 2)
    pairs = [(order[0], order[-1]), (order[0], order[midpt]), (order[midpt], order[-1])]
    try:
        #add_statistical_annotations(ax, pairs, data_weighted, order, statistical_test)
        plt.ylim([0,1])
    except:
        print("data all zeros")
        pass

    sns.despine(trim=False, offset=8)
    plt.tight_layout()


def plot_four_boxplot_grps(n_trials, x_data, x_labels, tstr, xstr, ystr, yname, directory, statistical_test, figure_size=(10, 8)):
    """Plot data stratified by #extrema

    Generate a boxplot comparing data groups by #extrema

    :param x_data (list of array-like): data arrays per group
    :param x_labels (list of str): labels per group
    :param tstr (str): title
    :param xstr (str): x-axis label
    :param ystr (str): y-axis label
    :param yname (str): variable of interest
    :param directory (str): directory to store plot
    :param statistical_test (str): statistical test name (i.e. "Wilcoxon")
    :param figure_size (tuple, optional): figure size (width, height)

    """
    
    data = prepare_data_for_boxplot(x_data, x_labels, n_trials)
    plot_boxplot(data, tstr, xstr, ystr, x_labels, statistical_test, figure_size=(10,8))
    #save_plot(directory, f"{tstr}_{yname}.png")


def plot_glm_hmm_engagement(group, data_path, cstring):
    """Plot proportion engaged per mouse 

    :param group (str): wiggler group ["good", "neutral", "bad"]
    :param data_path (str): data path to store files
    :param cstring (str): color hex code

    """

    df = pd.read_csv(Path(data_path, f"glm_hmm_analysis.csv"))
    grp_data = df.query(f"wiggler_group == '{group}'")
    mean_K_int = grp_data["mean_K"].values.tolist()
    median_K_int = grp_data["median_K"].values.tolist()
    Pengaged_int = grp_data["Pengage_normal"].values.tolist()

    svfg = plt.figure(figsize=(8,8))
    set_figure_style()
    plt.scatter(mean_K_int, Pengaged_int, 20, c=cstring)
    xdata = np.array(mean_K_int)
    ydata = np.array(Pengaged_int)

    try:
        a, b = np.polyfit(xdata, ydata, 1)
        plt.plot(xdata, xdata*a + b, c=cstring, lw=2)
        r, pval = pearsonr(xdata, ydata)
    except Exception as e:
        print("not enough data to do line of best fit")
        r = np.nan 
        pval = np.nan

    plt.xlabel("mean # of changes in wheel direction", fontsize=28)
    plt.ylabel("P(engaged and normal)", fontsize=28)
    plt.title(f"N = {len(grp_data)} mice, r = {round(r,2)} and p = {round(pval,2)}", fontsize=28, fontweight="bold")
    plt.tight_layout()
    plt.ylim(bottom=0)
    sns.despine(trim=False, offset=8)
    plt.show()


