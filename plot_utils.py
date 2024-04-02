## functions to plot data per mouse

## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator


def reindex_df(df, repeat_col):
    """ Expands dataframe for resampling.

    Repeats rows in the DataFrame based on the values in the specified column

    Args:
        df (DataFrame): input DataFrame
        repeat_col (str): DataFrame column containing the count of repetitions for each row 

    Returns:
        expanded DataFrame with one row per count per sample

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

    Args:
        font_family (str): font family for sans-serif font (default is "Arial")
        tick_label_size (float): tick label sizes (default is 24)
        axes_linewidth (float): axes linewidth (default is 2)

    """

    plt.rcParams['font.sans-serif'] = font_family
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['xtick.labelsize'] = tick_label_size
    plt.rcParams['ytick.labelsize'] = tick_label_size
    plt.rcParams["axes.linewidth"] = axes_linewidth

def prepare_data_for_boxplot(x_data, x_labels, trial_counts):
    """Prepare data for boxplot.

    Stratify data by #extrema in wheel data for boxplot.

    Args:
        x_data (list of array-like): data arrays for each group
        x_labels (list of str): labels for each group

    Returns:
        data (list): data values with corresponding group labels

    """

    ## prepare dataframe from inputs
    dfs = [pd.DataFrame(data, columns=[label]) for data, label in zip(x_data, x_labels)] 
    counts_df = pd.DataFrame(trial_counts, columns=["Count"])
    overall_df = pd.concat(dfs + [counts_df], axis=1)

    return overall_df.dropna() ## drop rows with NaN values


def add_statistical_annotations(ax, pairs, overall_df, order, statistical_test):
    """ Perform statistical analysis

    Add statistical annotations to plot

    Args:
        ax (Axes): Axes object
        pairs (list of tuple): group label pairs for statistical testing
        overall_df (DataFrame): data as DataFrame
        order (list): group label order on x-axis

    Raises:
        Exception: if any error occurs during annotation 

    """

    try:
        annotator = Annotator(ax, pairs, data=overall_df, order=order)
        annotator.configure(test = statistical_test, text_format="star", loc="outside", line_width="4", fontsize=28)
        annotator.apply_and_annotate()
    except Exception as e:
        raise e

def save_plot(directory, filename):
    """
    Save plot

    Args:
        directory (str): directory path where plot should be saved
        filename (str): file name to save plot as

    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    ## save plot
    plt.savefig(Path(directory) / filename, dpi=300)
    print("boxplot image saved")
    plt.close()


def plot_boxplot(data, tstring, xstring, ystring, order, statistical_test, figure_size=(10,8)):
    """Generate boxplot

    Plot data by #extrema in wheel data as boxplot

    Args:
        data (list): data values
        tstring (str): title
        xstring (str): x-axis label
        ystring (str): y-axis label
        figure_size (tuple, optional): figure size (width, height)

    """
    
    data_weighted = reindex_df(data, "Count") ## perform weighted average of data based on #trials
    
    ## plot
    plt.figure(figsize=figure_size)
    set_figure_style()
    sns.boxplot(data=data_weighted, order=order, linewidth=3, showfliers=False)
    sns.stripplot(data=data, order=order, jitter=True, edgecolor="gray", linewidth=3)
    plt.ylabel(ystring, fontsize=28)
    plt.xlabel(xstring, fontsize=28)
    plt.text(0, 0.05, tstring, fontsize=28, ha="left", fontweight="bold") ## put title as text to prevent overlap with stat annotations
    ## stats
    ax = plt.gca()
    pairs = [(order[0], order[4]), (order[0], order[2]), (order[2], order[4])]
    try:
        add_statistical_annotations(ax, pairs, data_weighted, order, statistical_test)
    except:
        print("data all zeros")
        pass

    sns.despine(trim=False, offset=8)
    plt.ylim(bottom=0)
    plt.tight_layout()


def plot_four_boxplot_grps(n_trials, x_data, x_labels, tstr, xstr, ystr, yname, directory, statistical_test, figure_size=(10, 8)):
    """Plot data stratified by #extrema

    Generate a boxplot comparing data groups by #extrema

    Args:
        n_trials (int): #trials
        x_data (list of array-like): data arrays per group
        x_labels (list of str): labels per group
        tstr (str): title
        xstr (str): x-axis label
        ystr (str): y-axis label
        yname (str): variable of interest
        directory (str): directory to store plot
        statistical_test (str): statistical test name (i.e. "Wilcoxon")
        figure_size (tuple, optional): figure size (width, height)

    """
    
    data = prepare_data_for_boxplot(x_data, x_labels, n_trials)
    plot_boxplot(data, tstr, xstr, ystr, x_labels, statistical_test, figure_size=(10,8))
    save_plot(directory, f"{tstr}_{yname}.png")

