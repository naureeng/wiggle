import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import brainbox.plot as bbplot
import math
import sys
import logging

from one.api import ONE, alfio
from brainbox.metrics.electrode_drift import estimate_drift
from brainbox.io.one import SpikeSortingLoader, load_channel_locations
from brainbox.processing import bincount2D
from ibllib.atlas import AllenAtlas
from scipy.ndimage import gaussian_filter
from plot_utils import set_figure_style, build_legend
from pathlib import Path
from matplotlib.ticker import MaxNLocator

ba = AllenAtlas()
one = ONE()


def sessions_with_region(acronym, one=None):
    """
    SESSIONS_WITH_REGION gets sessions containing brain region of interest
    author: guido meijer

    :param acronym: brain region of interest [str]

    :return eids: sessions containing acronym [list of str]
    :return probes: probes containing acronym [list of str] "probe0" or "probe1" for a given session

    """

    if one is None:
        one = ONE()
    query_str = f'channels__brain_region__acronym__icontains,{acronym}' 
    traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                         django=query_str)
    eids = np.array([i['session']['id'] for i in traj])
    probes = np.array([i['probe_name'] for i in traj])
    return eids, probes


def compute_maximum_drift(eid, probe):
    """
    COMPUTE_MAXIMUM_DRIFT gets maximum electrode drift for quality-control in N = 1 session

    :param eid: session [str]
    :param probe: "probe0" or "probe1" [str]

    :return max_drift: maximum electrode drift in eid, probe pair [int]

    """

    ## get dataset
    spikes = one.load_object(eid, 'spikes', collection=f'alf/{probe}')

    ## compute maximum drift (Allen Institute uses up to 80 microns)
    drift = estimate_drift(spikes.times, spikes.amps, spikes.depths, display=False)
    max_drift = max(abs(drift[0]))
    print(f"maximum drift: {max_drift} microns")

    return max_drift

def find_nearest(array, value):
    """
    FIND_NEAREST obtains the index where a value occurs in array
    author: michael schartner

    :param array: matrix [np.arr]
    :param value: search input [int]

    :return idx: index in array where value occurs [int]

    """

    idx = np.searchsorted( array, value, side="left" )
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]) ):
        return idx - 1
    else:
        return idx

def pair_firing_rate_contrast(eid, probe, d, reg, time_lower, time_upper, T_BIN, alignment_type):
    """
    PAIR_FIRING_RATE_CONTRAST obtains the spike data for a session

    :param eid: session [string]
    :param probe: probe [string]
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :param T_BIN: bin size [sec] [int]
    :param alignment_type: "motionOnset" or "stimOnset" [string]

    :return Res: 3D matrix of spike data [trials x time x units] [np.array]
    :return num_units: num of units [int]
    :return histology: ids of Res in histology [pd.dataframe]

    """

    #
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, els = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, els)
    print('data acquired')

    R, times, Clusters = bincount2D( spikes['times'], spikes['clusters'], T_BIN ) # raw spikes

    Acronyms = els['acronym']
    # clusters = alfio.load_object( probe_path, 'clusters' )
    cluster_chans = clusters['channels'][Clusters]
    acronyms = Acronyms[cluster_chans]

    ind_reg = [idx for idx, s in enumerate(acronyms) if reg in s]
    mask = np.zeros( len(acronyms), dtype='bool' )
    for i in ind_reg:
        mask[i] = 1

    D = R.T # transpose data

    Res = []

    # obtain mean firing rate and contrast
    m_ask = mask
    num_units = np.squeeze(np.where( m_ask==True ))
    noisy_units = 0

    if len(num_units) < 10: # min req of 10 neurons
        logging.error('not enough units')
    else:
        histology = acronyms[i] #[acronyms[i] for i in num_units]
        n_units = len(num_units)
        print(n_units, "units")

    trial_idx = d.keys()

    for i in trial_idx:
        if alignment_type == "motionOnset":
            start_idx = find_nearest(times, d[i][0])
            end_idx = find_nearest(times, d[i][1])
        else:
            start_idx = find_nearest(times, d[i][5])
            end_idx = find_nearest(times, d[i][6])

        data =  D[start_idx : end_idx, m_ask]
        data_len = (abs(time_lower) + abs(time_upper)) / T_BIN
        if len(data) != data_len:
            if len(data) < data_len:
                n = int(data_len - len(data))
                r, c = data.shape
                position_pad = np.zeros((n,c))
                data_final = np.append(data, position_pad, axis=0)
            else:
                logging.warning('data long')
                data_final = data[0: int(data_len) ]
        else:
            data_final = data

        Res.append(data_final)

    ## sanity check:
    n_trials = len(d)
    n_bins = int((time_upper- time_lower)/ T_BIN)

    ## [neurons x time x trials]
    Res = np.reshape(np.transpose(Res), [n_units, n_bins, n_trials])
    print(f"{n_units} units, {n_bins} bins, {n_trials} trials")

    return Res


def obtain_psth(Res, time_lower, time_upper, T_BIN):
    """
    OBTAIN_PSTH gets trial-averaged spiking activity for each unit in N = 1 session

    :param Res: 3D matrix of spike data [units x time x trials] [np.arr]
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :param T_BIN: bin size [sec] [int]

    :return mu_data: trial-averaged mean spiking activity per unit [units x time] [np.arr]
    :return std_data: trial-averaged std spiking activity per unit [units x time] [np.arr]
    :return n_trials: #trials in N = 1 session [int]

    """

    ## mean firing rate over trials per unit
    [n_units, n_bins, n_trials] = Res.shape
    mean_firing_unit = [(np.nanmean(Res[i,:,:], axis=1) / T_BIN) for i in range(n_units)]
    mu_data = gaussian_filter(np.nanmean(mean_firing_unit, axis=0), sigma=3)
    std_data = gaussian_filter( (1.96*np.nanstd(mean_firing_unit, axis=0)) / np.sqrt(len(mean_firing_unit)), sigma=3)

    return mu_data, std_data, n_trials


def plot_psth(mu_data_K, std_data_K, n_trials_K, time_lower, time_upper, T_BIN, n_units, alignment_type, data_path, subject_name, eid):
    """
    PLOT_PSTH plots trial-averaged spiking activity for each unit in N = 1 session

    :param mu_data_K:  trial_averaged mean spiking activity in trials sorted by #changes in wheel direction [units x time] [np.arr]
    :param std_data_K: trial_averaged std spiking activity in trials sorted by #changes in wheel direction [units x time] [np.arr]
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :param T_BIN: bin size [sec] [int]
    :param n_units: #units in N = 1 session [int]
    :param alignment_type: "motionOnset" or "stimOnset" [str]
    :param data_path: path to data files [str]
    :param subject_name: region or mouse name [str]
    :param eid: session name [str]

    """

    cstring = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    ## plot
    svfg = plt.figure(figsize=(10,8))
    set_figure_style(font_family="Arial", tick_label_size=20, axes_linewidth=2)
    time_bins = np.arange(time_lower, time_upper, T_BIN)
    [plt.plot(time_bins, mu_data_K[i], lw=3, color=cstring[i]) for i in range(len(cstring))]
    [plt.fill_between(time_bins, mu_data_K[i]-std_data_K[i], mu_data_K[i]+std_data_K[i], alpha=0.2, color=cstring[i]) for i in range(len(cstring))]
    plt.ylim(bottom=0)
    
    ## labels
    plt.xlabel(f"time aligned to {alignment_type} [sec]", fontsize=24)
    plt.ylabel("mean firing rate [spikes/sec]", fontsize=24)
    plt.title(f"N = {n_units} mouse VISp units, 1 session", fontsize=24, fontweight="bold")
    plt.axvline(x=0, ls="--", lw=2, color="k")

    # set y-axis limit to the maximum count
    max_count = np.max([np.max(mu_data_K[i]) for i in range(len(cstring))])
    plt.ylim(0, np.ceil(max_count))

    # set the y-axis to use integer ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    ## build legend
    m1 = [np.nanmean(mu_data_K[i]) for i in range(len(cstring))]
    st1 = [np.nanmean(std_data_K[i]) for i in range(len(cstring))]
    build_legend(m1, st1, n_trials_K)

    ## despine
    sns.despine(trim=False, offset=8)
    plt.savefig(Path(data_path) / subject_name / f"{eid}/{eid}_psth_{alignment_type}.png", dpi=300)
    print(f"psth aligned to {alignment_type} for [{time_lower},{time_upper}] sec saved")


