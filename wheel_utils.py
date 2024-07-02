## functions to compute wheel statistics

## import dependencies
import numpy as np
from scipy.signal import argrelmin, argrelmax
import math
from scipy.ndimage import gaussian_filter
from RT_dist_plot import *
from pathlib import Path
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from itertools import chain 

def find_nearest(array, value):
    """Obtains nearest value in array

    Computes the index where a value occurs in array
    author: michael schartner

    Args:
        array (np.array): multiple values
        value (int): search input

    Returns:
        idx (int): index in array where value occurs

    """

    idx = np.searchsorted( array, value, side="left" )
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]) ):
        return idx - 1
    else:
        return idx


def compute_n_extrema(t_eid, p_eid, n_trials):
    """Obtains #extrema
    
    Computes #extrema for N = 1 session, 1 mouse

    Args:
        t_eid (list): wheel time data [sec]
        p_eid (list): wheel position data [deg]
        n_trials (np.array): #trials

    Returns:
        n_extrema (list): #extrema

    """

    ## compute extrema 
    min_idx = [argrelmin(p_eid[i], order=1)[0] for i in range(n_trials)]
    max_idx = [argrelmax(p_eid[i], order=1)[0] for i in range(n_trials)]
    ext_idx = [np.concatenate((min_idx[i],max_idx[i]), axis=0) for i in range(n_trials)]
    n_extrema = [len(ext_idx[i]) for i in range(n_trials)]

    return n_extrema


def compute_rms(t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx, n_trials):
    """Obtains root mean square speed (RMS)

    Computes root mean square speed for N = 1 session, 1 mouse
    
    Args:
        t_eid (list): wheel time data [sec]
        p_eid (list): wheel position data [deg]
        motionOnset_eid_idx (int): motionOnset index
        responseTime_eid_idx (int): responseTime index
        n_trials (np.array): #trials

    Returns:
        rms_eid (list): root mean square speed [deg/sec]

    """

    ## analysis window = [motionOnset, responseTime]
    rms_eid = []
    for i in range(n_trials):

        ## obtain input data per trial
        time_trial = t_eid[i]
        pos_trial = p_eid[i]
        motionOnset_idx = motionOnset_eid_idx[i] 
        responseTime_idx = responseTime_eid_idx[i]
        
        ## compute times in seconds
        motionOnset = time_trial[int(motionOnset_idx)] ## (units: [sec])
        try:
            responseTime = time_trial[int(responseTime_idx)] ## (units: [sec])
        except:
            responseTime = time_trial[-1] ## use last time point for overshoot trials

        ## split data into 0.05 sec bins
        bins = np.arange(motionOnset, responseTime, 0.05) ## (units: [sec])
        idx_bins = [find_nearest(time_trial, i) for i in bins] ## indices of each bin

        n_bins = len(bins) ## normalize rms by #bins

        slope_trial_bin = []
        for j in range(len(bins)-1):
            start_time = bins[j]
            stop_time = bins[j+1]
            start_theta = pos_trial[idx_bins[j]]
            stop_theta = pos_trial[idx_bins[j+1]]
            slope = np.abs(stop_theta - start_theta)/(stop_time - start_time) ## compute slope magnitude in 0.05 sec bin
            slope_trial_bin.append(slope**2) ## compute squared slope magnitude

        rms = np.sqrt(sum(slope_trial_bin) / n_bins) ## normalize by #bins and take square root (units: [deg/sec]) 
        rms_eid.append(rms)

    return rms_eid


def compute_speed(t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx, n_trials, n_extrema):
    """Obtains speed and performs ballistic classification

    Computes wheel speed and classifies ballistic movements for N = 1 session, 1 mouse
    Computes wiggle amplitudes (wheel degrees) based on # changes in wheel direction
    
    Args:
        t_eid (list): wheel time data [sec]
        p_eid (list): wheel position data [deg]
        motionOnset_eid_idx (int): motionOnset index
        responseTime_eid_idx (int): responseTime index
        n_trials (np.array): #trials
        n_extrema (list): #extrema

    Returns:
        speed_eid (list): wheel speed [deg/sec]
        ballistic_eid (Boolean list): 0==False [non-ballistic] and 1==True [ballistic]

    """

    ## analysis window = [motionOnset, responseTime]
    speed_eid = []
    ballistic_eid = []
    wiggle_amplitude = {"K = 2": [], "K = 3": [], "K = 4": []} ## initialize empty dict for wiggle amplitudes (wheel degrees)

    for i in range(n_trials):

        ## align data to motionOnset
        t_motionOnset = t_eid[i] - t_eid[i][motionOnset_eid_idx[i]]
        p_motionOnset = p_eid[i] - p_eid[i][motionOnset_eid_idx[i]]
        motionOnset = motionOnset_eid_idx[i]
        k_trial = n_extrema[i]

        ## (1) obtain extrema
        min_idx = argrelmin(p_motionOnset, order=1)[0]
        max_idx = argrelmax(p_motionOnset, order=1)[0]
        ext_idx = np.concatenate((min_idx,max_idx), axis=0)

        ## (2) time vs position data of extrema
        t_idx = list([t_motionOnset[i] for i in ext_idx])
        p_idx = list([p_motionOnset[i] for i in ext_idx])

        t_idx.insert(0, 0.0) ## include motionOnset in time data
        p_idx.insert(0, 0.0) ## include motionOnset in position data

        t_extrema = np.sort(t_idx)
        idx = np.argsort(t_idx)
        p_extrema = [p_idx[i] for i in idx]

        ## compute duration
        duration = t_eid[i][-1] - t_eid[i][0]

        ## compute velocity
        v_final = np.diff(p_eid[i]) / np.diff(t_eid[i])
        v_smooth = gaussian_filter(v_final, sigma=3)
        max_v = max(abs(v_smooth))

        ## permit movement in opposite direction for a given time period after trial start
        neg_data_v = len( np.argwhere( v_smooth[20:-1] < 0 ))
        pos_data_v = len( np.argwhere( v_smooth[20:-1] > 0 ))

        ## classify ballistic
        if (duration<=1) and (neg_data_v==0):
            ballistic = 1
        elif (duration<=1) and (pos_data_v==0):
            ballistic = 1
        else:
            ballistic = 0

        if k_trial==0: 
            ## compute wheel speed for zero k trial
            print("zero extrema")
            speed = np.abs(p_motionOnset[-1] / t_motionOnset[-1]) ## wheel speed magnitude

        elif k_trial==1 and t_idx[0]<=0:
            ## trials with 1 extremum before motionOnset considered as zero k trial
            print("pre-motionOnset extremum")
            speed = np.abs(p_motionOnset[-1] / t_motionOnset[-1]) ## wheel speed magnitude

        else:
            ## trials with non-zero k are not ballistic
            ballistic = 0

            ## compute wiggle speed for non-zero k trial
            print("wiggle")
            T = np.abs(t_idx[-1])
            if T == 0:
                print("trial with motionOnset = extremum")
                T_wiggle = np.abs(t_extrema[-1])
            else:
                T_wiggle = T

            ## compute slope
            v_extrema = [np.abs(p_extrema[i+1]-p_extrema[i]) for i in range(len(p_extrema)-1)] 
            speed = sum(v_extrema) / T_wiggle

            ## store wiggle amplitudes as dict
            k_key = f"K = {min(k_trial, 4)}"
            #wiggle_amplitude[k_key].extend([p_idx[i] for i in idx])
            wiggle_amplitude[k_key].extend([speed])

        ## save data
        speed_eid.append(speed)
        ballistic_eid.append(ballistic)

    ## plot wiggle amplitudes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()  # Flatten the 2x2 array to make indexing easier

    for i, k in enumerate(range(2,5)):
        key = f"K = {k}"
        if key in wiggle_amplitude:
            myarray = wiggle_amplitude[key]
            weights = np.ones_like(myarray)/float(len(myarray))
            axs[i].hist(myarray, bins=35, edgecolor='black')
            axs[i].set_title(f'Histogram for {key}')
            axs[i].set_xlabel('Speed')
            axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return speed_eid, ballistic_eid


def get_extended_trial_windows(eid, time_lower, time_upper, movement_type):
    """
    GET_EXTENDED_TRIAL_WINDOWS creates dictionary for wheel movements in N = 1 session, 1 mouse 
    :param eid: session [string]
    :param time_lower: time prior motionOnset in [sec] [int]
    :param time_upper: time post motionOnset in [sec] [int]
    :return d: dictionary for wheel movements
    :return trial_issue: indices for excluded trials [list]
    """

    trials = one.load_object(eid, 'trials')
    wheel = WheelData(eid)
    wheel.calc_trialwise_wheel(trials.stimOn_times, trials.feedback_times)
    wheel.calc_movement_onset_times(trials.stimOn_times)

    ## compute contrast for trialData 
    trials.contrast = np.empty(len(trials.stimOn_times))
    contrastRight_idx = np.where(~np.isnan(trials.contrastRight))[0]
    contrastLeft_idx = np.where(~np.isnan(trials.contrastLeft))[0]

    trials.contrast[contrastRight_idx] = trials.contrastRight[contrastRight_idx]
    trials.contrast[contrastLeft_idx] = -1 * trials.contrastLeft[contrastLeft_idx]

    ## build dict
    d = {}
    trial_issue = []

    for tr in range(len(trials.stimOn_times)):
        a = wheel.first_movement_onset_times[tr]
        b = trials.goCue_times[tr]
        c = trials.feedback_times[tr]
        e = trials.contrast[tr]
        g = trials.stimOn_times[tr]
        m = trials.feedbackType[tr]

        if np.isnan(c) or np.isnan(b) or abs(g-b) >=0.05: # exclude trials with nan entries
            print(f"exclude trial {tr} in session {eid}")
            trial_issue.append(tr)
        else:
            react = np.round(a - b, 3)
            d[tr] = [a+time_lower, a+time_upper, a, react, e, g+time_lower, g+time_upper, m, g]

    return d, trial_issue


def plot_ballistic_movement(subject_name, data_path):
    """
    Assess ballistic code by visualizing trials per session for ballistic vs non-ballistic
    
    (de-bugging tool, will clean up)

    """ 
    eids = np.load(Path(data_path) / subject_name / f"{subject_name}_eids_wheel_test.npy")

    for i in range(len(eids)):
        eid = eids[i]
        ## load csv
        wh_csv  = pd.read_csv(Path(data_path) / subject_name / f"{eid}/{eid}_wheelData_test.csv")

        ## load wheel data
        f = open( Path(data_path) / subject_name / f"{eid}/{eid}.wheel", 'rb')
        wh_pkl = pickle.load(f)
        t_eid, p_eid, v_eid, motionOnset_eid, stimOnset_eid, goCueOnset_eid, feedbackOnset_eid = wh_pkl

        svfg = plt.figure(figsize=(8,8))
        df_ballistic = wh_csv.query("ballistic==1")
        df_non_ballistic = wh_csv.query("ballistic==0")

        ## plot ballistic vs non-ballistic trials for N = 1 session
        time_eid = []; pos_eid = []
        
        for i in range(len(wh_csv)):
            idx = wh_csv["trial_no"].iloc[i]
            t_motionOnset = t_eid[idx] - t_eid[idx][motionOnset_eid[idx]]
            p_motionOnset = p_eid[idx]

            if wh_csv["ballistic"].loc[idx]==1:
                plt.subplot(121)
                plt.plot(t_motionOnset, p_motionOnset)
            else:
                plt.subplot(122)
                plt.plot(t_motionOnset, p_motionOnset)

        plt.subplot(121); plt.title(f"N = {len(df_ballistic)} ballistic trials")
        plt.subplot(122); plt.title(f"N = {len(df_non_ballistic)} non-ballistic trials")

        plt.xlim([-0.25, 0.65])
        plt.ylim([-17,17])
        plt.xlabel("time aligned to motionOnset [sec]", fontsize=28)
        plt.ylabel("wheel position [deg]", fontsize=28)
        plt.rcParams['xtick.labelsize'] = 28
        plt.rcParams['ytick.labelsize'] = 28
        sns.despine(trim=False, offset=8)
        plt.tight_layout()

    plt.show()

