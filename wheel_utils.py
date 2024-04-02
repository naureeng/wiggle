## functions to compute wheel statistics

## import dependencies
import numpy as np
from scipy.signal import argrelmin, argrelmax
import math


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
    """Obtains speed

    Computes wheel speed for N = 1 session, 1 mouse
    
    Args:
        t_eid (list): wheel time data [sec]
        p_eid (list): wheel position data [deg]
        motionOnset_eid_idx (int): motionOnset index
        responseTime_eid_idx (int): responseTime index
        n_trials (np.array): #trials
        n_extrema (list): #extrema

    Returns:
        speed_eid (list): wheel speed [deg/sec]

    """

    ## analysis window = [motionOnset, responseTime]
    speed_eid = []
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

        if k_trial==0: 
            ## compute wheel speed for zero k trial
            print("zero extrema")
            speed = np.abs(p_motionOnset[-1] / t_motionOnset[-1]) ## wheel speed magnitude
        elif k_trial==1 and t_idx[0]<=0:
            ## trials with 1 extremum before motionOnset considered as zero k trial
            print("pre-motionOnset extremum")
            speed = np.abs(p_motionOnset[-1] / t_motionOnset[-1]) ## wheel speed magnitude
        else:
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

        ## save data
        speed_eid.append(speed)
    
    return speed_eid


