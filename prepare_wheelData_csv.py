## import dependencies
import numpy as np
import pandas as pd
import os
from RT_dist_plot import *
from scipy.signal import argrelmin, argrelmax
import pickle


def compute_wheel_data(wheel, n_trials):
    """
    COMPUTE_WHEEL_DATA outputs time and position data for N = 1 session, 1 mouse

    INPUTS
        [1] wheel: [user-defined object in RT_dist_plot]
        [2] n_trials: #trials [array] 
    OUTPUTS:
        [1] time data
        [2] position data

    """

    fs = 1000 ## (units = [Hz] sampling rate)
    t_eid = []; p_eid = []
    for i in range(n_trials):
        trial_time = abs(wheel.trial_timestamps[i][-1] - wheel.trial_timestamps[i][0])
        t_resampled = np.linspace(wheel.trial_timestamps[i][0], wheel.trial_timestamps[i][-1], num=int(trial_time*fs))
        p_resampled = np.rad2deg(np.interp(t_resampled, wheel.trial_timestamps[i], wheel.trial_position[i]))

        t_data = t_resampled - t_resampled[0]
        p_data = p_resampled - p_resampled[0]

        ## cut-off trials that exceed 0.3 radians
        stop_position = np.rad2deg(0.3)
        overshoot = np.argwhere(abs(p_data)>stop_position)
        if len(overshoot) != 0:
            stop_ind = np.squeeze(overshoot[0])
            t_final = t_data[0:stop_ind]
            p_final = p_data[0:stop_ind]
            print("overshoot trial")
        else:
            t_final = t_data
            p_final = p_data

        ## save wheel data
        t_eid.append(t_final); p_eid.append(p_final)

    return t_eid, p_eid

def compute_n_extrema(t_eid, p_eid, n_trials):
    """
    COMPUTE_N_EXTREMA computes #extrema for N = 1 session, 1 mouse

    INPUTS
        [1] t_eid: wheel time data (units: [seconds]) [list]
        [2] p_eid: wheel position data (units: [degrees]) [list]
        [3] n_trials: #trials [array]
    OUTPUTS:
        [1] n_extrema: #extrema [list]

    """
        
    ## compute extrema
    min_idx = [argrelmin(p_eid[i], order=1)[0] for i in range(n_trials)]
    max_idx = [argrelmax(p_eid[i], order=1)[0] for i in range(n_trials)]
    ext_idx = [np.concatenate((min_idx[i],max_idx[i]), axis=0) for i in range(n_trials)]
    n_extrema = [len(ext_idx[i]) for i in range(n_trials)]

    return n_extrema


def prepare_wheel_data_single_csv(mouse, eid):
    """
    PREPARE_WHEEL_DATA_SINGLE_CSV outputs csv for N = 1 session, 1 mouse

    INPUTS
        [1] mouse: mouse name [string]
        [2] eid: session name [string]
    OUTPUTS:
        [1] csv file: written to "/nfs/gatsbystor/naureeng/{mouse}/{eid}/{eid}_wheelData.csv" 
    
    """
    df = pd.DataFrame([], columns=["eid", "trial_no", "contrast", "block", "goCueRT", "stimOnRT", "duration", "choice", "feedbackType", "first_wheel_move", "last_wheel_move", "n_extrema", "rms"])

    ## obtain raw data
    trials = one.load_object(eid, 'trials')
    goCueRTs, stimOnRTs, durations, wheel = compute_RTs(eid, trials)
    n_trials = len(trials.stimOn_times)
    print(n_trials, "trials")
    
    ## include sessions with >=400 trials and complete wheel data  
    if n_trials >=400 and len(wheel.movement_directions)==n_trials: 
        path = f"/nfs/gatsbystor/naureeng/{mouse}/{eid}/trials/"
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
        
        ## build contrast array
        trials.contrast = np.empty(n_trials)
        contrastRight_idx = np.where(~np.isnan(trials.contrastRight))[0]
        contrastLeft_idx = np.where(~np.isnan(trials.contrastLeft))[0]
        trials.contrast[contrastRight_idx] = trials.contrastRight[contrastRight_idx]
        trials.contrast[contrastLeft_idx] = -1 * trials.contrastLeft[contrastLeft_idx]

        ## compute k 
        t_eid, p_eid = compute_wheel_data(wheel, n_trials)
        n_extrema = compute_n_extrema(t_eid, p_eid, n_trials)

        ## build df 
        df["eid"] = np.repeat(eid, n_trials)
        df["trial_no"] = np.arange(0, n_trials)
        df["contrast"] = trials.contrast
        df["block"] = trials.probabilityLeft
        df["goCueRT"] = goCueRTs
        df["stimOnRT"] = stimOnRTs
        df["duration"] = durations
        df["choice"] = trials.choice
        df["feedbackType"] = trials.feedbackType
        df["first_wheel_move"] = [wheel.movement_directions[i][0] for i in range(n_trials)]
        df["last_wheel_move"] = [wheel.movement_directions[i][-1] for i in range(n_trials)]
        df["n_extrema"] = n_extrema

        ## save df as sv
        df.to_csv(f"/nfs/gatsbystor/naureeng/{mouse}/{eid}/{eid}_wheelData.csv", index=False)
        print("csv saved")


