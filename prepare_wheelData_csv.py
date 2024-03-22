## import dependencies
import numpy as np
import pandas as pd
import os
from RT_dist_plot import *
from scipy.signal import argrelmin, argrelmax


def compute_n_extrema(wheel, n_trials):
    """
    COMPUTE_N_EXTREMA outputs #extrema for N = 1 session, 1 mouse

    INPUTS
        [1] wheel:  [string]
        [2] eid: session name [string]
    OUTPUTS:
        [1] csv file: written to "/nfs/gatsbystor/naureeng/{mouse}/{eid}/{eid}_wheelData.csv"

    """

    fs = 1000 ## units = [Hz] sampling rate
    n_extrema = []
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
        
        ## compute extrema
        min_idx = argrelmin(p_final, order=1)[0]
        max_idx = argrelmax(p_final, order=1)[0]
        ext_idx = np.concatenate((min_idx,max_idx), axis=0)
        n = len(ext_idx)
        n_extrema.append(n)

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
    df = pd.DataFrame([], columns=["eid", "trial_no", "contrast", "block", "goCueRT", "stimOnRT", "duration", "choice", "feedbackType", "first_wheel_move", "last_wheel_move", "n_extrema"])

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
        n_extrema = compute_n_extrema(wheel, n_trials)

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


