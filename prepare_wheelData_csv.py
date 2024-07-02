## import dependencies
import numpy as np
import pandas as pd
import os
from RT_dist_plot import *
import pickle
from wheel_utils import *
from pathlib import Path

def compute_wheel_data(wheel, n_trials, eid, data_path, subject_name):
    """ Computes wheel data

    Outputs time and position data for N = 1 session, 1 mouse
    
    Args:
        wheel (user-defined object): output of RT_dist_plot
        n_trials (int): #trials
        eid (str): session
        data_path (str): path to data files
        subject_name (str): mouse name
    Returns:
        t_eid (list): time data [sec]
        p_eid (list): position data [deg]
        motionOnset_eid_idx (int): index for motionOnset in t_eid and p_eid
        responseTime_eid_idx (int): index for responseTime in t_eid and p_eid

    """
    trials = one.load_object(eid, 'trials')

    fs = 1000 ## (units = [Hz] sampling rate)
    t_eid = []; p_eid = []; v_eid = []; motionOnset_eid_idx = []; stimOnset_eid_idx = []; goCueOnset_eid_idx = []; responseTime_eid_idx = []
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

        v_final = np.diff(p_final) / np.diff(t_final)
        
        ## compute motionOnset and responseTime
        motionOnset_idx = find_nearest(t_resampled, wheel.first_movement_onset_times[i]) ## find index in resampled data based on wheel data in seconds 
        stimOnset_idx = find_nearest(t_resampled, trials.stimOn_times[i])
        goCueOnset_idx = find_nearest(t_resampled, trials.goCue_times[i]) 
        responseTime_idx = find_nearest(t_resampled, trials.response_times[i])

        ## save wheel data
        t_eid.append(t_final); p_eid.append(p_final); v_eid.append(v_final);
        motionOnset_eid_idx.append(motionOnset_idx); stimOnset_eid_idx.append(stimOnset_idx); goCueOnset_eid_idx.append(goCueOnset_idx); responseTime_eid_idx.append(responseTime_idx)

    ## pickle wheel data
    with open( Path(data_path) / subject_name / f"{eid}/{eid}.wheel" , 'wb') as f:
        pickle.dump([t_eid, p_eid, v_eid, motionOnset_eid_idx, stimOnset_eid_idx, goCueOnset_eid_idx, responseTime_idx], f)

    return t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx


def prepare_wheel_data_single_csv(subject_name, eid, data_path):
    """Prepare wheel csv 

    Outputs wheel csv for N = 1 session, 1 mouse

    Args:
        subject_name (str): mouse name
        eid (str): session
        data_path (str): directory to save files

    Returns:
        eid (str): session (if csv made)

    """
    df = pd.DataFrame([], columns=["eid", "trial_no", "contrast", "block", "goCueRT", "stimOnRT", "duration", "choice", "feedbackType", "first_wheel_move", "last_wheel_move", "n_extrema", "rms", "speed"])

    ## obtain raw data
    trials = one.load_object(eid, 'trials')
    goCueRTs, stimOnRTs, durations, wheel = compute_RTs(eid, trials)
    n_trials = len(trials.stimOn_times)
    print(n_trials, "trials")
    
    ## include sessions with >=400 trials and complete wheel data  
    if n_trials >=400 and len(wheel.movement_directions)==n_trials: 
        ## create the directory if it does not exist
        Path(data_path).mkdir(parents=True, exist_ok=True)

        ## build contrast array
        trials.contrast = np.empty(n_trials)
        contrastRight_idx = np.where(~np.isnan(trials.contrastRight))[0]
        contrastLeft_idx = np.where(~np.isnan(trials.contrastLeft))[0]
        trials.contrast[contrastRight_idx] = trials.contrastRight[contrastRight_idx]
        trials.contrast[contrastLeft_idx] = -1 * trials.contrastLeft[contrastLeft_idx]

        ## compute wheel statistics 
        t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx = compute_wheel_data(wheel, n_trials, eid, data_path, subject_name)
        n_extrema = compute_n_extrema(t_eid, p_eid, n_trials) ## compute k 
        rms = compute_rms(t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx, n_trials) ## compute rms
        speed, ballistic = compute_speed(t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx, n_trials, n_extrema) ## compute speed

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
        df["rms"] = rms
        df["speed"] = speed
        df["ballistic"] = ballistic

        ## save df as sv
        csv_path = Path(data_path) / subject_name / f"{eid}/{eid}_wheelData_test.csv"
        df.to_csv(csv_path, index=False)
        print(f"{str(csv_path)} saved")
        return eid
    else:
        return None

