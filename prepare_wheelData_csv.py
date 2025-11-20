## import dependencies
import numpy as np
import pandas as pd
import os
from RT_dist_plot import *
import pickle
from wheel_utils import *
from pathlib import Path
from plot_utils import set_figure_style

def compute_wheel_data(wheel, n_trials, eid, data_path, subject_name):
    """ Computes wheel data

    Outputs time and position data for N = 1 session, 1 mouse
    
    :param wheel (user-defined object): output of RT_dist_plot
    :param n_trials (int): #trials
    :param eid (str): session
    :param data_path (str): path to data files
    :param subject_name (str): mouse name

    :return t_eid (list): time data [sec]
    :return p_eid (list): position data [deg]
    :return motionOnset_eid_idx (int): index for motionOnset in t_eid and p_eid
    :return stimOnset_eid_idx (int): index for stimOnset in t_eid and p_eid
    :return responseTime_eid_idx (int): index for responseTime in t_eid and p_eid

    """
    
    path = Path(data_path) / subject_name / f"{eid}/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
       print("The new directory is created!")

    trials = one.load_object(eid, 'trials')

    fs = 1000 ## (units = [Hz] sampling rate)
    t_eid = []; p_eid = []; v_eid = []; motionOnset_eid_idx = []; stimOnset_eid_idx = []; goCueOnset_eid_idx = []; responseTime_eid_idx = []
    for i in range(n_trials):
        try:
            trial_time = abs(wheel.trial_timestamps[i][-1] - wheel.trial_timestamps[i][0])
            t_resampled = np.linspace(wheel.trial_timestamps[i][0], wheel.trial_timestamps[i][-1], num=int(trial_time*fs))
            p_resampled = np.rad2deg(np.interp(t_resampled, wheel.trial_timestamps[i], wheel.trial_position[i]))


            trial_start = wheel.trial_timestamps[i][0]
            trial_end = wheel.trial_timestamps[i][-1]

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

        except:
            ## save zeros in trials with issues
            t_eid.append(0); p_eid.append(0); v_eid.append(0);
            motionOnset_eid_idx.append(0); stimOnset_eid_idx.append(0); goCueOnset_eid_idx.append(0); responseTime_eid_idx.append(0)

    ## pickle wheel data
    f = open(Path(data_path) / subject_name / f"{eid}/{eid}.wheel", 'wb')
    data = [t_eid, p_eid, v_eid, motionOnset_eid_idx, stimOnset_eid_idx, goCueOnset_eid_idx, responseTime_eid_idx]
    pickle.dump(data, f)
    f.close()
    print("wheel data pickled")

    return t_eid, p_eid, motionOnset_eid_idx, stimOnset_eid_idx, responseTime_eid_idx


def prepare_wheel_data_single_csv(subject_name, eid, data_path, time_window):
    """Prepare wheel csv 

    Outputs wheel csv for N = 1 session, 1 mouse

    Args:
        subject_name (str): mouse name
        eid (str): session
        data_path (str): directory to save files
        time_window (str): time window of analysis 
            There are three arguments for time_window: pre_stim, post_stim, total
            (1) pre_stim: stimulus onset - trial start
            (2) post_stim: trial end - stimulus onset
            (3) total: trial end - trial start

    Returns:
        eid (str): session (if csv made)

    """
    df = pd.DataFrame([], columns=["eid", "trial_no", "contrast", "block", "goCueRT", "stimOnRT", "duration", "choice", "feedbackType", "first_wheel_move", "last_wheel_move", "n_extrema", "rms", "speed", "n_changes", "motion_energy", "motion_energy_norm", "extrema_heights", "jitter_duration"])

    trials = one.load_object(eid, 'trials')
    goCueRTs, stimOnRTs, durations, wheel = compute_RTs(eid, trials)
    n_trials = len(trials.stimOn_times)
    print(n_trials, "trials")
    print(len(wheel.movement_directions), "movement directions")
    
    ## include sessions with complete wheel data  
    if len(wheel.movement_directions)==n_trials: 
        ## create the directory if it does not exist
        Path(data_path).mkdir(parents=True, exist_ok=True)

        ## build contrast array
        trials.contrast = np.empty(n_trials)
        contrastRight_idx = np.where(~np.isnan(trials.contrastRight))[0]
        contrastLeft_idx = np.where(~np.isnan(trials.contrastLeft))[0]
        trials.contrast[contrastRight_idx] = trials.contrastRight[contrastRight_idx]
        trials.contrast[contrastLeft_idx] = -1 * trials.contrastLeft[contrastLeft_idx]

        ## compute wheel statistics 
        t_eid, p_eid, motionOnset_eid_idx, stimOnset_eid_idx, responseTime_eid_idx = compute_wheel_data(wheel, n_trials, eid, data_path, subject_name)

        ## wheel analysis in time_window
        n_extrema, _, high_freq_powers = compute_n_extrema(t_eid, p_eid, n_trials, stimOnset_eid_idx, time_window) ## compute k 
        rms, _ = compute_rms(t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx, n_trials) ## compute rms
        speed, ballistic, motion_energy, extrema_heights, jitter_duration = compute_speed(t_eid, p_eid, motionOnset_eid_idx, responseTime_eid_idx, n_trials, n_extrema, data_path, subject_name, eid) ## compute speed
        n_changes, _ = compute_n_changes(t_eid, p_eid, n_trials, stimOnset_eid_idx, time_window, velocity_thresh=1e-4)

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
        #df["first_wheel_move"] = [wheel.movement_directions[i][0] for i in range(n_trials)]
        #df["last_wheel_move"] = [wheel.movement_directions[i][-1] for i in range(n_trials)]
        df["n_extrema"] = n_extrema
        df["rms"] = rms
        df["speed"] = speed
        df["ballistic"] = ballistic
        df["n_changes"] = n_changes 
        df["motion_energy"] = motion_energy_norm
        df["extrema_heights"] = extrema_heights
        df["jitter_duration"] = jitter_duration

        ## save df as csv
        csv_path = Path(data_path) / subject_name / f"{eid}/{eid}_wheelData_{time_window}.csv"
        df.to_csv(csv_path, index=False)
        print(f"{str(csv_path)} saved")
        return eid
    else:
        return None

def plot_wheel_trajectories(eid, data_path, subject_name):
    """

    Load wheel .pickle and plot each wheel trace separately as .svg files

    Args:
        :param eid (str): session
        :param data_path (str): path to data files
        :param subject_name (str): mouse name

    """

    ## load wheel data 
    wheel_path = Path(data_path) / subject_name / eid / f"{eid}.wheel"

    # Load wheel data
    try:
        with open(wheel_path, 'rb') as f:
            t_eid, p_eid, v_eid, motionOnset_idx, stimOnset_idx, goCueOnset_idx, responseTime_idx = pickle.load(f)
            print(f"{eid}: wheel data loaded")
    except Exception as e:
        print(f"Error loading wheel pickle for {eid}: {str(e)}")
        return

    # Create output directory
    output_dir = Path(data_path) / subject_name / eid / "trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trials = len(t_eid)
    n_extrema, ext_idx, _ = compute_n_extrema(t_eid, p_eid, n_trials, stimOnset_idx, "total") ## compute k 

    for i in range(n_trials):
        t = t_eid[i]
        p = p_eid[i]

        # plot wheel trace
        plt.figure(figsize=(10,8))
        plt.plot(t, p, lw=2)
        plt.xlabel("time [sec]", fontsize=24)
        plt.ylabel("wheel position [deg]", fontsize=24)
        set_figure_style(font_family="Arial", tick_label_size=24, axes_linewidth=2)

        # plot trial timepoints
        try:
            plt.axvline(x=t[motionOnset_idx[i]], c="c", label="motion onset")
            plt.axvline(x=t[stimOnset_idx[i]], c="r", label="stimulus onset")
            plt.axvline(x=t[goCueOnset_idx[i]], c="r", ls="--", label="go-cue onset")
        except:
            print(f"trial_{i} has been cut-off: indexing issues")

        ## plot extrema
        extrema = ext_idx[i]
        [plt.axvline(x=t[extrema[j]], c="k", ls="--", label=f"extremum #{j}") for j in range(len(extrema))]

        ## prettify plot
        plt.legend(fontsize=12)
        plt.xlim(left=0)
        sns.despine(trim=False, offset=4)
        plt.tight_layout()

        # save wheel img
        out_path = output_dir / f"trial_{i}.svg"
        plt.savefig(out_path, dpi=300)
        print(f"{eid}: trial #{i} wheel img saved")
        plt.close("all")

    print(f"Saved {n_trials} wheel traces to {output_dir}")

