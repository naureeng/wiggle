## reaction time analysis
## author: naoki hiratani

## import dependencies

import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt
import sys

from math import *
from one.api import ONE, alfio
from os import path

one = ONE()

## pretty up plots
plt.rc('font',family='Arial')


class WheelData():
    def __init__(wheel, eid):
        wheel_data = one.load_object(eid, 'wheel', collection='alf')
        wheel.position = wheel_data.position
        wheel.timestamps = wheel_data.timestamps

        if str(type(wheel.position)) == "<class 'pathlib.PosixPath'>" or \
            str(type(wheel.timestamps)) == "<class 'pathlib.PosixPath'>":
            wheel.data_error = True
        else:
            wheel_velocity = []; wheel_velocity.append(0.0);
            for widx in range( len(wheel.position)-1 ):
                wheel_velocity.append( (wheel.position[widx+1] - wheel.position[widx])/(wheel.timestamps[widx+1] - wheel.timestamps[widx]) )
                wheel.velocity = wheel_velocity

    def calc_trialwise_wheel(wheel, stimOn_times, feedback_times):
        ## divide the wheel information into trialwise format
        ## stimOn_time - pre_duration < t < feedback_time

        wheel.stimOn_pre_duration = 0.3 # [s]
        wheel.total_trial_count = len(stimOn_times)

        wheel.trial_position = []
        wheel.trial_timestamps = []
        wheel.trial_velocity = []
        for tridx in range( wheel.total_trial_count ):
            wheel.trial_position.append([])
            wheel.trial_timestamps.append([])
            wheel.trial_velocity.append([])

        tridx = 0
        for tsidx in range( len(wheel.timestamps) ):
            timestamp = wheel.timestamps[tsidx]

            while tridx < len(stimOn_times) - 1 and timestamp > stimOn_times[tridx+1] - wheel.stimOn_pre_duration:
                tridx += 1
            if stimOn_times[tridx] - wheel.stimOn_pre_duration <= timestamp and timestamp < feedback_times[tridx]:
                wheel.trial_position[tridx].append( wheel.position[tsidx] )
                wheel.trial_timestamps[tridx].append( wheel.timestamps[tsidx] )
                wheel.trial_velocity[tridx].append( wheel.velocity[tsidx] )
        
    def calc_movement_onset_times(wheel, stimOn_times):
        #a collection of timestamps with a significant speed (>0.5) after more than 50ms of stationary period
        speed_threshold = 0.5
        duration_threshold = 0.05 #[s]

        wheel.movement_onset_times = []
        wheel.first_movement_onset_times = np.zeros( (wheel.total_trial_count) ) #FMOT
        wheel.last_movement_onset_times = np.zeros( (wheel.total_trial_count) ) #LMOT
        wheel.movement_onset_counts = np.zeros( (wheel.total_trial_count) )

        wheel.movement_directions = []
        wheel.first_movement_directions = np.zeros( (wheel.total_trial_count) )
        wheel.last_movement_directions = np.zeros( (wheel.total_trial_count) )

        for tridx in range(len(wheel.trial_timestamps)):
            wheel.movement_onset_times.append([])
            wheel.movement_directions.append([])
            cm_dur = 0.0; #continous stationary duration
            for tpidx in range( len(wheel.trial_timestamps[tridx]) ):
                t = wheel.trial_timestamps[tridx][tpidx];
                if tpidx == 0:
                    tprev = stimOn_times[tridx] - wheel.stimOn_pre_duration
                cm_dur += (t - tprev)
                if abs(wheel.trial_velocity[tridx][tpidx]) > speed_threshold:
                    if cm_dur > duration_threshold: #and t > stimOn_times[tridx]:
                        wheel.movement_onset_times[tridx].append( t )
                        wheel.movement_directions[tridx].append( np.sign(wheel.trial_velocity[tridx][tpidx]) )
                    cm_dur = 0.0;
                tprev = t
            wheel.movement_onset_counts[tridx] = len(wheel.movement_onset_times[tridx])
            if len(wheel.movement_onset_times[tridx]) == 0: #trials with no explicit movement onset
                wheel.first_movement_onset_times[tridx] = np.NaN
                wheel.last_movement_onset_times[tridx] = np.NaN
                wheel.first_movement_directions[tridx] = 0
                wheel.last_movement_directions[tridx] = 0
            else:
                wheel.first_movement_onset_times[tridx] = wheel.movement_onset_times[tridx][0]
                wheel.last_movement_onset_times[tridx] = wheel.movement_onset_times[tridx][-1]
                wheel.first_movement_directions[tridx] = wheel.movement_directions[tridx][0]
                wheel.last_movement_directions[tridx] = wheel.movement_directions[tridx][-1]

def compute_RTs(eid, trials):
    goCueRTs = []; stimOnRTs = []
    wheel = WheelData(eid)
    wheel.calc_trialwise_wheel(trials.stimOn_times, trials.feedback_times)
    wheel.calc_movement_onset_times(trials.stimOn_times)
    durations = trials.response_times - trials.stimOn_times

    for rtidx in range( len(wheel.first_movement_onset_times) ):
        goCueRTs.append(wheel.first_movement_onset_times[rtidx] - trials.goCue_times[rtidx])
        stimOnRTs.append(wheel.first_movement_onset_times[rtidx] - trials.stimOn_times[rtidx])

    return goCueRTs, stimOnRTs, durations, wheel 
