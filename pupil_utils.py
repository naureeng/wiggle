import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainbox.io.one import SessionLoader

def average_data_in_epoch(times, values, trials_df, align_event='stimOn_times', epoch=(-0.6, -0.1)):
    """
    Aggregate values in a given epoch relative to align_event for each trial. For trials for which the align_event
    is NaN or the epoch contains timestamps outside the times array, the value is set to NaN.

    Parameters
    ----------
    times: np.array
        Timestamps associated with values, assumed to be sorted
    values: np.array
        Data to be aggregated in epoch per trial, one value per timestamp
    trials_df: pd.DataFrame
        Dataframe with trials information
    align_event: str
        Event to align to, must be column in trials_df
    epoch: tuple
        Start and stop of time window to aggregate values in, relative to align_event in seconds


    Returns
    -------
    epoch_array: np.array
        Array of average values in epoch, one per trial in trials_df
    """

    # Make sure timestamps and values are arrays and of same size
    times = np.asarray(times)
    values = np.asarray(values)
    if not len(times) == len(values):
        raise ValueError(f'Inputs to times and values must be same length but are {len(times)} and {len(values)}')
    # Make sure times are sorted
    if not np.all(np.diff(times) >= 0):
        raise ValueError('Times must be sorted')
    # Get the events to align to and compute the ideal intervals for each trial
    events = trials_df[align_event].values
    intervals = np.c_[events + epoch[0], events + epoch[1]]
    # Make a mask to exclude trials were the event is nan, or interval starts before or ends after bin_times
    valid_trials = (~np.isnan(events)) & (intervals[:, 0] >= times[0]) & (intervals[:, 1] <= times[-1])
    # This is the first index to include to be sure to get values >= epoch[0]
    epoch_idx_start = np.searchsorted(times, intervals[valid_trials, 0], side='left')
    # This is the first index to exclude (NOT the last to include) to be sure to get values <= epoch[1]
    epoch_idx_stop = np.searchsorted(times, intervals[valid_trials, 1], side='right')
    # Create an array to fill in with the average epoch values for each trial
    epoch_array = np.full(events.shape, np.nan)
    epoch_array[valid_trials] = np.asarray(
        [np.nanmean(values[start:stop]) if ~np.all(np.isnan(values[start:stop])) else np.nan
         for start, stop in zip(epoch_idx_start, epoch_idx_stop)],
        dtype=float)

    return epoch_array

def extract_epoch_values(times, values, trials_df, align_event, epoch):
    """
    Extracts values in an epoch relative to an align_event for each trial.

    Parameters
    ----------
    times : array-like
        Array of time points corresponding to `values`.
    values : array-like
        Array of values sampled at `times`.
    trials_df : pd.DataFrame
        DataFrame with one row per trial containing the align_event timestamps.
    align_event : str
        Column name in `trials_df` to align to.
    epoch : tuple of float
        Time window relative to the align_event (e.g., (-0.2, 0.5)).

    Returns
    -------
    epoch_values_list : list of 1D np.arrays
        List where each entry contains the values in the epoch for that trial.
        Trials outside the valid range will have np.nan.
    """
    # Make sure timestamps and values are arrays and of same size
    times = np.asarray(times)
    values = np.asarray(values)
    if not len(times) == len(values):
        raise ValueError(f'Inputs to times and values must be same length but are {len(times)} and {len(values)}')
    if not np.all(np.diff(times) >= 0):
        raise ValueError('Times must be sorted')

    events = trials_df[align_event].values
    intervals = np.c_[events + epoch[0], events + epoch[1]]
    valid_trials = (~np.isnan(events)) & (intervals[:, 0] >= times[0]) & (intervals[:, 1] <= times[-1])

    epoch_idx_start = np.searchsorted(times, intervals[valid_trials, 0], side='left')
    epoch_idx_stop = np.searchsorted(times, intervals[valid_trials, 1], side='right')

    epoch_values_list = [np.nan] * len(events)  # list of np.nan placeholders

    # Fill in values for valid trials
    valid_trial_indices = np.where(valid_trials)[0]
    for i, (trial_idx, start, stop) in enumerate(zip(valid_trial_indices, epoch_idx_start, epoch_idx_stop)):
        if ~np.all(np.isnan(values[start:stop])):
            epoch_values_list[trial_idx] = values[start:stop]
        else:
            epoch_values_list[trial_idx] = np.nan  # or np.array([]), if you prefer

    return epoch_values_list

def prepare_pupil(one, session_id, time_window=(-0.6, -0.1), align_event='stimOn_times', camera='left'):
    ## code written by: matthew whiteaway

    # Load the trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    pupil_data = one.load_object(session_id, f'{camera}Camera', attribute=['lightningPose', 'times'])
    # Extract x-position values in the epoch per trial
    epochs_x = extract_epoch_values(
        pupil_data['times'],
        pupil_data['lightningPose']['pupil_top_r_x'].values,
        sl.trials,
        align_event=align_event,
        epoch=time_window
    )

    # Compute the midpoint between pupil top and bottom y-positions
    pupil_y_mid = (
        pupil_data['lightningPose'][['pupil_bottom_r_y', 'pupil_top_r_y']].sum(axis=1) / 2
    ).values

    # Compute pupil diameter
    pupil_diameter = (
        pupil_data['lightningPose']['pupil_bottom_r_y'] - pupil_data['lightningPose']['pupil_top_r_y']).values

    # Extract y-position values in the epoch per trial
    epochs_y = extract_epoch_values(
        pupil_data['times'],
        pupil_diameter,
        sl.trials,
        align_event='stimOn_times',
        epoch=time_window
    )

    # Option 1: Return as list of (x, y) arrays per trial
    xy_epochs = [
        np.c_[x, y] if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and len(x) == len(y) else np.nan
        for x, y in zip(epochs_x, epochs_y)
    ]

    return xy_epochs

