## import dependencies
import numpy as np
from one.api import ONE
from pathlib import Path
from prepare_wheelData_csv import *
from ephys_utils import sessions_with_region, compute_maximum_drift 
one = ONE()

def curate_eids_mouse(subject_name, data_path):
    """Obtains sessions per mouse.

    Curates sessions for N = 1 mouse.

    :param subject_name (str): mouse name
    :param data_path (str): path to data files

    :return eids_final (list): sessions with valid wheel data

    """

    eids = one.search(subject=subject_name, data="trials.table")

    eids_final = [] ## initialize 

    #for eid in eids:
    for i in range(len(eids)):
        eid = eids[i]
        try:
            processed_eid = prepare_wheel_data_single_csv(subject_name, eid, data_path)
            if processed_eid is not None:
                eids_final.append(processed_eid)
        except Exception as e:
            print(f"Error processing session {eid}: {str(e)}")
    
    np.save(Path(data_path) / subject_name / f"{subject_name}_eids_wheel_test.npy", eids_final)
    #print(f"{subject_name}: {len(eids_final)} sessions with valid wheel data saved.")
    print(f"{subject_name}: {len(eids_final)} sessions for ballistic testing saved")

    return eids_final

def curate_eids_neural(reg, data_path):
    """Obtains sessions per region.

    Curates sessions for < 80 microns maximum drift.

    :param reg (str): brain region
    :param data_path (str): path to data files

    :return eids_final (list): sessions with valid neural data

    """
    eids, probes = sessions_with_region(reg, one=one)
    print(len(eids), "sessions")

    eids_final = [] ## initialize

    for i in range(len(eids)):
        ## obtain session and probe
        eid = eids[i]
        probe = probes[i]

        try:
            ## compute drift
            max_drift = compute_maximum_drift(eid, probe)
            if max_drift > 80:
                print(f"session {eid} exceeds drift of 80 microns: not used")
            else:
                eids_final.append(eid)
        except Exception as e:
            print(f"Error processing session {eid}: {str(e)}")

    np.save(Path(data_path) / f"{reg}" / f"{reg}_eids_neural.npy", eids_final)
    print(f"{reg}: {len(eids_final)} sessions with valid neural data saved.")

    return eids_final


