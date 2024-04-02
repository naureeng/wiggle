## import dependencies
import numpy as np
from one.api import ONE
from pathlib import Path
from prepare_wheelData_csv import *
one = ONE()

def curate_eids_mouse(subject_name, data_path):
    """Obtains sessions per mouse.

    Curates sessions for N = 1 mouse.

    Args:
        subject_name (str): mouse name
        data_path (str): path to data files

    Returns:
        eids_final (list): sessions with valid wheel data

    """

    eids = one.search(subject=subject_name, data="trials.table")

    eids_final = [] ## initialize 

    for eid in eids:
        try:
            processed_eid = prepare_wheel_data_single_csv(subject_name, eid, data_path)
            if processed_eid is not None:
                eids_final.append(processed_eid)
        except Exception as e:
            print(f"Error processing session {eid}: {str(e)}")
    
    np.save(Path(data_path) / subject_name / f"{subject_name}_eids_wheel.npy", eids_final)
    print(f"{subject_name}: {len(eids_final)} sessions with valid wheel data saved.")

    return eids_final

