## curate data

## import dependencies
import numpy as np
from one.api import ONE
import os
from prepare_wheelData_csv import *
one = ONE()

def curate_eids_mouse(subject_name):
    """
    CURATE_EIDS_MOUSE outputs csv with wheelData "{eid}_wheelData.csv" for N = 1 session, 1 mouse

    :param subject_name: mouse name [string]
    :return eids_final: wheel data sessions [list]

    """

    eids = one.search(subject=subject_name, data="trials.table")
    for i in range(len(eids)):
        eid = eids[i]
        try:
            prepare_wheel_data_single_csv(subject_name, eid)
        except:
            print("eid issue with wheel data")

    csv_check = [os.path.exists(f"/nfs/gatsbystor/naureeng/{subject_name}/{eid}/{eid}_wheelData.csv") for eid in eids]
    idx_remove = [i for i in range(len(csv_check)) if csv_check[i]==False]
    eids_final = np.delete(eids, idx_remove)

    path = f"/nfs/gatsbystor/naureeng/{subject_name}/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
       print("The new directory is created!")

    np.save(f"/nfs/gatsbystor/naureeng/{subject_name}/{subject_name}_eids_wheel.npy", list(eids_final))
    print(f"{subject_name}: {len(eids_final)} sessions")
    print("eids saved")

    return eids_final

