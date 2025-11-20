import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nested_lookup import nested_lookup
#from prepare_wheelData_csv import *
from one.api import ONE
from pathlib import Path

one = ONE()

def prepare_pupil_csv():
    sessions = one.search(dataset='lightningPose', details=False)
    print(len(sessions), "sessions")
    df_pupil = pd.DataFrame(columns=["subject_name", "eid"])
    for i in range(len(sessions)):
        try:
            eid = sessions[i]
            session_info = one.alyx.rest('sessions', 'read', id=eid)
            subject_name = nested_lookup('subject', session_info)[0]
            data_path = "/nfs/gatsbystor/naureeng/"
            time_window = "post_stim"
            print(f"{subject_name}: {eid}")
            csv_path = Path(data_path, subject_name, eid, f"{eid}_wheelData_post_stim.csv")
            if os.path.exists(csv_path):
                print(f"csv made for {eid} in {subject_name}")
            else:
                prepare_wheel_data_single_csv(subject_name, eid, data_path, time_window)

            df_pupil.loc[i, "subject_name"] = subject_name
            df_pupil.loc[i, "eid"] = eid
        except Exception as e:
            print(f"failed to process session {eid} in {subject_name}")
            pass

    df_pupil.to_csv("/nfs/gatsbystor/naureeng/pupil_sessions.csv")
    print("pupil csvs done")

df_pupil = pd.read_csv("/nfs/gatsbystor/naureeng/pupil_sessions.csv")
for i in range(0,1): #range(len(df_pupil)):
    subject_name = df_pupil.loc[i, "subject_name"]
    eid = df_pupil.loc[i, "eid"]
    df_eid_path = f"/nfs/gatsbystor/naureeng/{subject_name}/{eid}/{eid}_wheelData_post_stim.csv"
    df_eid = pd.read_csv(df_eid_path)
    df_eid_wiggle = df_eid.query(
    "goCueRT > 0.08 and duration <= 1 and abs(goCueRT - stimOnRT) <= 0.05 and abs(contrast) == 0.0625 and n_extrema >= 4")
    print(len(df_eid_wiggle), "wiggles")
    print(df_eid_wiggle)




