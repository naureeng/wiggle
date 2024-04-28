## integrate GLM-HMM results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def build_mouse_GLM_HMM_dict(subject_name, data_path):
    """Build 3-state GLM-HMM dictionary per mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files 

    """
    
    ## load data from JSON file
    with open(Path(data_path).joinpath(f"GLM-HMM/subject_results_dict.json")) as f:
        data = json.load(f)
    
    ## extract weight vectors for different states
    stim = data[f'{subject_name}']['3']['weight_vectors']['stim']
    pc = data[f'{subject_name}']['3']['weight_vectors']['pc']
    wsls = data[f'{subject_name}']['3']['weight_vectors']['wsls']
    bias = data[f'{subject_name}']['3']['weight_vectors']['bias']

    ## calculate absolute values of weight vectors for each state
    state_1 = [abs(stim[0]), abs(bias[0]), abs(pc[0]), abs(wsls[0])]
    state_2 = [abs(stim[1]), abs(bias[1]), abs(pc[1]), abs(wsls[1])]
    state_3 = [abs(stim[2]), abs(bias[2]), abs(pc[2]), abs(wsls[2])]

    ## classify states
    labels = ["state_1", "state_2", "state_3"]

    ## build dictionary of states
    states = {"state_1": state_1,
              "state_2": state_2,
              "state_3": state_3,
              "labels": labels
            }

    return states


def build_mouse_GLM_HMM_csv(subject_name, data_path):
    """Build 3-state GLM-HMM csv per mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files

    """

    ## load data
    df_glm = pd.read_parquet(Path(data_path, f"GLM-HMM/data_by_animal/{subject_name}_trials_table.pqt"))
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_wheel.npy"))
    glm_eids = []
   
    for eid in eids:
        try:
            df_eid = pd.read_csv(Path(data_path, subject_name, eid, f"{eid}_wheelData.csv"))

            df_glm_eid = df_glm.query(f"session=='{eid}'")

            ## save state probabilities
            P1 = [df_glm_eid["glm-hmm_3"].iloc[i][0] for i in range(len(df_glm_eid))]
            P2 = [df_glm_eid["glm-hmm_3"].iloc[i][1] for i in range(len(df_glm_eid))]
            P3 = [df_glm_eid["glm-hmm_3"].iloc[i][2] for i in range(len(df_glm_eid))]
            ## save state classification
            idx_max = [np.argmax(df_glm_eid["glm-hmm_3"].iloc[i]) for i in range(len(df_glm_eid))]

            ## store to dataframe
            df_eid["P_state_1"] = P1
            df_eid["P_state_2"] = P2
            df_eid["P_state_3"] = P3
            df_eid["idx_glm_hmm_3"] = idx_max

            states = build_mouse_GLM_HMM_dict(subject_name, data_path)
            labels = states["labels"]
            glm_state = [labels[i] for i in idx_max]
            df_eid["state_glm_hmm_3"] = glm_state

            ## save csv
            df_eid.to_csv(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv"))
            print(f"{eid} csv updated with glm-hmm")
            
            ## append eid
            glm_eids.append(eid)

        except Exception as e:
            print(f"{eid} could not be updated with glm-hmm")

    ## save eid list
    np.save(Path(data_path, subject_name, f"{subject_name}_eids_glm_hmm.npy"), glm_eids)
    print(f"{subject_name}: {len(glm_eids)} glm-hmm eids")
        

def build_wiggle_GLM_HMM_analysis(subject_name, data_path):
    """Build 3-D matrix of 3-state GLM-HMM probabilities per mouse

    Args:
        subject_name (str): mouse name
        data_path (str): path to store data files

    """

    ## load glm-hmm eids
    eids = np.load(Path(data_path, subject_name, f"{subject_name}_eids_glm_hmm.npy"))
    print(f"{subject_name}: {len(eids)} glm-hmm eids")

    ## obtain 3-D matrix across sessions per mouse
    points_mouse = []
    for eid in eids:
        df_eid = pd.read_csv(Path(data_path, subject_name, eid, f"{eid}_glm_hmm.csv"))
        points_eid = [(df_eid["P_state_1"].iloc[i], df_eid["P_state_2"].iloc[i], df_eid["P_state_3"].iloc[i]) for i in range(len(df_eid))]
        for i in points_eid: ## save each element
            points_mouse.append(i)
    
    ## save data
    np.save(Path(data_path).joinpath(f"{subject_name}/{subject_name}_points_mouse.npy"), points_mouse)

    return points_mouse

