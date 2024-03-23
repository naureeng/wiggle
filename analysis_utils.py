## functions to compute variable of interest across eids per mouse

## import dependencies
import numpy as np
import pandas as pd

def compute_wiggle_var_by_grp(mouse, eids, contrast_value, yname):

    k0_grp = []; k1_grp = []; k2_grp = []; k3_grp = []; k4_grp = []; counts_grp = []

    for i in range(len(eids)):
        eid = eids[i]
        ## load wheel csv per eid
        df_eid = pd.read_csv(f"/nfs/gatsbystor/naureeng/{mouse}/{eid}/{eid}_wheelData.csv")
        df_eid["feedbackType"] = df_eid["feedbackType"].replace(-1,0) ## to compute average
        df_lo = df_eid.query(f"abs(contrast)=={contrast_value} and abs(goCueRT-stimOnRT)<=0.05") ## to query trials by contrast

        ## stratify data into groups by k
        ## k = #extrema
        df_k0 = df_lo.query("n_extrema == 0 and duration<=1")
        df_k1 = df_lo.query("n_extrema == 1 and duration<=1")
        df_k2 = df_lo.query("n_extrema == 2 and duration<=1")
        df_k3 = df_lo.query("n_extrema == 3 and duration<=1")
        df_k4 = df_lo.query("n_extrema >= 4 and duration<=1")

        ## check trial length
        n_trials = 1 ## minimum #trials per group
        if len(df_k0)>=n_trials and len(df_k1)>=n_trials and len(df_k2)>=n_trials and len(df_k3)>=n_trials and len(df_k4)>=n_trials:
            ## accuracy of each group
            grp_k0 = df_k0[f"{yname}"].mean()
            grp_k1 = df_k1[f"{yname}"].mean()
            grp_k2 = df_k2[f"{yname}"].mean()
            grp_k3 = df_k3[f"{yname}"].mean()
            grp_k4 = df_k4[f"{yname}"].mean()

            ## save each group
            k0_grp.append(grp_k0)
            k1_grp.append(grp_k1)
            k2_grp.append(grp_k2)
            k3_grp.append(grp_k3)
            k4_grp.append(grp_k4)
            counts_grp.append(len(df_lo))

    return k0_grp, k1_grp, k2_grp, k3_grp, k4_grp, counts_grp

