## master script for individual mouse analysis
from plot_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle
from numpy import nan

## import subdirectories
pth_dir = "/nfs/gatsbystor/naureeng/"
sys.path.append(Path(pth_dir, "wiggle/preprocess/"))
sys.path.append(Path(pth_dir, "wiggle/plotting/"))
from preprocess.build_analysis_per_mouse import *
from preprocess.integrate_GLM_HMM import *
from plotting.plot_analysis_per_mouse import *

#pos_mouse_names = ['CSHL_007', 'CSHL_008', 'CSHL_010', 'CSHL_015', 'CSH_ZAD_001', 'CSH_ZAD_002', 'CSH_ZAD_003', 'CSH_ZAD_006', 'CSH_ZAD_019', 'DY_001', 'DY_002', 'DY_003', 'DY_010', 'DY_011', 'IBL-T1', 'IBL-T2', 'IBL-T3', 'IBL-T4', 'IBL_002', 'IBL_10', 'IBL_13', 'IBL_34', 'IBL_47', 'KM_005', 'KS003', 'KS004', 'KS005', 'KS016', 'KS022', 'KS023', 'KS024', 'KS051', 'KS055', 'KS074', 'KS079', 'KS083', 'KS084', 'KS085', 'KS086', 'KS089', 'KS092', 'KS093', 'KS095', 'KS096', 'MFD_05', 'MFD_07', 'NR_0005', 'NR_0008', 'NR_0009', 'NR_0011', 'NR_0012', 'NR_0014', 'NR_0017', 'NR_0018', 'NR_0020', 'NR_0021', 'NR_0023', 'NR_0027', 'NR_0028', 'NR_0029', 'NR_0031', 'NYU-01', 'NYU-02', 'NYU-04', 'NYU-06', 'NYU-23', 'NYU-25', 'NYU-38', 'NYU-39', 'NYU-41', 'NYU-44', 'NYU-47', 'NYU-48', 'NYU-49', 'NYU-52', 'NYU-57', 'NYU-58', 'PL011', 'PL016', 'PL017', 'PL019', 'PL022', 'PL023', 'PL024', 'PL026', 'PL029', 'PL034', 'PL035', 'PL050', 'SH015', 'SH024', 'SWC_013', 'SWC_014', 'SWC_021', 'SWC_023', 'SWC_038', 'SWC_NM_024', 'SWC_NM_026', 'SWC_NM_031', 'SWC_NM_036', 'SWC_NM_049', 'UCLA005', 'UCLA006', 'UCLA008', 'UCLA009', 'UCLA011', 'UCLA013', 'UCLA014', 'UCLA015', 'UCLA016', 'UCLA020', 'UCLA023', 'UCLA030', 'UCLA031', 'UCLA033', 'UCLA035', 'UCLA036', 'UCLA037', 'UCLA044', 'UCLA045', 'UCLA047', 'UCLA049', 'ZFM-01937', 'ZFM-01938', 'ZFM-02183', 'ZFM-02368', 'ZFM-02370', 'ZFM-02373', 'ZFM-02600', 'ZFM-03841', 'ZFM-04300', 'ZFM-04301', 'ZFM-04307', 'ZFM-05229', 'ZFM-05926', 'ZM_1084', 'ZM_1085', 'ZM_1086', 'ZM_1091', 'ZM_1092', 'ZM_1097', 'ZM_1367', 'ZM_1371', 'ZM_1372', 'ZM_1743', 'ZM_1745', 'ZM_1746', 'ZM_1895', 'ZM_1898', 'ZM_2240', 'ibl_witten_04', 'ibl_witten_05', 'ibl_witten_06', 'ibl_witten_12', 'ibl_witten_13', 'ibl_witten_14', 'ibl_witten_16', 'ibl_witten_26', 'ibl_witten_28', 'ibl_witten_29', 'ibl_witten_32', 'KS078', 'HB_005']

pos_mouse_names = ['CSHL_003', 'CSHL_014', 'CSH_ZAD_004', 'CSH_ZAD_007', 'CSH_ZAD_010', 'IBL_11', 'IBL_45', 'KS014', 'KS015', 'KS019', 'KS025', 'KS056', 'KS094', 'NR_0013', 'NR_0019', 'NYU-21', 'NYU-45', 'NYU-54', 'PL015', 'PL018', 'PL025', 'PL027', 'PL028', 'PL030', 'PL031', 'PL032', 'PL033', 'PL037', 'SWC_NM_022', 'SWC_NM_035', 'SWC_NM_043', 'UCLA004', 'UCLA007', 'UCLA010', 'UCLA017', 'UCLA019', 'UCLA022', 'UCLA034', 'UCLA048', 'UCLA050', 'ZFM-01936', 'ZFM-02184', 'ZFM-02369', 'ZFM-02372', 'ZFM-05233', 'ZM_1369', 'ZM_2107', 'ibl_witten_15', 'ibl_witten_30', 'ibl_witten_31']


def compute_K_data(subject_name):
    data = []
    for K in range(5):
        low_contrast_speed_K, low_contrast_accu_K = build_fix_K_speed_accu(subject_name, pth_dir, K)
        try:
            m = pearsonr(low_contrast_speed_K, low_contrast_accu_K).statistic
        except:
            m = np.nan
        data.append(m)

    return data

if __name__=="__main__":
    
    ## preprocess one mouse
    df = pd.read_csv(f"/nfs/gatsbystor/naureeng/final_eids/mouse_names.csv")
    mouse_names = df["mouse_names"].tolist()

    bwm_data = []
    for i in range(len(pos_mouse_names)):
        mouse = pos_mouse_names[i]
        data = compute_K_data(mouse)
        bwm_data.append(data)

final_data = pd.DataFrame(bwm_data)
plt.figure(figsize=(10,8))
set_figure_style()
sns.boxplot(data=final_data, linewidth=3)
sns.stripplot(data=final_data, jitter=True, edgecolor="gray", linewidth=3)
plt.ylabel("r", fontsize=28)
plt.xlabel("# of wheel direction changes", fontsize=28)
## stats
ax = plt.gca()
sns.despine(trim=False, offset=8)
plt.ylim([-1.05,1.05])
plt.tight_layout()
plt.show()

