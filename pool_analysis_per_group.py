## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from pathlib import Path

## obtain mouse names
pth_dir = '/nfs/gatsbystor/naureeng/' ## state path
pth_res = Path(pth_dir, 'results')
pth_res.mkdir(parents=True, exist_ok=True)
mouse_names = np.load(Path(pth_dir, "mouse_names.npy"), allow_pickle=True) ## load mouse_names

## compute slope across mice



slope = np.load(f"/nfs/gatsbystor/naureeng/slope.npy")
pos_slope = np.where(slope>0)[0]
print(len(pos_slope), "mice")

avg_bwm = []
for i in range(len(pos_slope)):
    subject_name = mouse_names[pos_slope[i]]
    avg_imagesc_mouse = np.load(f"/nfs/gatsbystor/naureeng/{subject_name}/{subject_name}_avg_prop_wiggle_90.npy")
    if len(avg_imagesc_mouse) !=0: ## take non-empty arrays
        avg_bwm.append(avg_imagesc_mouse)

avg_imagesc_bwm = np.mean(avg_bwm, axis=0)
plt.imshow(avg_imagesc_bwm, cmap="coolwarm")
plt.clim([1,3]) ## set color-scale
plt.show()

