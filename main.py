## master script 
## author: naureen ghani

## import dependencies
from one.api import ONE
from curate_eids import *

## example mouse
subject_name = "CSHL_008"

## [1] curate eids with csv files per mouse
#curate_eids_mouse(subject_name)
eids = np.load(f"/nfs/gatsbystor/naureeng/{subject_name}/{subject_name}_eids_wheel.npy")
print(f"{subject_name}: {len(eids)} wheel sessions")
