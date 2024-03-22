## master script 
## author: naureen ghani

## import dependencies
from one.api import ONE
from prepare_wheelData_csv import *
one = ONE()

## example mouse
mouse = "CSHL_008"
eid = "15ba1fc3-bc8a-4c3c-99b6-e6cf9f12b447"

## [1] build wheel csv
prepare_wheel_data_single_csv(mouse, eid)
