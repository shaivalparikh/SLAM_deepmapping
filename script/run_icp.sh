#!/bin/bash

# path to dataset
DATA_DIR=../data/2D/v1_pose0
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=icp_v1_pose0
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=point

# python incremental_icp.py --name $NAME -d $DATA_DIR -m $METRIC 
python o3d_icp.py --name $NAME -d $DATA_DIR -m $METRIC -r 0.00001
