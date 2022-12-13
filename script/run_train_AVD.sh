#!/bin/bash

# path to dataset
DATA_DIR=../data/ActiveVisionDataset/Home_011_1/
# trajectiory file name
TRAJ=traj1
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=AVD_Home_011_1_${TRAJ}
# training epochs
EPOCH=5000
# batch size
BS=100
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=35
# logging interval
LOG=10 
#init 
INIT_POSE=/content/gdrive/MyDrive/SLAM_deepmapping/results/AVD/AVD_v1_pose0_point2point/pose_est.npy
### training from scratch
modelpath=/content/gdrive/MyDrive/SLAM_deepmapping/results/AVD/AVD_Home_011_1_traj1/model_best.pth
python train_AVD.py --name $NAME -d $DATA_DIR -t ${TRAJ}.txt -e $EPOCH -b $BS -l $LOSS -n $N -i $INIT_POSE --log_interval $LOG --model $modelpath
