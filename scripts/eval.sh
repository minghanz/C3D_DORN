#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python eval.py -c config/dorn_eval_kitti.yaml \
-p config/dorn_path_eval_kitti_mct.yaml \
-r /home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_07_14_42_19/epoch-14.pth --vpbin #--vrgb --vdepth --vnormal --vmask --save-pcl-pred --save-pcl-gt