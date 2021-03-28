#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python eval.py -c config/dorn_eval_kitti.yaml \
-p config/dorn_path_eval_kitti_mct.yaml \
--vpath vis/2021_03_11_16_07_48/ \
-r /home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_11_16_07_48/epoch-14.pth --vent #--vpbin #--vrgb --vdepth --vnormal --vmask --save-pcl-pred --save-pcl-gt