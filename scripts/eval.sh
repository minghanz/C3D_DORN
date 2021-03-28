#!/bin/sh
# CUDA_VISIBLE_DEVICES=1 \
python eval.py -c config/dorn_eval_kitti.yaml \
-p config/dorn_path_eval_kitti_cluster.yaml \
--vpath vis/2021_03_22_01_58_35/ \
--txtpath eval_results/2021_03_22_01_58_35-epoch-14.txt \
-r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_22_01_58_35/epoch-14.pth #--vent #--vpbin #--vrgb --vdepth --vnormal --vmask --save-pcl-pred --save-pcl-gt
# -r /home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_22_01_58_35/epoch-14.pth --vent #--vpbin #--vrgb --vdepth --vnormal --vmask --save-pcl-pred --save-pcl-gt