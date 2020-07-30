#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python eval.py -c config/dorn_kitti.yaml -r /home/minghanz/DORN_pytorch/checkpoints/dorn_kitti/2020_07_26_11_51_42/epoch-30.pth
# CUDA_VISIBLE_DEVICES=1 python eval.py -c config/dorn_kitti_filled.yaml -r /home/minghanz/DORN_pytorch/checkpoints/dorn_kitti_filled/2020_07_23_11_44_09/epoch-30.pth
# CUDA_VISIBLE_DEVICES=1 python eval.py -c config/dorn_kitti_sparse.yaml -r /home/minghanz/DORN_pytorch/checkpoints/dorn_kitti_sparse/2020_07_23_11_44_56/epoch-35.pth