#!/bin/bash
#SBATCH --job-name DORN_C3D_TRAIN
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAILNE
#SBATCH --gpus-per-node=2
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
###SBATCH --mem-per-gpu=16g
#SBATCH --mem-per-cpu=3g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tp36dup
# CUDA_VISIBLE_DEVICES=0 
./train.sh 2 -c config/dorn_kitti_sparse.yaml
# python bts_test_dataloader.py arguments_train_eigen_c3d.txt