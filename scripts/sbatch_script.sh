#!/bin/bash
#SBATCH --job-name DORN_C3D_TRAIN
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAILNE
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
###SBATCH --mem-per-gpu=16g
#SBATCH --mem-per-cpu=3g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tp36dup
# CUDA_VISIBLE_DEVICES=0 
./train.sh 2 -c config/dorn_kitti_sparse.yaml -p config/dorn_path_kitti_mct.yaml
# python bts_test_dataloader.py arguments_train_eigen_c3d.txt
# python -m torch.distributed.launch --nproc_per_node=2 train.py -c config/dorn_kitti_sparse.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/KittiRawLidar/dorn/2020_07_23_11_44_56/epoch-10.pth
# python train.py -c config/dorn_kitti_filled.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2020_07_23_11_44_09/epoch-10.pth
# python train.py -c config/dorn_kitti_sparse.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/KittiRawLidar/dorn/2020_07_23_11_44_56/epoch-10.pth
# ./train.sh 2 -c config/dorn_kitti_sparse.yaml
# ./train.sh 1 -c config/dorn_kitti_filled.yaml
# ./train.sh 1 -c config/dorn_kitti.yaml
# python bts_test_dataloader.py arguments_train_eigen_c3d.txt


# python eval.py -c config/dorn_kitti_sparse.yaml -r /home/minghanz/repos/DORN_pytorch/snap_dir/monodepth/KittiRawLidar/dorn/2020_07_21_17_58_07/epoch-13.pth

# python eval.py -c config/dorn_kitti_filled.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2020_07_23_11_44_09/epoch-40.pth
# python eval.py -c config/dorn_kitti_sparse.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/KittiRawLidar/dorn/2020_07_23_11_44_56/epoch-40.pth
# python eval.py -c config/dorn_kitti.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2020_07_26_11_51_42/epoch-40.pth
# python eval.py -c config/dorn_kitti_sparse.yaml -r /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/KittiRawLidar/dorn/2020_07_23_11_44_56/epoch-20.pth

python activate_all_files.py
