#!/bin/bash
#SBATCH --job-name DORN_C3D_TBOARD
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=hpeng1
#SBATCH --mail-type=END,FAILNE

#### SBATCH --partition=gpu
#### SBATCH --gpus-per-node=1
#### SBATCH --gpus=1
#### SBATCH --cpus-per-gpu=6
#### SBATCH --mem-per-gpu=16g

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
# conda activate tp36dup
conda activate pt14

### This is to set the port and print it
let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

tensorboard --logdir /scratch/hpeng_root/hpeng1/minghanz/tmp/DORN_pytorch/snap_dir/monodepth/vKitti2/dorn --port=$ipnport