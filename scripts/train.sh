#!/bin/bash
n_gpus=$1
flag=$2
path=$3
pflag=$4
ppath=$5

if [ $n_gpus -gt 1 ]
  then
    echo "running [python -m torch.distributed.launch --nproc_per_node=$n_gpus train.py $flag $path $pflag $ppath]"
    python -m torch.distributed.launch --nproc_per_node=$n_gpus train.py $flag $path $pflag $ppath
  else
    echo "running [python train.py $flag $path $pflag $ppath]"
    python train.py $flag $path $pflag $ppath
 fi