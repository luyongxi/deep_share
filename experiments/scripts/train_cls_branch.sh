#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
model=$4
last_low_rank=$5
rounds=$6
sim_metric=$7
cls_id=$8

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_dynamic_branch_$7_$8.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_branch.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters $iters \
    --base_lr $base_lr \
    --clip_gradients 20 \
    --loss Sigmoid \
    --model $model \
    --snapshot_prefix $model-celeba-branch_$last_low_rank \
    --last_low_rank $last_low_rank \
    --use_svd \
    --exp low-vgg16-branch-$last_low_rank-$sim_metric-$cls_id \
    --num_rounds $rounds \
    --stepsize $stepsize \
    --sim_metric $sim_metric \
    --weights data/pretrained/gender.caffemodel \
    --share_basis \
    --cls_id $cls_id
