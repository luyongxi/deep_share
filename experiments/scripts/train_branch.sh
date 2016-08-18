#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
clip_gradients=$4
loss=$5
model=$6
first_low_rank=$7
rounds=$8

set -x
set -e

# TODO: change this script to something that runs train_branch.py

export PYTHONUNBUFFERED="True"

LOG="../logs/train_dynamic_branch_$5_loss_[model=$model,lr=$base_lr,first_low_rank=$first_low_rank].txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_branch.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters $iters \
    --base_lr $base_lr \
    --clip_gradients $clip_gradients \
    --loss $loss \
    --model $model \
    --snapshot_prefix $model-facial-attr-branch_$first_low_rank-$loss \
    --first_low_rank $first_low_rank \
    --use_svd \
    --exp low-vgg16-branch-$first_low_rank-$loss \
    --weights data/pretrained/gender.caffemodel \
    --num_rounds $rounds \
    --stepsize $stepsize