#!/bin/bash

iters=$1
base_lr=$2
clip_gradients=$3
loss=$4
model=$5
first_low_rank=$6
cut_depth=$7

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_dynamic_$4_loss_[model=$model,lr=$base_lr,first_low_rank=$first_low_rank,cut_depth=$cut_depth].txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_multi.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters $iters \
    --base_lr $base_lr \
    --clip_gradients $clip_gradients \
    --loss $loss \
    --model $model \
    --snapshot_prefix $model-facial-attr-pretrained-$first_low_rank-$loss \
    --first_low_rank $first_low_rank \
    --use_svd \
    --exp low-vgg16-$cut_depth-$first_low_rank-$loss-share_basis \
    --weights data/pretrained/gender.caffemodel \
    --cut_depth $cut_depth \
    --cut_points [[6,7,8,10,13,14,19,24,28,34,35,37,38],[0,1,2,3,4,5,9,11,12,15,16,17,18,20,21,22,23,25,26,27,29,30,31,32,33,36,39]] \
    --share_basis