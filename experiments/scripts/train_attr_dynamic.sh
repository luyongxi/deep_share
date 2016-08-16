#!/bin/bash

iters=$1
base_lr=$2
clip_gradients=$3
loss=$4
model=$5
first_low_rank=$6

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_dynamic_$4_loss_[model=$model,lr=$base_lr,first_low_rank=$first_low_rank].txt.`date +'%Y-%m-%d_%H-%M-%S'`"
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
    --exp low-vgg16-scratch-$first_low_rank-$loss
    # --weights data/pretrained/gender.caffemodel \
    # --exp $model-facial-attr-pretrained-$first_low_rank-$loss \
