#!/bin/bash

iters=$1
base_lr=$2
loss=$3
model=$4

# Need a better way to handel cases where files are corrupted. 

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_dynamic_$3_loss_[model=$model,lr=$base_lr].txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_attr.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters $iters \
    --infix [base_lr=$2] \
    --base_lr $base_lr \
    --loss $loss \
    --model $model \
    --snapshot_prefix $model-facial-attr