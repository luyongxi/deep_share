#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
model=$4
last_low_rank=$5
cls_id=$6

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_baseline_celeba_${model}_${last_low_rank}_${cls_id}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_cls.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters ${iters} \
    --base_lr ${base_lr} \
    --clip_gradients 20 \
    --loss Sigmoid \
    --model ${model} \
    --snapshot_prefix celeba_train\
    --last_low_rank ${last_low_rank} \
    --cls_id [${cls_id}] \
    --use_svd \
    --exp celeba_baseline_${model}_${last_low_rank}_${cls_id} \
    --max_rounds 1 \
    --stepsize ${stepsize} \
    --weights data/pretrained/gender.caffemodel \
    --use_bn