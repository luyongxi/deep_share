#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
model=$4
last_low_rank=$5
rounds=$6
aff_type=$7

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_dynamic_branch_scratch_${aff_type}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
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
    --snapshot_prefix ${model}-celeba-branch_scratch_${last_low_rank} \
    --last_low_rank ${last_low_rank} \
    --use_svd \
    --exp ${model}-branch-scratch-${last_low_rank}-${aff_type} \
    --num_rounds ${rounds} \
    --stepsize ${stepsize} \
    --aff_type ${aff_type} \
    --share_basis \
    --use_bn
