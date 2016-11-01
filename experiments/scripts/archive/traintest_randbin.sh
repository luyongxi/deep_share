#!/bin/bash

iters=$1
model=$2
cut_depth=$3

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/traintest_randbin_${model}_${cut_depth}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_cls.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters $iters \
    --base_lr 0.001 \
    --clip_gradients 20 \
    --loss Sigmoid \
    --model $model \
    --snapshot_prefix celeba_randbin-${cut_depth} \
    --last_low_rank 16 \
    --use_svd \
    --stepsize 8000 \
    --exp $model-randbin-${cut_depth} \
    --weights data/pretrained/gender.caffemodel \
    --cut_depth ${cut_depth} \
    --cut_points [[11,12,15,16,17,6,7,8,10,13,14,19,24,28,34,35,37,38],[0,1,2,3,4,5,9,18,20,21,22,23,25,26,27,29,30,31,32,33,36,39]] \
    --share_basis

time ./tools/test_cls.py --gpu 0 \
  --model output/$model-randbin-${cut_depth}/celeba_train/prototxt/test.prototxt \
  --weights output/$model-randbin-${cut_depth}/celeba_train/celeba_randbin-${cut_depth}_iter_${iters}.caffemodel \
  --imdb celeba_test