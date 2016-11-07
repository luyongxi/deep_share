#!/bin/bash

iters=$1
model=$2
cut_depth=$3

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/traintest_binary_${model}_${cut_depth}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
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
    --last_low_rank 16 \
    --use_svd \
    --stepsize 8000 \
    --exp $model-binary-$cut_depth \
    --weights data/pretrained/gender.caffemodel \
    --cut_depth ${cut_depth} \
    --cut_points [[0,1,4,6,9,14,16,19,20,21,22,24,25,29,30,31,32,33,35,38],[2,3,5,7,8,10,11,12,13,15,17,18,23,26,27,28,34,36,37,39]] \
    --share_basis

time ./tools/test_cls.py --gpu 0 \
  --model output/$model-binary-${cut_depth}/celeba_train/prototxt/test.prototxt \
  --weights output/$model-binary-${cut_depth}/celeba_train/celeba_binary-${cut_depth}_iter_${iters}.caffemodel \
  --imdb celeba_test