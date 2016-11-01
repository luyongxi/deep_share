#!/bin/bash

iters=$1
model=$2
cut_depth=$3

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/traintest_ecmsim_${model}_${cut_depth}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
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
    --snapshot_prefix celeba_ecmsim-${cut_depth} \
    --last_low_rank 16 \
    --use_svd \
    --stepsize 8000 \
    --exp $model-ecmsim-$cut_depth \
    --weights data/pretrained/gender.caffemodel \
    --cut_depth ${cut_depth} \
    --cut_points [[1,2,6,11,18,25,27,32,33,34,37],[0,3,4,5,7,8,9,10,12,13,14,15,16,17,19,20,21,22,23,24,26,28,29,30,31,35,36,38,39]] \
    --share_basis

time ./tools/test_cls.py --gpu 0 \
  --model output/$model-ecmsim-${cut_depth}/celeba_train/prototxt/test.prototxt \
  --weights output/$model-ecmsim-${cut_depth}/celeba_train/celeba_ecmsim-${cut_depth}_iter_${iters}.caffemodel \
  --imdb celeba_test