#!/bin/bash

num=$1
method=$2

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/test_cluster_${num}_${method}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/test_cluster.py --gpu 0 \
    --model /dccstor/luyo1/multi-task-output/narrow-low-vgg-16-binary-0/celeba_train/prototxt/test.prototxt \
    --weights /dccstor/luyo1/multi-task-output/narrow-low-vgg-16-binary-0/celeba_train/celeba_binary-0_iter_20000.caffemodel \
    --imdb celeba_val \
    --n_cluster $num \
    --method $method