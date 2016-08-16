#!/bin/bash

dataset=$1

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/test_model.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/test_multi.py --gpu 0 \
    --model output/low-vgg-16-facial-attr-pretrained-7-Sigmoid/celeba_train/prototxt/low-vgg-16/test.prototxt \
    --weights output/low-vgg-16-facial-attr-pretrained-7-Sigmoid/celeba_train/low-vgg-16-facial-attr-pretrained-7-Sigmoid_iter_80000.caffemodel \
    --imdb $dataset