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
    --model output/low-vgg16-2-0-Sigmoid/celeba_train/prototxt/test.prototxt \
    --weights output/low-vgg16-2-0-Sigmoid/celeba_train/low-vgg-16-facial-attr-pretrained-0-Sigmoid_iter_30000.caffemodel \
    --cls_id 6,7,8,10,13,14,19,24,28,34,35,37,38,0,1,2,3,4,5,9,11,12,15,16,17,18,20,21,22,23,25,26,27,29,30,31,32,33,36,39 \
    --imdb $dataset

# time ./tools/test_multi.py --gpu 0 \
#     --model output/low-vgg16-0-0-Sigmoid/celeba_train/prototxt/test.prototxt \
#     --weights output/low-vgg16-0-0-Sigmoid/celeba_train/low-vgg-16-facial-attr-pretrained-0-Sigmoid_iter_30000.caffemodel \
#     --imdb $dataset