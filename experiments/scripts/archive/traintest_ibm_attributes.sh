#!/bin/bash

last_low_rank=$1

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/traintest_ibmattributes-$1.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/save_softlabels.py --gpu 0 \
  --imdb IBMattributes_train

time ./tools/train_cls.py --gpu 0 \
    --traindb IBMattributes_train \
    --valdb IBMattributes_val \
    --iters 20000 \
    --base_lr 0.001 \
    --clip_gradients 20 \
    --stepsize 16000 \
    --model low-vgg-16 \
    --loss Sigmoid \
    --last_low_rank $1 \
    --use_svd \
    --exp low-vgg-16-ibmattributes-$1 \
    --weights data/pretrained/gender.caffemodel \

time ./tools/test_cls.py --gpu 0 \
    --model output/low-vgg-16-ibmattributes-$1/IBMattributes_train/prototxt/test.prototxt \
    --weights output/low-vgg-16-ibmattributes-$1/IBMattributes_train/low-vgg-16-ibmattributes-$1_iter_20000.caffemodel \
    --imdb IBMattributes_val