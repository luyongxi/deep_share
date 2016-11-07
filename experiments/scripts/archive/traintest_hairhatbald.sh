#!/bin/bash

last_low_rank=$1

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/traintest_hairhatbald-$1.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_cls.py --gpu 0 \
    --traindb celeba_plus_webcam_cls_train \
    --valdb celeba_plus_webcam_cls_val \
    --iters 10000 \
    --base_lr 0.001 \
    --clip_gradients 20 \
    --stepsize 6000 \
    --model low-vgg-16 \
    --loss Softmax \
    --last_low_rank $1 \
    --use_svd \
    --exp low-vgg-16-hairhatbald-$1 \
    --weights data/pretrained/hairhatbald.caffemodel \
    --task_name singlelabel

time ./tools/test_cls.py --gpu 0 \
    --model output/low-vgg-16-hairhatbald-$1/celeba_plus_webcam_cls_train/prototxt/test.prototxt \
    --weights output/low-vgg-16-hairhatbald-$1/celeba_plus_webcam_cls_train/low-vgg-16-hairhatbald-$1_iter_10000.caffemodel \
    --task_name singlelabel \
    --imdb celeba_plus_webcam_cls_val