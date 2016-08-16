#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_joint_entropyloss.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/train_multi.py --gpu 0 \
    --solver models/joint_entropy_loss/solver.prototxt \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters 80000 \
    --exp vgg16_scratch
# --weights data/pretrained/gender.caffemodel \

