#!/bin/bash

base_iter=$1
weight_file=$2

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/hairhatbald.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/train_single.py --gpu 0 \
    --solver models/hairhatbald/solver.prototxt \
    --weights $weight_file \
    --traindb celeba_plus_webcam_cls_train \
    --valdb celeba_plus_webcam_cls_val \
    --iters 5000 \
    --base_iter $base_iter
