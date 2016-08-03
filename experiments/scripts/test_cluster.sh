#!/bin/bash

num=$1

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/test_cluster_align.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/test_cluster.py --gpu 0 \
    --model models/joint_entropy_loss/test.prototxt \
    --weights output/joint/vgg16_att_cls_square_loss_iter_80000.caffemodel \
    --imdb celeba_val_align \
    --n_cluster $num