#!/bin/bash

cls_id=$1

if [ $# -eq 0]; then
	echo Need to specify the class to test
	exit 1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_single[cls=$cls_id]_entropyloss.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/train_attr.py --gpu 0 \
    --solver models/single_entropy_loss/solver.prototxt \
    --weights data/pretrained/gender.caffemodel \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters 8000 \
    --cls_id $cls_id \
    --infix [cls=$cls_id]