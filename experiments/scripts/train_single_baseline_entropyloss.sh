#!/bin/bash

cls_id=$1

if [ $# -eq 0]; then
	echo "Need to specify class id"
	exit 1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_baseline_single[cls=$cls_id]_entropyloss.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/train_attr.py --gpu 0 \
    --solver models/baseline_single_entropy_loss/solver.prototxt \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters 40000 \
    --cls_id "$cls_id" \
    --infix [cls=$cls_id]