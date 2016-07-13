#!/bin/bash

cls_id1=$1
cls_id2=$2
name=$3

if [ $# -lt 3]; then
	echo "Need to specify (cls1, cls2) and name of the model (baseline or branch)"
	exit 1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_[$name]_pair[cls=$cls_id1,$cls_id2]_entropyloss.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/train_attr.py --gpu 0 \
    --solver models/"$name"_pair_entropy_loss/solver.prototxt \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters 20000 \
    --cls_id "$cls_id1,$cls_id2" \
    --infix [cls=$cls_id1,$cls_id2]