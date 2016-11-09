#!/bin/bash

iters=$1
solver=$2
exp_postfix=$3

if [ $# -eq 4 ] ; then
  class_id='[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]'
elif [ $# -eq 4 ] ; then
  class_id=$4
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_branch_celeba_${exp_postfix}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/train_cls.py --gpu 0 \
    --traindb celeba_train \
    --valdb celeba_val \
    --iters ${iters} \
    --solver ${solver} \
    --exp celeba_solver_${exp_postfix} \
    --share_basis \
    --cls_id ${class_id} \
    --use_bn