#!/bin/bash

train_imdb=$1
test_imdb=$2
datapath=$3
round=$4

if [ $# -eq 4 ] ; then
  outputpath="output"
elif [ $# -eq 5 ] ; then
  outputpath=$5
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/test_model_${train_imdb}_${test_imdb}_${datapath}_${round}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/test_cls.py --gpu 0 \
  --model ${outputpath}/${datapath}/${train_imdb}/prototxt/round_${round}/test.prototxt \
  --weights ${outputpath}/${datapath}/${train_imdb}/round_${round}_deploy.caffemodel \
  --imdb ${test_imdb}