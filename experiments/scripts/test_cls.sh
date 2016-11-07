#!/bin/bash

train_imdb=$1
test_imdb=$2
datapath=$3

if [ $# -eq 3 ] ; then
  outputpath="output"
elif [ $# -eq 4 ] ; then
  outputpath=$4
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/test_model_${train_imdb}_${test_imdb}_${datapath}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/test_cls.py --gpu 0 \
  --model ${outputpath}/${datapath}/${train_imdb}/prototxt/test.prototxt \
  --weights ${outputpath}/${datapath}/${train_imdb}/deploy.caffemodel \
  --imdb ${test_imdb}