#!/bin/bash

test_class=$1

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/soft_labels-$1.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/save_softlabels.py --gpu 0 \
  --imdb IBMattributes_train \
  --test_class $test_class