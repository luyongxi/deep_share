#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/test_model.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
# time ./tools/test_cls.py --gpu 0 \
#   --model output/low-vgg16-$1-0-Sigmoid-share_basis/celeba_train/prototxt/test.prototxt \
#   --weights output/low-vgg16-$1-0-Sigmoid-share_basis/celeba_train/low-vgg-16-facial-attr-pretrained-0-Sigmoid_iter_80000.caffemodel \
#   --imdb celeba_val

# time ./tools/test_cls.py --gpu 0 \
#   --model output/low-vgg16-$1-0-Sigmoid-share_basis/celeba_train/prototxt/test.prototxt \
#   --weights output/low-vgg16-$1-0-Sigmoid-share_basis/celeba_train/low-vgg-16-facial-attr-pretrained-0-Sigmoid_iter_80000.caffemodel \
#   --imdb celeba_test

time ./tools/test_cls.py --gpu 0 \
  --model models/joint_entropy_loss/test.prototxt \
  --weights output/joint/vgg16_att_cls_entropy_loss_iter_20000.caffemodel \
  --imdb celeba_test

# time ./tools/test_cls.py --gpu 0 \
#   --model output/low-vgg16-$1-0-Sigmoid/celeba_train/prototxt/test.prototxt \
#   --weights output/low-vgg16-$1-0-Sigmoid/celeba_train/low-vgg-16-facial-attr-pretrained-0-Sigmoid_iter_80000.caffemodel \
#   --imdb celeba_val

# time ./tools/test_cls.py --gpu 0 \
#   --model output/low-vgg16-$1-0-Sigmoid/celeba_train/prototxt/test.prototxt \
#   --weights output/low-vgg16-$1-0-Sigmoid/celeba_train/low-vgg-16-facial-attr-pretrained-0-Sigmoid_iter_80000.caffemodel \
#   --imdb celeba_test