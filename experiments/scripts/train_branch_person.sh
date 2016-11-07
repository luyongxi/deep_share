#!/bin/bash

iters=$1
stepsize=$2
base_lr=$3
model=$4
last_low_rank=$5
max_rounds=$6
max_stall=$7
split_thresh=$8

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/train_branch_person_${model}_${last_low_rank}_${max_rounds}_${max_stall}_${split_thresh}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..

time ./tools/load_person.py --gpu 0 \
    --model_face output/celeba_baseline_lowvgg16_0/celeba_train/prototxt/test.prototxt \
    --weights_face output/celeba_baseline_lowvgg16_0/celeba_train/celeba_train_iter_40000.caffemodel \
    --model_clothes output/deepfashion_baseline_lowvgg16_0/deepfashion_train/prototxt/test.prototxt \
    --weights_clothes output/deepfashion_baseline_lowvgg16_0/deepfashion_train/deepfashion_train_iter_40000.caffemodel \
    --imdb_face person_clothes_train \
    --imdb_clothes person_face_train

time ./tools/load_person.py --gpu 0 \
    --model_face output/celeba_baseline_lowvgg16_0/celeba_train/prototxt/test.prototxt \
    --weights_face output/celeba_baseline_lowvgg16_0/celeba_train/celeba_train_iter_40000.caffemodel \
    --model_clothes output/deepfashion_baseline_lowvgg16_0/deepfashion_train/prototxt/test.prototxt \
    --weights_clothes output/deepfashion_baseline_lowvgg16_0/deepfashion_train/deepfashion_train_iter_40000.caffemodel \
    --imdb_face person_clothes_val \
    --imdb_clothes person_face_val

time ./tools/train_cls.py --gpu 0 \
    --traindb person_train \
    --valdb person_val \
    --iters ${iters} \
    --base_lr ${base_lr} \
    --clip_gradients 20 \
    --loss Sigmoid \
    --model ${model} \
    --last_low_rank ${last_low_rank} \
    --use_svd \
    --exp person_${model}_${last_low_rank}_${max_rounds}_${max_stall}_${split_thresh} \
    --max_rounds ${max_rounds} \
    --stepsize ${stepsize} \
    --weights data/pretrained/gender.caffemodel \
    --share_basis \
    --use_bn \
    --max_stall ${max_stall} \
    --split_thresh ${split_thresh}