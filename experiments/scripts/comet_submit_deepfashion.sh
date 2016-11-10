#!/bin/bash

# This script submits deep-fashion related jobs to the comet server. 

# baseline models
sbatch -o=~/.slurm/slurm.%N.%j.out -J="baseline_vgg16" -N=1 -p=gpu-shared --gres=gpu:1 -t=48:00:00 --wrap="./train_baseline_deepfashion.sh 40000 16000 0.001 lowvgg16 0"
sbatch -o=~/.slurm/slurm.%N.%j.out -J="baseline_lowvgg16" -N=1 -p=gpu-shared --gres=gpu:1 -t=48:00:00 --wrap="./train_baseline_deepfashion.sh 40000 16000 0.001 lowvgg16 16"

# # experiments with branching on DeepFashion
sbatch -o=~/.slurm/slurm.%N.%j.out -J="br_low" -N=1 -p=gpu-shared --gres=gpu:1 -t=48:00:00 --wrap="./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 0.0"
sbatch -o=~/.slurm/slurm.%N.%j.out -J="br_med" -N=1 -p=gpu-shared --gres=gpu:1 -t=48:00:00 --wrap="./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 1.0"
sbatch -o=~/.slurm/slurm.%N.%j.out -J="br_high" -N=1 -p=gpu-shared --gres=gpu:1 -t=48:00:00 --wrap="./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 2.0"