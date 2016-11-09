#!/bin/bash

# experiments with baseline using thin models
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_celeba.sh 60000 20000 0.001 small32-lowvgg16 0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0

# experiments with baseline using thin models (from scratch)
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_scratch_celeba.sh 60000 20000 0.001 small32-lowvgg16 0

# # experiments with baseline using large models
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_celeba.sh 40000 16000 0.001 lowvgg16 0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_deepfashion.sh 40000 16000 0.001 lowvgg16 0

# # experiments with baseline using low rank models
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_celeba.sh 40000 16000 0.001 lowvgg16 16
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_deepfashion.sh 40000 16000 0.001 lowvgg16 16