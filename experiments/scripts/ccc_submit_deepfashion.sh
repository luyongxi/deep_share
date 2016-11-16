#!/bin/bash

# submit deepfashion experiments

jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_baseline_deepfashion.sh 40000 16000 0.001 lowvgg16 16

# # experiments with branching on DeepFashion
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 0.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 1.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 2.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 3.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 4.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 0.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 0.5
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 1.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 2.0
