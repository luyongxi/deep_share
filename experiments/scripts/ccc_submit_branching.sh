#!/bin/bash

# experiments with branching on CelebA
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 1.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 2.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 3.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 4.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 1.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 0.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 0.5
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_celeba.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 2.0


# # experiments with branching on DeepFashion
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 0.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 1.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small32-lowvgg16 0 15 1000 2.0
# jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_deepfashion.sh 60000 20000 0.001 small64-lowvgg16 0 15 1000 2.0


# # experiments with joint dataset
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_person.sh 100000 40000 0.001 small64-lowvgg16 0 15 2000 1.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_person.sh 100000 40000 0.001 small64-lowvgg16 0 15 2000 2.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_person.sh 100000 40000 0.001 small32-lowvgg16 0 15 2000 1.0
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_branch_person.sh 100000 40000 0.001 small32-lowvgg16 0 15 2000 2.0