#!/bin/bash

jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_prototxt.sh 60000 output/celeba_small32-lowvgg16_0_15_1000_2.0/celeba_train/prototxt/round_11/solver.prototxt original [19,31,18,26,36,4,13,14,17,38,15,16,22,30,1,8,27,29,33,34,37,3,7,39,23,32,21,25,6,5,10,20,2,9,28,0,12,24,11,35]
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_prototxt.sh 60000 output/celeba_small32-lowvgg16_0_15_1000_2.0/celeba_train/prototxt/round_11/solver.prototxt rand1 [27,20,26,17,25,28,18,13,23,14,33,37,6,2,12,22,11,24,5,3,35,39,21,31,15,1,34,30,8,38,10,29,32,36,19,16,7,9,4,0]
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_prototxt.sh 60000 output/celeba_small32-lowvgg16_0_15_1000_2.0/celeba_train/prototxt/round_11/solver.prototxt rand2 [31,26,20,5,35,11,28,25,21,2,34,6,3,22,9,17,37,27,29,33,30,8,24,16,32,0,14,19,39,13,12,15,18,4,10,1,38,23,7,36]
jbsub -mem 10g -cores 4+1 -queue x86 -mail ./train_prototxt.sh 60000 output/celeba_small32-lowvgg16_0_15_1000_2.0/celeba_train/prototxt/round_11/solver.prototxt rand3 [3,6,31,28,1,36,8,18,35,21,17,2,15,10,9,12,0,4,33,11,16,37,24,20,38,39,30,14,29,32,7,22,19,27,5,13,26,25,34,23]

