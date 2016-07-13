#!/bin/bash

jbsub -mem 3g -mail -cores 1+1 -queue x86 ./train_joint_entropyloss.sh
jbsub -mem 3g -mail -cores 1+1 -queue x86 ./train_joint_squareloss.sh