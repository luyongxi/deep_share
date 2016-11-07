#!/bin/bash

for i in `seq 0 39`;
do
    jbsub -mem 10g -mail -cores 4+1 -queue x86_short ./train_baseline_single_celeba.sh 10000 8000 0.001 small32-lowvgg16 0 ${i}
done

# for i in `seq 0 39`;
# do
#     jbsub -mem 10g -mail -cores 4+1 -queue x86 ./train_baseline_single_celeba.sh 10000 8000 0.001 lowvgg16 0 ${i}
# done