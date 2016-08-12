#!/bin/bash

for i in `seq 7 15`;
do
    jbsub -mem 3g -mail -cores 1+1 -queue x86_short ./train_attr_dynamic.sh 8000 0.001 20 Sigmoid low-vgg-16 $i
done
