#!/bin/bash

for i in `seq 0 39`;
do
	jbsub -mem 3g -mail -cores 1+1 -queue x86_short ./train_single_entropyloss.sh $i
done