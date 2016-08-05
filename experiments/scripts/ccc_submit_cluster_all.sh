#!/bin/bash

for i in `seq 2 10`;
do
    jbsub -mem 3g -mail -cores 1+1 -queue x86_short ./test_cluster.sh $i lcm
done