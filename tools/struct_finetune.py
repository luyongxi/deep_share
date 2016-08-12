#!/usr/bin/env python

# -------------------------
# Written by Yongxi Lu
# -------------------------


""" Finetune the structure of the network to improve accuracy """

# TODO: add options related to branching, call this parameter something like a "rounds"
# By default, we merely have one round of training. 
# If there is more than one round of training, then we should allow --iters options to
# be more than a single value. What is the most elegant way to achieve this? Intuitively, 
# most of the training iterations should be spent on the first phase of training as we need
# to boostrap the parameters, after creating a branch more training iterations are not
# necessary. 

# Perhaps the most elegant solution is to treat the initial training phase as something "special".
# This could mean, we need to train the model from scratch, or we initilize a low-rank version 
# of a pre-trained model.

# Then, using the pretrained model, we perform multiple-rounds of structured finetuning. 

