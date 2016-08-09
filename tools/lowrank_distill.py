#!/usr/bin/env python

# -------------------------
# Written by Yongxi Lu
# -------------------------

""" Distill a low-rank (and likely smaller) student model using 
    soft inputs from a teacher model.
"""

# The motivation of this procedure is to initialize a low-rank model 
# given a (set of) full rank models. We experiment with the simple
# procedure of training the low rank models using the soft outputs
# from the full rank model. The full rank model can easily take 
# advantage of pre-trained full rank models. 

# This file is very similar to train_attr.py, except that we need to provide it with
# some source of soft labels. Soft labels can be easily obtained by testing teacher
# models on training set, which should be implemented as a separated function. 

