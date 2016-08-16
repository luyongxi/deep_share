#---------------------------
# Written by Yongxi Lu
#---------------------------

"""
Train Single Label Classifier
"""

import datasets
from utils.config import cfg
from utils.timer import Timer
from .solver import SolverWrapper
import numpy as np
import os
import caffe

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SingleLabelSW(SolverWrapper):
    """ Wrapper around Caffe's solver """
    
    def __init__(self, imdb, solver_prototxt, output_dir, pretrained_model=None, param_mapping=None, use_svd=True):
        """ Initialize the SolverWrapper. """
        SolverWrapper.__init__(self, imdb, solver_prototxt, output_dir, pretrained_model, param_mapping, use_svd)

        # load imdb to layers
        self._solver.net.layers[0].set_imdb(imdb['train'])
        if cfg.TRAIN.USE_VAL is True:
            self._solver.test_nets[0].layers[0].set_imdb(imdb['val'])
       
        # get the number of prediction classes
        self._num_classes = self._solver.net.layers[0].num_classes

    def do_train_model(self, max_iters, base_iter):
        """Train the model with iterations=max_iters"""

        last_snapshot_iter = -1
        timer = Timer()
        while self._solver.iter < max_iters:
            timer.tic()
            self._solver.step(1)
            timer.toc()
            # adjust iteration
            cur_iter = self._solver.iter + base_iter
            # evaluate training performance
            acc_train = self._solver.net.blobs['acc'].data
            loss_train = self._solver.net.blobs['loss'].data
            print 'Iteration {}: training accuracy = {}'.format(cur_iter, acc_train.ravel())
            print 'Iteration {}: training loss = {}'.format(cur_iter, loss_train)

            acc_val = np.zeros((1,))            
            if cur_iter % cfg.TRAIN.VAL_FREQ == 0:

                # display training speed 
                print 'speed: {:.3f}s /iter'.format(timer.average_time)
                # perform validation    
                for _ in xrange(cfg.TRAIN.VAL_SIZE):
                    self._solver.test_nets[0].forward()
                    acc_val += self._solver.test_nets[0].blobs['acc'].data
                acc_val /= cfg.TRAIN.VAL_SIZE
                print 'Iteration {}: validation accuracy = {}'.format(cur_iter, acc_val.ravel())

            if cur_iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = cur_iter
                self.snapshot(base_iter)

        # save snapshot if we haven't done so
        if last_snapshot_iter != cur_iter:
            self.snapshot(base_iter)
