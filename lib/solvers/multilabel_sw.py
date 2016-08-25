#---------------------------
# Written by Yongxi Lu
#---------------------------

"""
Train multilabel classifier
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

import json

class MultiLabelSW(SolverWrapper):
    """ Wrapper around Caffe's solver """
    
    def __init__(self, imdb, solver_prototxt, output_dir, 
        pretrained_model=None, param_mapping=None, use_svd=True, cls_id=None):
        """ Initialize the SolverWrapper. """

        SolverWrapper.__init__(self, imdb, solver_prototxt, output_dir, 
            pretrained_model, param_mapping, use_svd)

        # load imdb to layers
        self._solver.net.layers[0].set_imdb(imdb['train'])
        if cfg.TRAIN.USE_VAL is True:
            self._solver.test_nets[0].layers[0].set_imdb(imdb['val'])
       
        # get the number of prediction classes
        self._num_classes = self._solver.net.layers[0].num_classes

        # set class list
        self._cls_id = cls_id
        if cls_id is not None:
            self._solver.net.layers[0].set_classlist(cls_id)
            if cfg.TRAIN.USE_VAL is True:
                self._solver.test_nets[0].layers[0].set_classlist(cls_id)

    def snapshot(self, base_iter):
        """ Save class list to the text file with the same name as caffemodel"""
        caffename = self.snapshot_name(base_iter)
        fn = os.path.splitext(caffename)[0] + '.clsid'
        with open(fn, 'wb') as f:
            f.write(json.dumps(self._cls_id))

        SolverWrapper.snapshot(self, base_iter)

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
            err_train = self._solver.net.blobs['error'].data
            loss_train = self._solver.net.blobs['loss'].data
            print 'Iteration {}: training error = {}'.format(cur_iter, err_train.ravel())
            print 'Iteration {}: training loss = {}'.format(cur_iter, loss_train)

            err_val = np.zeros((1, self._num_classes))            
            if cur_iter % cfg.TRAIN.VAL_FREQ == 0:

                # display training speed 
                print 'speed: {:.3f}s /iter'.format(timer.average_time)
                # perform validation    
                for _ in xrange(cfg.TRAIN.VAL_SIZE):
                    self._solver.test_nets[0].forward()
                    err_val += self._solver.test_nets[0].blobs['error'].data
                err_val /= cfg.TRAIN.VAL_SIZE
                print 'Iteration {}: validation error = {}'.format(cur_iter, err_val.ravel())

            if cur_iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = cur_iter
                self.snapshot(base_iter)

        # save snapshot if we haven't done so
        if last_snapshot_iter != cur_iter:
            self.snapshot(base_iter)
