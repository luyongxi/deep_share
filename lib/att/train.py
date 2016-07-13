#---------------------------
# Written by Yongxi Lu
#---------------------------

"""
Train attribute classifiers
"""

import datasets
from utils.config import cfg
from utils.timer import Timer
import numpy as np
import os
import caffe

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """ Wrapping around Caffe's solver """
    
    def __init__(self, imdb, solver_prototxt, output_dir, pretrained_model=None, cls_id=None):
        """ Initialize the SolverWrapper. """

        # use output path
        self._output_dir = output_dir

        # initialize solver, load pretrained model is provided
        self._solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self._solver.net.copy_from(pretrained_model)
 
        # load solver parameters
        self._solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self._solver_param)

        # load imdb to layers
        self._solver.net.layers[0].set_imdb(imdb['train'])
        if cfg.TRAIN.USE_VAL is True:
            self._solver.test_nets[0].layers[0].set_imdb(imdb['val'])
       
        # get the number of prediction classes
        self._num_classes = self._solver.net.layers[0].num_classes

        # set class list
        if cls_id is not None:
            self._solver.net.layers[0].set_classlist(cls_id)
            if cfg.TRAIN.USE_VAL is True:
                self._solver.test_nets[0].layers[0].set_classlist(cls_id)

    def snapshot(self, base_iter):
        """ Save a snapshot of the network """
        
        net = self._solver.net

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX 
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        # filename should adjust to current iterations
        filename = (self._solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self._solver.iter + 
                    base_iter) + '.caffemodel')
        filename = os.path.join(self._output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

    def train_model(self, max_iters, base_iter):
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
            acc_train = self._solver.net.blobs['error'].data
            print 'Iteration {}: training error = {}'.format(cur_iter, acc_train.ravel())

            acc_val = np.zeros((1, self._num_classes))            
            if cur_iter % cfg.TRAIN.VAL_FREQ == 0:

                # display training speed 
                print 'speed: {:.3f}s /iter'.format(timer.average_time)
                # perform validation    
                for _ in xrange(cfg.TRAIN.VAL_SIZE):
                    self._solver.test_nets[0].forward()
                    acc_val += self._solver.test_nets[0].blobs['error'].data
                acc_val /= cfg.TRAIN.VAL_SIZE
                print 'Iteration {}: validation error = {}'.format(cur_iter, acc_val.ravel())

            if cur_iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = cur_iter
                self.snapshot(base_iter)

        # save snapshot if we haven't done so
        if last_snapshot_iter != cur_iter:
            self.snapshot(base_iter)

def train_attr(imdb, solver_prototxt, output_dir, pretrained_model=None, max_iters=40000, base_iter=0, cls_id=None):
    """ 
    Train a attribute classification model
    """

    sw = SolverWrapper(imdb, solver_prototxt, output_dir, pretrained_model=pretrained_model, cls_id=cls_id)

    print 'Solving...'
    sw.train_model(max_iters, base_iter)
    print 'done solving'
