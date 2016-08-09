# Written by Yongxi Lu

""" Dynamically creates solver prototxt files """

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb2
import os.path as osp
import os
import caffe
from utils.config import cfg


class SolverWrapper(object):
    """ Wrapper for a sovler used in training. """
    def __init__(self, imdb, solver_prototxt, output_dir, pretrained_model=None):
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
        return NotImplementedError

class DynamicSolver(object):
    """This class represents a solver prototxt file """

    def __init__(self, path='', base_lr=0.01, lr_policy="step", 
        gamma=0.1, stepsize=20000, momentum=0.9, weight_decay=0.0005,
        clip_gradients=None, snapshot_prefix='default'):

        self._net = osp.join(path, 'train_val.prototxt')
        self._base_lr = base_lr
        self._lr_policy = lr_policy
        self._gamma = gamma
        self._stepsize = stepsize
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._clip_gradients = clip_gradients
        self._snapshot_prefix = snapshot_prefix

    def solver_file(self):
        solver_path = osp.dirname(self._net)
        return osp.join(solver_path, 'solver.prototxt')

    def to_proto(self):
        """ Save a solver.prototxt file to the specified path """

        solver = caffe_pb2.SolverParameter()
        solver.net = self._net
        solver.base_lr = self._base_lr
        solver.lr_policy = self._lr_policy
        solver.gamma = self._gamma
        solver.stepsize = self._stepsize
        solver.momentum = self._momentum
        solver.weight_decay = self._weight_decay
        # caffe solver snapshotting is disabled
        solver.snapshot = 0
        solver.snapshot_prefix = self._snapshot_prefix
        # shut down caffe display
        solver.display = 0
        # shut down caffe validation
        solver.test_iter.append(0)
        solver.test_interval = 1000
        # clip_gradients
        if self._clip_gradients is not None:
            solver.clip_gradients = self._clip_gradients

        solver_path = osp.dirname(self._net)
        if not osp.exists(solver_path):
            os.makedirs(solver_path)
        with open(self.solver_file(), 'w') as f:
            f.write(text_format.MessageToString(solver))

if __name__ == '__main__':
    # test by generating a default solver
    ds = DynamicSolver()
    ds.to_proto()
