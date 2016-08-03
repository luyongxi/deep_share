# Written by Yongxi Lu

""" Dynamically creates solver prototxt files """

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import os.path as osp

class DynamicSolver(object):
    """This class represents a solver prototxt file """

    def __init__(self, net, base_lr=0.01, lr_policy="step", 
        gamma=0.1, stepsize=20000, momentum=0.9, weight_decay=0.0005, 
        snapshot_prefix='default'):

        self._net = net
        self._base_lr = base_lr
        self._lr_policy = lr_policy
        self._gamma = gamma
        self._stepsize = stepsize
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._snapshot_prefix = snapshot_prefix

    def to_proto(self, solver_path=''):
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

        fn = osp.join(solver_path, 'solver.prototxt')
        with open(fn, 'w') as f:
            f.write(text_format.MessageToString(solver))

if __name__ == '__main__':
    # test by generating a default solver
    ds = DynamicSolver('train_val.prototxt')
    ds.to_proto()
