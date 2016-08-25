# Written by Yongxi Lu

""" Dynamically creates solver prototxt files """

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb2
import os.path as osp
import os
import caffe
from utils.config import cfg
import numpy as np
from numpy.linalg import svd


# TODO: add a function that adds small random noise to break even!
# TODO: essentially, this function takes as input a list of parameter
# names. These names are aggregated by a NetModel class through insert branch.

class SolverWrapper(object):
    """ Wrapper for a sovler used in training. """
    def __init__(self, imdb, solver_prototxt, output_dir, pretrained_model=None, 
        param_mapping=None, use_svd=True):
        """ Initialize the SolverWrapper. """

        # use output path
        self._output_dir = output_dir

        # initialize solver, load pretrained model is provided
        self._solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
               'weights from {:s}').format(pretrained_model)
            if param_mapping is None:
                self._solver.net.copy_from(pretrained_model)
            else:
                self._load_mapped_params(pretrained_model, param_mapping, use_svd)

        # load solver parameters
        self._solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self._solver_param)

    def _init_params_svd(self, W, k):
        """ Given input filters, return a set of basis and the linear combination
            required to approximate the original input filters
            Input: 
                W: [dxc] matrix, where c is the input dimension, 
                    d is the output dimension
            Output:
                B: [kxc] matrix, where c is the input dimension, 
                    k is the maximum rank of output filters
                L: [dxk] matrix, where k is the maximum rank of the
                    output filters, d is the output dimension

            Note that k <= min(c,d). It is an error if that is encountered.
        """
        d, c = W.shape
        assert k <= min(c,d), 'k={} is too large for c={}, d={}'.format(k,c,d)
        # S in this case is a vector with len=K=min(c,d), and U is [d x K], V is [K x c]
        u, s, v = svd(W, full_matrices=False)
        # compute square of s -> s_sqrt
        s_sqrt = np.sqrt(s[:k])
        # extract L from u
        B = v[:k, :] * s_sqrt[:, np.newaxis]
        # extract B from v
        L = u[:, :k] * s_sqrt
 
        return B, L

    def _load_mapped_params(self, pretrained_model, param_mapping, use_svd=True):
        """ Load selected parameters specified in param_mapping
            from the pre-trained model to the new model. 

            param_mapping: key is the names in the new model, value
                           is the names in the pretrained model. 
        """
        with open(pretrained_model, 'rb') as f:
            binary_content = f.read()

        model = caffe_pb2.NetParameter()
        model.ParseFromString(binary_content)
        layers = model.layer

        weight_cache = {}
        for key, value in param_mapping.iteritems():
            # 1-1 matching, direct copy
            if len(key) == 1:
                print 'saving net[{}] <- pretrained[{}]...'.format(key[0], value)
                found=False
                for layer in layers:
                    if layer.name == value:
                        self._solver.net.params[key[0]][0].data[...] = \
                            np.reshape(np.array(layer.blobs[0].data), layer.blobs[0].shape.dim) 
                        self._solver.net.params[key[0]][1].data[...] = \
                            np.reshape(np.array(layer.blobs[1].data), layer.blobs[1].shape.dim)
                        found=True
                        print 'saving net[{}] <- pretrained[{}] done.'.format(key[0], value)
                if not found:
                    print '!!! pretrained[{}] not found!'.format(value)
            elif len(key) == 2:
                if use_svd:
                    print 'saving net[{}, {}] <- pretrained[{}] ...'.format(key[0], key[1], value)
                    found=False
                    for layer in layers:
                        if layer.name == value:
                            # use svd to initialize
                            # W is the weight matrix, k is the number of outputs
                            W = np.reshape(np.array(layer.blobs[0].data), (layer.blobs[0].shape.dim[0], -1))
                            # size of the target parameters
                            basis_shape = self._solver.net.params[key[0]][0].data.shape
                            linear_shape = self._solver.net.params[key[1]][0].data.shape
                            # perform decomposition, save results. 

                            if weight_cache.has_key(value):
                                B, L = weight_cache[value]
                                print 'using cached svd decomposition of pretrained[{}]...'.format(value)
                            else:
                                B, L = self._init_params_svd(W, basis_shape[0])
                                weight_cache[value] = (B, L)

                            self._solver.net.params[key[0]][0].data[...] = B.reshape(basis_shape)
                            self._solver.net.params[key[1]][0].data[...] = L.reshape(linear_shape)
                            # use the bias of the original conv filter in the linear combinations
                            self._solver.net.params[key[1]][1].data[...] = \
                                np.reshape(np.array(layer.blobs[1].data), layer.blobs[1].shape.dim)
                            found=True
                            print 'net[{}, {}] <- pretrained[{}] done.'.format(key[0], key[1], value)
                    if not found:
                        print '!!! pretrained[{}] not found!'.format(value)
                else:
                    print 'use_svd is set to False, skipping net[({}, {})] <- pretrained[{}]'.\
                        format(key[0], key[1], value)

    def snapshot_name(self, base_iter):
        """ Return the name of the snapshot file """
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX 
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self._solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self._solver.iter + 
                    base_iter) + '.caffemodel')
        filename = os.path.join(self._output_dir, filename)

        return filename

    def snapshot(self, base_iter):
        """ Save a snapshot of the network """
        
        net = self._solver.net

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        # filename should adjust to current iterations
        filename = self.snapshot_name(base_iter)
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

    def train_model(self, max_iters, base_iter):
        """Train the model with iterations=max_iters"""
        print 'Solving...'
        self.do_train_model(max_iters, base_iter)
        print 'done solving'

class SolverParameter(object):
    """This class represents a solver prototxt file """

    def __init__(self, path, base_lr=0.01, lr_policy="step", 
        gamma=0.1, stepsize=20000, momentum=0.9, weight_decay=0.0005,
        regularization_type="L2", clip_gradients=None, snapshot_prefix='default'):

        self._net = osp.join(path, 'train_val.prototxt')
        self._base_lr = base_lr
        self._lr_policy = lr_policy
        self._gamma = gamma
        self._stepsize = stepsize
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._regularization_type = regularization_type
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
        solver.regularization_type = self._regularization_type

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
    sp = SolverParameter()
    sp.to_proto()
