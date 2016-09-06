# Written by Yongxi Lu

""" Wrappers for solvers and training hyperparameters """

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb2
import os.path as osp
import os
import caffe
from utils.config import cfg
import numpy as np
from numpy.linalg import svd

class SolverWrapper(object):
    """ Wrapper for a sovler used in training. """
    def __init__(self, imdb, output_dir, solver_params, pretrained_params, model_params):
        """ Initialize the SolverWrapper. """
        self._cur_round = 0
        # use output path
        self._output_dir = output_dir
        # imdb
        self._imdb = imdb
        # save parameters
        self._solver_params = solver_params
        self._pretrained_params = pretrained_params
        self._model_params = model_params
        # load solvers and pretrained models
        # initialize solver
        self._solver = self._load_solver(self._solver_params, self._model_params)
        # load pretrained models if provided
        self._load_pretrained_model(self._pretrained_params)

    def _load_solver(self, solver_params, model_params):
        """ Load solver """
        solver_path = solver_params.path        
        if model_params.num_rounds > 1:
            solver_path = osp.join(solver_path, 'round_{}'.format(self._cur_round))
            solver_params.set_path(solver_path)

        model_params.model.to_proto(solver_path, deploy=False)
        model_params.model.to_proto(solver_path, deploy=True)
        print 'Model files saved at {}'.format(solver_path)

        return caffe.SGDSolver(solver_params.to_proto())

    def _load_pretrained_model(self, pp):
        """ load pre-trained model """
        if pp.pretrained_model is not None:
            print ('Loading pretrained model '
               'weights from {:s}').format(pp.pretrained_model)
            if pp.param_mapping is None:
                self._solver.net.copy_from(pp.pretrained_model)
            else:
                self._load_mapped_params(pp.pretrained_model, pp.param_mapping, pp.use_svd)

            if pp.param_rand is not None:
                self._add_noise(pp.param_rand, cfg.TRAIN.NOISE_FACTOR)

    def _add_noise(self, param_rand, factor):
        """ Add random noise to the parameters of layers specfied in param_rand """
        for name in param_rand:
            # heuristic: replace 1/4th of the variance with noise
            if factor > 0:
                param_var = np.var(self._solver.net.params[name][0].data)
                var = param_var * factor
                self._solver.net.params[name][0].data[...] =\
                    np.sqrt(1.0-factor)*self._solver.net.params[name][0].data +\
                    np.sqrt(var)*np.random.randn(*(self._solver.net.params[name][0].data.shape))
                print "Added random noise to layer {} with variance {}".format(name, var)

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

    def _load_mapped_params(self, pretrained_model, param_mapping, use_svd):
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


    # snapshot should automatically determine if we are training a multiple round procedure
    def snapshot_name(self, base_iter):
        """ Return the name of the snapshot file """
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX 
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        if self._model_params.num_rounds > 1:
            filename = (self._solver_params.snapshot_prefix + '_' + 'round_{}'.format(self._cur_round) + 
                        infix + '_iter_{:d}'.format(self._solver.iter +
                        base_iter) + '.caffemodel')
        else:
            filename = (self._solver_params.snapshot_prefix + infix +
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

    def _do_improve_model(self):
        """ Create better models with new branches """
        return NotImplementedError

    def _do_train_model(self, max_iters, base_iter):
        """Train the model with iterations=max_iters"""
        return NotImplementedError

    def _do_reinit(self):
        self._solver = self._load_solver(self._solver_params, self._model_params)
        # load pretrained models if provided
        self._load_pretrained_model(self._pretrained_params)      

    def train_model(self, max_iters, base_iter=0):
        """Train the model with iterations=max_iters """
        print 'Solving...'
        while self._cur_round < self._model_params.num_rounds:
            print 'Staring round {}...'.format(self._cur_round)
            # update model
            if self._cur_round > 0:
                self._do_improve_model()
                # re-initialize solver
                self._do_reinit()
                
            self._do_train_model(max_iters, base_iter)
            self._cur_round += 1
        print 'done solving'

class SolverParameter(object):
    """This class represents a solver prototxt file """

    def __init__(self, solver_prototxt=None, path=None, base_lr=0.01, lr_policy="step", 
        gamma=0.1, stepsize=20000, momentum=0.9, weight_decay=0.0005,
        regularization_type="L2", clip_gradients=None, snapshot_prefix='default'):

        assert (path is not None) or (solver_prototxt is not None),\
            'Need to specify either path or solver_prototxt.'

        self._solver = caffe_pb2.SolverParameter()

        if solver_prototxt is not None:
            self._solver_prototxt = solver_prototxt
            with open(solver_prototxt, 'rt') as f:
                pb2.text_format.Merge(f.read(), self._solver)                                   
        elif path is not None:
            self._solver_prototxt = osp.join(path, 'solver.prototxt')
            # update proto object
            self._solver.net = osp.join(path, 'train_val.prototxt')
            self._solver.base_lr = base_lr
            self._solver.lr_policy = lr_policy
            self._solver.gamma = gamma
            self._solver.stepsize = stepsize
            self._solver.momentum = momentum
            self._solver.weight_decay = weight_decay
            self._solver.regularization_type = regularization_type
            self._solver.clip_gradients = clip_gradients
            self._solver.snapshot_prefix = snapshot_prefix
            # caffe solver snapshotting is disabled
            self._solver.snapshot = 0
            # shut down caffe display
            self._solver.display = 0
            # shut down caffe validation
            self._solver.test_iter.append(0)
            self._solver.test_interval = 1000
            if clip_gradients is not None:
                self._solver.clip_gradients = clip_gradients

    def _save_prototxt(self):
        """ Save solver prototxt file to a path """
        solver_path = osp.dirname(self._solver_prototxt)
        if not osp.exists(solver_path):
            os.makedirs(solver_path)
        with open(self._solver_prototxt, 'w') as f:
            f.write(text_format.MessageToString(self._solver))

    @property
    def path(self):
        return osp.dirname(self._solver_prototxt)

    @property
    def snapshot_prefix(self):
        return self._solver.snapshot_prefix    

    def to_proto(self):
        self._save_prototxt()
        return self._solver_prototxt
    
    def set_path(self, path):
        """ set the path of the solver """
        # always ensure the solver and the net definitions are within the same
        # folder
        self._solver_prototxt = osp.join(path, 'solver.prototxt')
        self._solver.net = osp.join(path, 'train_val.prototxt')

class ModelParameter(object):
    """ A bookkeeping class for all branching parameters. """

    def __init__(self, model=None, aff_type=None, num_rounds=1):
        """ Initialize the parameters """

        assert (model is not None) or (num_rounds==1),\
            '"num_rounds" must be 1 (currently {}) when "model" is not specified.'.format(num_rounds)

        assert (model is None) or (aff_type is not None),\
            'Must specify "aff_type" when "model" is None.'

        self._model = model
        self._aff_type = aff_type
        self._num_rounds = num_rounds

    @property
    def model(self):
        return self._model
    
    @property
    def aff_type(self):
        return self._aff_type
    
    @property
    def num_rounds(self):
        return self._num_rounds

class PretrainedParameter(object):
    """ A bookkeeping class for all parameters for loading pre-trained models """

    def __init__(self, pretrained_model, param_mapping=None, param_rand=None, use_svd=True):
        """ Initialize the parameters """

        self._pretrained_model = pretrained_model
        self._param_mapping = param_mapping
        self._param_rand = param_rand
        self._use_svd = use_svd

    def set_param_mapping(self, param_mapping):
        self._param_mapping = param_mapping

    def set_param_rand(self, param_rand):
        self._param_rand = param_rand

    @property
    def pretrained_model(self):
        return self._pretrained_model
        
    @property
    def param_mapping(self):
        return self._param_mapping
        
    @property
    def param_rand(self):
        return self._param_rand
    
    @property
    def use_svd(self):
        return self._use_svd

    def set_param_mapping(self, param_mapping):
        self._param_mapping = param_mapping

    def set_param_rand(self, param_rand):
        self._param_rand = param_rand

if __name__ == '__main__':
    # test by generating a default solver
    sp = SolverParameter()
    sp.to_proto()