# Written by Yongxi Lu

""" Wrappers for solvers and training hyperparameters """

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb2
import os.path as osp
import os
import caffe
import numpy as np

from utils.config import cfg
from utils.svd import truncated_svd
from utils.somp import somp_cholesky as somp_solve
# from utils.somp import somp_naive as somp_solve

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
        if model_params.max_rounds > 1:
            if self._cur_round > 0:
                solver_path = osp.join(osp.dirname(solver_path), 'round_{}'.format(self._cur_round))
            else:
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
                if (self._cur_round==0) and (pp.fit_params is not None):
                    src_params = self._load_src_params_fit(pp.pretrained_model, pp.fit_params)
                else:
                    src_params = self._load_src_params_plain(pp.pretrained_model)
                self._load_mapped_params(src_params, pp.param_mapping, pp.use_svd)

            if pp.param_rand is not None:
                self._add_noise(pp.param_rand, cfg.TRAIN.NOISE_FACTOR)

    def _add_noise(self, param_rand, factor):
        """ Add random noise to the parameters of layers specfied in param_rand """
        for name in param_rand:
            # heuristic: replace a fraction of the variance with noise
            if factor > 0:
                param_var = np.var(self._solver.net.params[name][0].data)
                var = param_var * factor
                self._solver.net.params[name][0].data[...] =\
                    np.sqrt(1.0-factor)*self._solver.net.params[name][0].data +\
                    np.sqrt(var)*np.random.randn(*(self._solver.net.params[name][0].data.shape))
                print "Added random noise to layer {} with variance {}".format(name, var)

    def _load_src_params_plain(self, pretrained_model):
        """ Load parameters from the source model
            All parameters are saved in a dictionary where
            the keys are the original layer names
        """
        # load pretrained model
        with open(pretrained_model, 'rb') as f:
            binary_content = f.read()

        model = caffe_pb2.NetParameter()
        model.ParseFromString(binary_content)
        layers = model.layer

        src_params = {}

        for lc in layers:
            name = lc.name
            src_params[name] = [np.reshape(np.array(lc.blobs[i].data), lc.blobs[i].shape.dim) for i in xrange(len(lc.blobs))]
            # if len(lc.blobs) >= 2:
                # src_params[name] = [np.reshape(np.array(lc.blobs[0].data), lc.blobs[0].shape.dim), 
                #     np.reshape(np.array(lc.blobs[1].data), lc.blobs[1].shape.dim)]

        return src_params

    def _load_src_params_fit(self, pretrained_model, fit_params):
        """ Load parameters from the source model
            fit_params: an ordered dictionary, the keys are names of the layers, 
                the values are the dimensions. 
        """
        # load pretrained model
        with open(pretrained_model, 'rb') as f:
            binary_content = f.read()

        model = caffe_pb2.NetParameter()
        model.ParseFromString(binary_content)
        layers = model.layer

        src_params = {}
        # record original dimension, and selection
        # indices of the "num" dimension of the last layer
        Ind_last_new_num = None
        N_last_orig_num = None

        for i in xrange(len(fit_params.keys())):
            name = fit_params.keys()[i]
            N_new_num = fit_params.values()[i]
            lc = [l for l in layers if l.name==name][0]            
            src_params[name] = [np.reshape(np.array(lc.blobs[0].data), lc.blobs[0].shape.dim), 
                np.reshape(np.array(lc.blobs[1].data), lc.blobs[1].shape.dim)]
            # save input dimensionality
            orig_shape = src_params[name][0].shape
            N_orig_num, N_orig_channel = orig_shape[0:2]
            # infer the original feature dimension of the current layer
            N_orig_spatial = 1
            if N_last_orig_num is not None:
                N_orig_spatial = N_orig_channel/N_last_orig_num
                N_orig_channel = N_last_orig_num
            # update N_last_orig_num
            N_last_orig_num = N_orig_num
            # input channels to be kept
            if Ind_last_new_num is None:
                # keep all feature dimensions
                Ind_last_new_num = range(N_orig_channel)
            N_new_channel = len(Ind_last_new_num) * N_orig_spatial
            # keep feature dimensions that corresponds to the selected output dimensions of the last layers
            W = src_params[name][0]
            # reshape it into (N_orig_num, N_orig_channel, -1) (a 3-D matrix)
            # then select along the feature dimensions
            W = np.reshape(W, (N_orig_num, N_orig_channel, -1))
            W = np.reshape(W[:, Ind_last_new_num, :], (N_orig_num, -1))
            # check the dimensionality
            assert N_new_num <= N_orig_num, 'The target network should be thinner.'
            # find new shapes of the blob
            new_shape = list(orig_shape)
            new_shape[0] = N_new_num
            new_shape[1] = N_new_channel
            print 'Wide2Thin | {} | Dim: {} -> {}'.format(name, orig_shape, new_shape)
            # find the rows to keep
            if N_new_num < N_orig_num:
                Ind_last_new_num = somp_solve(np.transpose(W), np.transpose(W), N_new_num)
            else:
                Ind_last_new_num = range(N_orig_num)
            # save updated weights and bias terms
            src_params[name][0] = np.reshape(W[Ind_last_new_num], new_shape)
            src_params[name][1] = src_params[name][1][Ind_last_new_num]

        return src_params

    def _load_mapped_params(self, src_params, param_mapping, use_svd):
        """ Load selected parameters specified in param_mapping
            from the parameters of the pretrained model (after wide2fit)

            param_mapping: key is the names in the new model, value
                           is the names in the pretrained model. 
        """
        weight_cache = {}
        for key, value in param_mapping.iteritems():
            # 1-1 matching, direct copy
            if len(key) == 1:
                print 'saving net[{}] <- pretrained[{}]...'.format(key[0], value)
                for i in xrange(len(src_params[value])):
                    self._solver.net.params[key[0]][i].data[...] = src_params[value][i]
                    if key[0] == 'bn_fc15_1_2':
                        print i, src_params[value][i]

                # self._solver.net.params[key[0]][0].data[...] = src_params[value][0]
                # self._solver.net.params[key[0]][1].data[...] = src_params[value][1]
            elif len(key) == 2:
                if use_svd:
                    print 'saving net[{}, {}] <- pretrained[{}] ...'.format(key[0], key[1], value)      
                    # use svd to initialize
                    # W is the weight matrix
                    W = np.reshape(src_params[value][0], (src_params[value][0].shape[0], -1))
                    # size of the target parameters
                    basis_shape = self._solver.net.params[key[0]][0].data.shape
                    linear_shape = self._solver.net.params[key[1]][0].data.shape
                    # perform decomposition, save results. 
                    if weight_cache.has_key(value):
                        B, L = weight_cache[value]
                        print 'using cached svd decomposition of pretrained[{}]...'.format(value)
                    else:
                        B, L = truncated_svd(W, basis_shape[0])
                        weight_cache[value] = (B, L)

                    self._solver.net.params[key[0]][0].data[...] = B.reshape(basis_shape)
                    self._solver.net.params[key[1]][0].data[...] = L.reshape(linear_shape)
                    # use the bias of the original conv filter in the linear combinations
                    self._solver.net.params[key[1]][1].data[...] = src_params[value][1]
                else:
                    print 'use_svd is set to False, skipping net[({}, {})] <- pretrained[{}]'.\
                        format(key[0], key[1], value)

    def snapshot_name(self):
        """ Return the name of the snapshot file 
            automatically determines if we are training a multiple round procedure,
            and adjust naming conventions accordingly.
        """
        if self._model_params.max_rounds > 1:
            filename = ('round_{}'.format(self._cur_round) + '_iter_{:d}'.format(self._solver.iter) + '.caffemodel')
        else:
            filename = ('iter_{:d}'.format(self._solver.iter) + '.caffemodel')
        filename = os.path.join(self._output_dir, filename)

        return filename

    def snapshot(self):
        """ Save a snapshot of the network """
        
        net = self._solver.net

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        # filename should adjust to current iterations
        filename = self.snapshot_name()
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

    def _do_train_model(self, max_iters):
        """Train the model with iterations=max_iters"""
        return NotImplementedError

    def _do_reinit(self):
        self._solver = self._load_solver(self._solver_params, self._model_params)
        # load pretrained models if provided
        self._load_pretrained_model(self._pretrained_params)      

    def train_model(self, max_iters):
        """Train the model with iterations=max_iters """
        # print 'solving'
        is_end = False

        while (not is_end) and (self._cur_round < self._model_params.max_rounds):
            print 'Round {}: Preparing the training procedure...'.format(self._cur_round)
            # train the model, and determine if the training should end
            # after this round.
            use_all_iters = (self._cur_round == self._model_params.max_rounds-1)
            is_end = self._do_train_model(max_iters, use_all_iters)
            if not is_end:
                # update pretrained model name
                self._pretrained_params.\
                    set_pretrained_model(self.snapshot_name())
                # load the newly created model into the training proceudre. 
                self._cur_round += 1
                self._do_reinit()

        # print 'done solving'

class SolverParameter(object):
    """This class represents a solver prototxt file """

    def __init__(self, solver_prototxt=None, path=None, base_lr=0.01, lr_policy="step", 
        gamma=0.1, stepsize=20000, momentum=0.9, weight_decay=0.0005,
        regularization_type="L2", clip_gradients=None):

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

    def __init__(self, model=None, max_rounds=1, max_stall=1000, split_thresh=1.0, error_decay_factor=0.95, 
        branch_depth=1, shrink_factor=2):
        """ Initialize the parameters """

        assert (model is not None) or (max_rounds==1),\
            '"max_rounds" must be 1 (currently {}) when "model" is not specified.'.format(max_rounds)

        self._model = model
        self._max_rounds = max_rounds
        self._max_stall = max_stall
        self._split_thresh = split_thresh
        self._error_decay_factor = error_decay_factor
        self._branch_depth = branch_depth
        self._shrink_factor=shrink_factor

    @property
    def model(self):
        return self._model
       
    @property
    def max_rounds(self):
        return self._max_rounds

    @property
    def max_stall(self):
        return self._max_stall
    
    @property
    def split_thresh(self):
        return self._split_thresh

    @property
    def error_decay_factor(self):
        return self._error_decay_factor

    @property
    def branch_depth(self):
        return self._branch_depth

    @property
    def shrink_factor(self):
        return self._shrink_factor

class PretrainedParameter(object):
    """ A bookkeeping class for all parameters for loading pre-trained models """

    def __init__(self, pretrained_model, fit_params=None, param_mapping=None, param_rand=None, use_svd=True):
        """ Initialize the parameters """

        self._pretrained_model = pretrained_model
        self._fit_paramas = fit_params
        self._param_mapping = param_mapping
        self._param_rand = param_rand
        self._use_svd = use_svd

        # TODO: add the control over types of fitting method
        # We can compare against random selection, and simply leaving
        # the parameters with random initialization. 

    def set_param_mapping(self, param_mapping):
        self._param_mapping = param_mapping

    def set_param_rand(self, param_rand):
        self._param_rand = param_rand

    def set_fit_params(self, fit_params):
        self._fit_params = fit_params

    def set_pretrained_model(self, pretrained_model):
        self._pretrained_model = pretrained_model

    @property
    def fit_params(self):
        return self._fit_params

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