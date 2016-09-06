#---------------------------
# Written by Yongxi Lu
#---------------------------

"""Train classifier """

import datasets
from utils.config import cfg
from utils.timer import Timer
from utils.holder import CircularQueue
from utils.branch import aff_mean_KL, aff_max_KL, aff_dot, compute_normalized_cut
from .solver import SolverWrapper
import numpy as np
import os
import caffe

from caffe.proto import caffe_pb2
import google.protobuf as pb2

import json
from sklearn.cluster import SpectralClustering

class ClassificationSW(SolverWrapper):
    """ Wrapper around Caffe's solver """
    
    def __init__(self, imdb, output_dir, solver_params, pretrained_params, 
        model_params, cls_id=None):
        """ Initialize the SolverWrapper. """

        SolverWrapper.__init__(self, imdb, output_dir, 
            solver_params, pretrained_params, model_params)
        # initialize the input layers
        self._old_cls_id = cls_id
        # initialize layers
        self._init_input_layer(imdb, cls_id)
        # initialize variables that records supervision information
        self._init_sup_data()

    def _do_reinit(self):
        SolverWrapper._do_reinit(self)
        # initialize the input layers
        self._init_input_layer(self._imdb, self._cls_id)
        # re-initialize the supervision data
        self._init_sup_data()

    def _init_input_layer(self, imdb, cls_id):
        """ initialize the input layer """
        # load imdb to layers
        self._solver.net.layers[0].set_imdb(imdb['train'])
        if cfg.TRAIN.USE_VAL == True:
            self._solver.test_nets[0].layers[0].set_imdb(imdb['val'])
        # get the number of prediction classes
        self._num_classes = self._solver.net.layers[0].num_classes
        # set class list
        self._cls_id = cls_id
        if cls_id is not None:
            self._solver.net.layers[0].set_classlist(cls_id)
            if cfg.TRAIN.USE_VAL == True:
                self._solver.test_nets[0].layers[0].set_classlist(cls_id)

    def _init_sup_data(self):
        """ Initialize data holder for recording of supervision data """
        model = self._model_params.model
        if model is not None:
            # names of all branches at edges
            names = []
            edges = self._model_params.model.list_edges()
            for e in edges:
                for k in xrange(model.num_branch_at(e[0], e[1])):
                    names.append(model.branch_name_at_i_j_k(e[0], e[1], k))
            # for each name in names, assign a CircularQueue
            self._sup_queue = {}
            for name in names:
                self._sup_queue[name] = CircularQueue(cfg.TRAIN.SUP_LENGTH)

    def _trace_gradient(self, net, name):
        """ Trace the gradient to the basis """
        Y = net.blobs[name][0].diff
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1], -1))
        L = net.params[name][0].data
        L = np.reshape(L, (L.shape[0], -1)).transpose()

        return np.dot(L, Y).transpose((1,0,2))

    def _update_sup_train(self):
        """ update the queue storing the supervision from training data """
        for name, queue in self._sup_queue.iteritems():
            if self._model_params.aff_type == 'output_error':
                queue.append(self._collect_error_info(self._solver.net))
            elif self._model_params.aff_type == 'gradient':
                queue.append(self._trace_gradient(self._solver.net, name))

    def _update_sup_val(self):
        """ update the queue stroing the supervision from validation data """
        for _ in xrange(cfg.TRAIN.SUP_LENGTH):
            if self._model_params.aff_type == 'output_error':
                self._solver.test_nets[0].forward()
                queue.append(self._collect_error_info(self._solver.test_nets[0]))

    def snapshot(self, base_iter):
        """ Save class list to the text file with the same name as caffemodel"""
        caffename = self.snapshot_name(base_iter)
        fn = os.path.splitext(caffename)[0] + '.clsid'
        with open(fn, 'wb') as f:
            f.write(json.dumps(self._cls_id))

        SolverWrapper.snapshot(self, base_iter)

    def _load_layer_at(self, layer, col, expanded=False):
        """Load the model parameters at (layer, col) """
        # find out all the names, save them to a dictionary.
        model = self._model_params.model
        names = []
        if model is None:
            return

        for k in xrange(model.num_branch_at(layer, col)):
            names.append(model.branch_name_at_i_j_k(layer, col, k))
        basis_name = model.col_name_at_i_j(layer, col)

        Xlist = [None for _ in xrange(len(names))]
        # construct a 3D tensor where the linear combination matrices
        # are concantenated along the axis=2.
        for name in names:
            L = self._solver.net.params[name][0].data
            L = np.reshape(L, (L.shape[0], -1))
            if expanded:
                W = self._solver.net.params[basis_name][0].data
                W = np.reshape(W, (W.shape[0], -1))
                L = L*W
            # print 'weights = {}'.format(L)
            # print 'bias = {}'.format(self._solver.net.params[name][1].data)
            Xlist[names.index(name)] = L

        # from Xlist to tensor
        X = np.stack(Xlist, axis=-1)

        return X

    def _load_sup_at(self, layer, col):
        """Load the supervision infos at (layer, col) """
        # find out all the names, save them to a dictionary.
        model = self._model_params.model
        names = []
        if model is None:
            return

        for k in xrange(model.num_branch_at(layer, col)):
            names.append(model.branch_name_at_i_j_k(layer, col, k))

        Xlist = [None for _ in xrange(len(names))]
        # construct a 3D tensor where the linear combination matrices
        # are concantenated along the axis=2.
        for name in names:
            D = self._sup_queue[name].mean()
            Xlist[names.index(name)] = D

        # from Xlist to tensor
        X = np.stack(Xlist, axis=-1)

        return X

    def _compute_affinity(self, layer, col):
        if self._model_params.aff_type == 'linear_basis_mean':
            Le = self._load_layer_at(layer, col, expanded=False)
            return aff_mean_KL(Le)
        elif self._model_params.aff_type == 'linear_basis_max':
            Le = self._load_layer_at(layer, col, expanded=False)
            return aff_max_KL(Le)
        elif self._model_params.aff_type == 'expanded_basis_mean':
            Le = self._load_layer_at(layer, col, expanded=True)
            return aff_mean_KL(Le)
        elif self._model_params.aff_type == 'expanded_basis_max':
            Le = self._load_layer_at(layer, col, expanded=True)
            return aff_max_KL(Le)
        elif self._model_params.aff_type == 'output_error':
            # a function implemented by children classes
            return self._get_output_error_aff(layer, col)
        elif self._model_params.aff_type == 'gradient':
            Fe = self._load_sup_at(layer, col)
            return (aff_basis_max_dot(Fe)+1.0)/2.0

    def _find_best_branch(self):
        """ Find out the best branch """
        edges = self._model_params.model.list_edges()
        # branch utility
        u = np.zeros((len(edges), ))
        # labels
        labels = [[] for _ in xrange(len(edges))] 
        for i in xrange(len(edges)):
            e = edges[i]
            print 'Performing clustering for layer {} branch {}...'.\
                format(*e)
            # compute the affinity matrix
            A, Fs = self._compute_affinity(e[0], e[1])
            # initialize spectral clustering object
            spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack',
                                      affinity="precomputed")
            print 'Fs = {}'.format(Fs)
            print 'A = {}'.format(A)
            # store the predicted labels.
            # handle the trivial case
            if A.shape[0] > 2:
                labels[i] = spectral.fit_predict(A)
            else:
                labels[i] = np.array([0,1])

            # compute the normalized cut as branch cue
            u[i] = compute_normalized_cut(A, labels[i])

        # return br_idx and br_split associated with the branch with smallest Ncut
        imin = np.argmin(u)
        br_idx = edges[imin]
        br_split = [list(np.where(labels[imin]==0)[0]), list(np.where(labels[imin]==1)[0])]
        
        return br_idx, br_split

    def _print_train_val(self, cur_iter):
        """ Print training and validation information"""
        return

    def _do_improve_model(self):
        """ Create better models with new branches """
        br_idx, br_split = self._find_best_branch()
        # idx is a tuple, (layer_idx, col_idx)
        # split is a list of lists, each list is the index into the tops (branches)
        # insert branch. 
        model = self._model_params.model
        pp = self._pretrained_params
        param_mapping, param_rand = model.insert_branch(br_idx, br_split)
        pp.set_param_mapping(param_mapping)
        pp.set_param_rand(param_rand)
        self._cls_id = [self._old_cls_id[t] for t in model.list_tasks()]
        print 'Round {}: Creating new branches at layer {} branch {}...'.\
            format(self._cur_round, *br_idx)
        for i in xrange(len(br_split)):
            print 'Split {}: {}'.format(i, br_split[i])
        for k,v in param_mapping.iteritems():
            print 'Round {}: Net2Net initialization: {} <- {}'.format(self._cur_round, k[0], v)

    def _do_train_model(self, max_iters, base_iter):
        """Train the model with iterations=max_iters"""

        last_snapshot_iter = -1
        timer = Timer()
        while self._solver.iter < max_iters:
            timer.tic()
            self._solver.step(1)
            timer.toc()
            # adjust iteration
            cur_iter = self._solver.iter + base_iter
            # print logging information
            self._print_train_val(cur_iter)
            # collect supervision information
            self._update_sup_train()
            # display information about model improvement
            if (self._model_params.model is not None) and\
                    (cur_iter % cfg.TRAIN.CLUSTER_FREQ == 0) and\
                    self._model_params.num_rounds > 1:
                self._find_best_branch()

            if cur_iter % cfg.TRAIN.TIMER_FREQ == 0:
                # display training speed 
                print 'speed: {:.3f}s /iter'.format(timer.average_time)

            if cur_iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = cur_iter
                self.snapshot(base_iter)

        # save snapshot if we haven't done so
        if last_snapshot_iter != cur_iter:
            self.snapshot(base_iter)


# TODO: the current method in calculating the correlation matrix is simply wrong!
# In fact, the dimensionality is completely not right.
# We need to use the task structure from the model to determine what are the tasks
# associated with each branch. We then need to use that information to determine
# how large the error correlation matrix is.

class MultiLabelSW(ClassificationSW):
    """ Wrapper around Caffe's solver """

    def _collect_error_info(self, net):
        """ provide a matrix summarizing the error info at the current mini-batch 
            For MultiLabel classification, we want to summarize an error vector
            (similar to what we display at the output)
        """

        # TODO: use task list to find out appropriate error vectors. 

        err_vec = net.blobs['error'].data
        err_vec = err_vec*2.0-1.0
        return err_vec

    def _get_output_error_aff(layer, col):
        # error correlation matrix
        Fe = self._load_sup_at(layer, col)
        return aff_basis_max_dot(Fe)

    def _print_train_val(self, cur_iter):
        """ Print training and validation information """

        ClassificationSW._print_train_val(self, cur_iter)

        # evaluate training performance
        err_train = self._solver.net.blobs['error'].data
        loss_train = self._solver.net.blobs['loss'].data
        print 'Iteration {}: training error = {}'.format(cur_iter, err_train.mean())
        print 'Iteration {}: training loss = {}'.format(cur_iter, loss_train)

        err_val = np.zeros((1, self._num_classes))            
        if cur_iter % cfg.TRAIN.VAL_FREQ == 0:
            # perform validation    
            for _ in xrange(cfg.TRAIN.VAL_SIZE):
                self._solver.test_nets[0].forward()
                err_val += self._solver.test_nets[0].blobs['error'].data
            err_val /= cfg.TRAIN.VAL_SIZE
            print 'Iteration {}: validation error = {}'.format(cur_iter, err_val.mean())

class SingleLabelSW(ClassificationSW):
    """ Wrapper around Caffe's solver """

    def __init__(self, imdb, solver_prototxt, output_dir, 
        pretrained_model=None, param_mapping=None, param_rand=None, 
        use_svd=True, cls_id=None):
        """ Initialize the SolverWrapper. """

        if cls_id is not None:
            assert set(cls_id)==set(range(imdb['train'].num_classes)),\
                'Single label classification must use all classes!'

        ClassificationSW.__init__(self, imdb, solver_prototxt, output_dir, 
            pretrained_model, param_mapping, param_rand, use_svd, cls_id)

    def _collect_error_info(self, net):
        """ provide a matrix summarizing the error info at the current mini-batch """
        return NotImplementedError

    def _get_output_error_aff(layer, col):
        # symmetric confusion matrix
        return NotImplementedError

    def _print_train_val(self, cur_iter):
        """ Print training and validation information """
        # evaluate training performance

        ClassificationSW._print_train_val(self, cur_iter)

        acc_train = self._solver.net.blobs['acc'].data
        loss_train = self._solver.net.blobs['loss'].data
        print 'Iteration {}: training accuracy = {}'.format(cur_iter, acc_train.ravel())
        print 'Iteration {}: training loss = {}'.format(cur_iter, loss_train)

        acc_val = np.zeros((1,))            
        if cur_iter % cfg.TRAIN.VAL_FREQ == 0:
            # perform validation    
            for _ in xrange(cfg.TRAIN.VAL_SIZE):
                self._solver.test_nets[0].forward()
                acc_val += self._solver.test_nets[0].blobs['acc'].data
            acc_val /= cfg.TRAIN.VAL_SIZE
            print 'Iteration {}: validation accuracy = {}'.format(cur_iter, acc_val.ravel())