#---------------------------
# Written by Yongxi Lu
#---------------------------

"""Train classifier """

import datasets
from utils.config import cfg
from utils.timer import Timer
# from utils.holder import CircularQueue
from numpy.linalg import norm
from sklearn.metrics.pairwise import pairwise_kernels
from .solver import SolverWrapper
import numpy as np
import os
import os.path as osp
import caffe
from scipy.io import savemat

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
        self._cur_iter = 0
        # initialize the input layers
        self._old_cls_id = cls_id
        # initialize layers
        self._init_input_layer(imdb, cls_id)
        # initialize variables that records supervision information
        self._init_error_data()
        # initialize the training phase
        self._first_round = True

    def _do_reinit(self):
        SolverWrapper._do_reinit(self)
        # initialize the input layers
        self._init_input_layer(self._imdb, self._cls_id)

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

    def _init_error_data(self):
        """ Initialize data holder for recording of error data """
        self._err_div_factor = cfg.EPS
        N = self._num_classes
        # mean error rate
        self._err_mean = np.ones((N, ))
        # mean error margin
        self._margin_mean = np.zeros((N, ))
        # difficult probability
        # self._diff_prob = np.ones((N, ))
        # joint difficult probability
        self._joint_diff_prob = np.zeros((N, N))
        # joint easy probability
        self._joint_easy_prob = np.zeros((N, N))
        # error correlation
        self._err_corr = np.zeros((N, N))

    def _save_error_data(self):
        """ Save the error vectors as a .mat file that can be easily analyzed on MATLAB """
        fname = osp.splitext(self.snapshot_name())[0] + '.err'

        err_info = {}
        err_info['err_div_factor'] = self._err_div_factor
        err_info['err_mean'] = self._err_mean
        err_info['margin_mean'] = self._margin_mean
        # err_info['diff_prob'] = self._diff_prob
        err_info['joint_diff_prob'] = self._joint_diff_prob
        err_info['joint_easy_prob'] = self._joint_easy_prob
        err_info['err_corr'] = self._err_corr

        savemat(fname, err_info)

    def _update_error_data(self):
        """ incrementally add the current training mini-batch to the values """

        # decay factor
        error_decay_factor = self._model_params.error_decay_factor

        # the indexing is always in the order of the old_cls_id
        match_idx = [self._cls_id.index(idx) for idx in self._old_cls_id]
        cur_err_vec = self._collect_error(self._solver.net)[:, match_idx].astype(np.float64)

        # update mean training error
        cur_err_mean = np.nanmean(cur_err_vec>=0.5, axis=0)
        # update mean margin
        cur_margin_mean = np.nanmean(cur_err_vec, axis=0)
        # update difficulty probabilities
        cur_diff_vec = (cur_err_vec>=self._margin_mean).astype(np.float64)
        # cur_diff_prob = np.nanmean(cur_diff_vec, axis=0)
        cur_joint_diff_prob = np.dot(np.transpose(cur_diff_vec), cur_diff_vec)/cur_diff_vec.shape[0]
        cur_joint_easy_prob = np.dot(np.transpose(1.0-cur_diff_vec), 1.0-cur_diff_vec)/cur_diff_vec.shape[0]

        sum_err_mean = self._err_div_factor * error_decay_factor * self._err_mean + cur_err_mean
        sum_margin_mean = self._err_div_factor * error_decay_factor * self._margin_mean + cur_margin_mean
        # sum_diff_prob = self._err_div_factor * error_decay_factor * self._diff_prob + cur_diff_prob
        sum_joint_diff_prob = self._err_div_factor * error_decay_factor * self._joint_diff_prob + cur_joint_diff_prob
        sum_joint_easy_prob = self._err_div_factor * error_decay_factor * self._joint_easy_prob + cur_joint_easy_prob        
        # update estimates
        self._err_div_factor *= error_decay_factor
        self._err_div_factor += 1
        self._err_mean = sum_err_mean / self._err_div_factor
        self._margin_mean = sum_margin_mean / self._err_div_factor
        # self._diff_prob = sum_diff_prob / self._err_div_factor
        self._joint_diff_prob = sum_joint_diff_prob / self._err_div_factor
        self._joint_easy_prob = sum_joint_easy_prob / self._err_div_factor

        # # C(Xi, Xj) = (P(Xi, Xj)/P(Xi) + P(Xi, Xj)/P(Xj))/2 = (P(Xi|Xj) + P(Xj|Xi))/2
        # self._err_corr = (self._joint_diff_prob/self._diff_prob + np.transpose(self._joint_diff_prob/self._diff_prob))/2
        # probablity of jointly difficult + probably of joint easy
        self._err_corr = self._joint_easy_prob + self._joint_diff_prob

    def snapshot(self):
        """ Save class list to the text file with the same name as caffemodel, 
            and then save the error correlation information. 
        """
        caffename = self.snapshot_name()
        fn = os.path.splitext(caffename)[0] + '.clsid'
        with open(fn, 'wb') as f:
            f.write(json.dumps(self._cls_id))

        self._save_error_data()

        SolverWrapper.snapshot(self)

    def _find_partition(self, layer, col, max_col):
        """ Find the partition at the particular layer and col 
            This procedure tests partition with different number of 
            clusters, and outputs the one that strikes a best tradeoff
            between model complexity and separability of less related tasks.
        """

        model = self._model_params.model
        task_list_e = model.tasks_at(layer, col)
        num_branches = len(task_list_e)

        # compute error correlation matrix for the group of tasks.
        A = np.zeros((num_branches, num_branches))
        for j in xrange(num_branches):
            t_j = task_list_e[j]
            for k in xrange(num_branches):
                if k != j:
                    t_k = task_list_e[k]
                    # find the average of correlation between tasks from the two groups
                    A[j,k] = np.mean(np.min(self._err_corr[t_j, :][:, t_k], axis=1))
                    A[k,j] = np.mean(np.min(self._err_corr[t_k, :][:, t_j], axis=1))
                elif k==j:
                    A[j,j] = np.mean(np.min(self._err_corr[t_j, :][:, t_j], axis=1))

        A = (A + np.transpose(A))/2.0

        # print 'A={}'.format(A)

        labels = [None for _ in xrange(1, min(max_col, num_branches)+1)]
        loss = np.zeros((min(max_col, num_branches), ))
        alpha = self._model_params._split_thresh
        for C in xrange(1, min(max_col, num_branches)+1):
            # C == 1 is an exception, because we already know the label
            if C == 1:
                labels[0] = np.zeros((num_branches,))
            elif C==num_branches:
                labels[C-1] = np.arange(num_branches)
            else:
                spectral = SpectralClustering(n_clusters=C, eigen_solver='arpack',
                                          affinity="precomputed")
                # repeat the clustering procedure to reduce randomness
                sep_cost = np.inf
                for _ in xrange(cfg.TRAIN.CLUSTER_REP):
                    labels_this = spectral.fit_predict(A)
                    sep_cost_this = 0.0
                    for j in xrange(C):
                        t_j = np.where(labels_this==j)[0]
                        sep_cost_this += (1.0-np.mean(np.min(A[t_j, :][:, t_j], axis=1)))/C
                    if sep_cost_this <= sep_cost:
                        sep_cost = sep_cost_this
                        labels[C-1] = labels_this
                # add layer-wise cost
            loss[C-1] += (C-1) * model.col_cost_at(layer)

            # add separation loss, split_thresh is a hyperparameter that control
            # how important is maintaining separation compared to reducing complexity.
            for j in xrange(C):
                t_j = np.where(labels[C-1]==j)[0]
                loss[C-1] += alpha * (1.0-np.mean(np.min(A[t_j, :][:, t_j], axis=1)))/C
            
            # display cost information
            print 'C={}: Total loss={}, the complexity cost is {}.'.\
                format(C, loss[C-1], (C-1) * model.col_cost_at(layer))
                
        # find out the minimum cost partition
        Cmin = np.argmin(loss) + 1

        partition = [list(np.where(labels[Cmin-1]==c)[0]) for c in xrange(Cmin)]

        return partition

    def _improve_model(self):
        """ Make the current model larger by creating columns at lower layers
            Return "model_improved"
        """
        model = self._model_params.model
        if model is None:
            return False

        if self._first_round == True:
            # count the number of iterations for this round
            self._stall_count = 0
            # no longer the first round afterwards
            self._first_round = False

        self._stall_count += 1

        if self._model_params.max_stall <= self._stall_count:
            self._stall_count = 0            
            active_layer = model.num_layers - self._cur_round
            # cannot create a branch when the active layer is already 0.
            if active_layer == 0:
                return False

            partition = self._find_partition(active_layer, 0, model.max_cols_at(active_layer))
            if len(partition) > 1:
                param_mapping, param_rand = model.insert_branch((active_layer, 0), partition)
            else:
                return False

            # set information for re-initialization
            pp = self._pretrained_params
            # set new param mappings and param_rand for Net2Net initialization
            pp.set_param_mapping(param_mapping)
            pp.set_param_rand(param_rand)
            # update the class list to the new task list
            self._cls_id = [self._old_cls_id[t] for t in model.list_tasks()]
            # visually show the parameter matchings
            for k,v in param_mapping.iteritems():
                print 'Round {}: Net2Net initialization: {} <- {}'.format(self._cur_round, k[0], v)
            # display current network structure relative to the tasks
            print 'Round {}: Showing network structures'.format(self._cur_round+1)
            task_names = [self._imdb['train'].classes[cls_id] for cls_id in self._old_cls_id]
            model.display_net(task_names)
 
            self._corr_layer = active_layer - 1

            return True

        return False

    def _print_train_val(self):
        """ Print training and validation information"""
        return

    def _do_train_model(self, max_iters, use_all_iters):
        """ Train the model with iterations=max_iters
            Return "is_end" 
        """

        last_snapshot_iter = -1
        timer = Timer()
        while self._solver.iter < max_iters:

            # perform a graident update
            timer.tic()
            self._solver.step(1)
            timer.toc()
            self._cur_iter += 1

            # update error data
            self._update_error_data()
            # print out training and validation information
            self._print_train_val()

            if (not use_all_iters) and self._improve_model():
                if last_snapshot_iter != self._solver.iter:
                    self.snapshot()
                return False

            if self._solver.iter % cfg.TRAIN.TIMER_FREQ == 0:
                # display training speed 
                print 'speed: {:.3f}s /iter'.format(timer.average_time)

            if self._solver.iter % cfg.TRAIN.CORR_FREQ == 0:
                self._save_error_data()

            if self._solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self._solver.iter
                self.snapshot()

        # save snapshot if we haven't done so
        if last_snapshot_iter != self._solver.iter:
            self.snapshot()

        return True

class MultiLabelSW(ClassificationSW):
    """ Wrapper around Caffe's solver """

    def _collect_error(self, net):
        """ Collect error, each row is a data point. """
        return net.blobs['error'].data

    def _print_train_val(self):
        """ Print training and validation information """

        ClassificationSW._print_train_val(self)

        cur_iter = self._cur_iter
        cur_round = self._cur_round

        # display training errors
        if cur_iter % cfg.TRAIN.TRAIN_FREQ == 0:
            err_train = self._err_mean
            print 'Round {}, Iteration {}: training error = {}'.format(cur_round, cur_iter, err_train.mean())
            # if self._model_params.model is not None:
            #     print 'err_corr: {}'.format(self._err_corr)

        # display validation errors
        if cur_iter % cfg.TRAIN.VAL_FREQ == 0:
            # perform validation
            err_val = np.zeros((cfg.TRAIN.VAL_SIZE * cfg.TRAIN.IMS_PER_BATCH, self._num_classes))
            for i in xrange(cfg.TRAIN.VAL_SIZE * cfg.TRAIN.IMS_PER_BATCH):
                self._solver.test_nets[0].forward()
                err_val[i,:] = (self._solver.test_nets[0].blobs['error'].data > 0.5)
            err_val = np.nanmean(err_val, axis=0)
            print 'Round {}, Iteration {}: validation error = {}'.format(cur_round, cur_iter, np.nanmean(err_val))

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

    def _collect_error(self, net):
        """ Collect error vector """
        acc_vec = net.blobs['acc'].data
        return 1.0-acc_vec

    def _print_train_val(self):
        """ Print training and validation information """
        # evaluate training performance

        ClassificationSW._print_train_val(self)

        cur_iter = self._cur_iter
        cur_round = self._cur_round

        # display training errors
        if cur_iter % cfg.TRAIN.TRAIN_FREQ == 0:
            err_train = self._err_mean
            print 'Round {}, Iteration {}: training error = {}'.format(cur_round, cur_iter, err_train.mean())

        # display validation errors
        if cur_iter % cfg.TRAIN.VAL_FREQ == 0:
            # perform validation
            err_val = np.zeros((cfg.TRAIN.VAL_SIZE * cfg.TRAIN.IMS_PER_BATCH, ))
            for i in xrange(cfg.TRAIN.VAL_SIZE * cfg.TRAIN.IMS_PER_BATCH):
                self._solver.test_nets[0].forward()
                err_val[i,:] = 1.0 - self._solver.test_nets[0].blobs['acc'].data
            err_val = np.nanmean(err_val, axis=0)
            print 'Round {}, Iteration {}: validation error = {}'.format(cur_round, cur_iter, np.nanmean(err_val))