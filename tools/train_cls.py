#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------

"""Train multilabel classifier """

import _init_paths
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
from models.factory import get_models, get_io, get_sw
from solvers.solver import SolverParameter, PretrainedParameter, ModelParameter
import caffe
import argparse
import pprint
import numpy as np
import sys, os
import os.path as osp
import cPickle
import json

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a model for classification")
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [None]',
                        default=None, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--task_name', dest='task_name',
                        help='the name of the task',
                        default='multilabel',type=str)
    parser.add_argument('--mean_file', dest='mean_file',
                        help='the path to the mean file to be used',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--traindb', dest='traindb_name',
                        help='dataset to train on',
                        default='celeba_train', type=str)
    parser.add_argument('--valdb', dest='valdb_name',
                        help='dataset to validate on',
                        default='celeba_val', type=str)
    parser.add_argument('--exp', dest='exp_dir',
                        help='experiment path',
                        default=None, type=str)
    parser.add_argument('--cls_id', dest='cls_id',
                        help='comma-separated list of classes to train',
                        default=None, type=str)
    parser.add_argument('--base_iter', dest='base_iter',
                        help='the base iteration to train',
                        default=0 ,type=int)
    parser.add_argument('--infix', dest='infix',
                        help='additional infix to add',
                        default='',type=str)
    # options concerning solver, will be overriden if --solver is specified. 
    parser.add_argument('--base_lr', dest='base_lr',
                        help='base learning rate',
                        default=0.01,type=float)
    parser.add_argument('--lr_policy', dest='lr_policy',
                        help='learning rate policy',
                        default='step', type=str)
    parser.add_argument('--gamma', dest='gamma',
                        help='gamma in SGD solver',
                        default=0.1, type=float)
    parser.add_argument('--stepsize', dest='stepsize',
                        help='stepsize to change learning rate',
                        default=20000, type=int)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum in SGD solver',
                        default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        default=0.0005, type=float)
    parser.add_argument('--regularization_type', dest='regularization_type',
                        default='L2', type=str)
    parser.add_argument('--clip_gradients', dest='clip_gradients',
                        default=None, type=float)
    parser.add_argument('--snapshot_prefix', dest='snapshot_prefix',
                        default='default', type=str)
    # network model to use, will be overriden if --solver is specified.  
    parser.add_argument('--model', dest='model',
                        default='low-vgg-m', type=str)
    parser.add_argument('--last_low_rank', dest='last_low_rank',
                        help='the last layer to use low-rank factorization',
                        default=0, type=int)
    parser.add_argument('--cut_depth', dest='cut_depth',
                        help='the depth of the cut in the model',
                        default=0, type=int)
    parser.add_argument('--cut_points', dest='cut_points',
                        help='the point in which the model is cut',
                        default=None, type=str)
    parser.add_argument('--use_svd', dest='use_svd',
                        help='use svd to initialize',
                        action='store_true')
    parser.add_argument('--use_mdc', dest='use_mdc',
                        help='use mean deivation consistency to regularize',
                        action='store_true')
    parser.add_argument('--share_basis', dest='share_basis',
                        help="share basis filters", 
                        action='store_true')
    parser.add_argument('--loss', dest='loss',
                        default=None, type=str)
    # arguments related to branching
    parser.add_argument('--num_rounds', dest='num_rounds',
                        help='number of branching rounds in training',
                        default=1, type=int)
    parser.add_argument('--aff_type', dest='aff_type',
                        help='the method used to compute similarity',
                        default='linear_basis_mean', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg_set_path(args.exp_dir)

    # add additional infix if necessary
    cfg.TRAIN.SNAPSHOT_INFIX += args.infix

    # use mean file if provided
    if args.mean_file is not None:
        with open(args.mean_file, 'rb') as fid:
            cfg.PIXEL_MEANS = cPickle.load(fid)
            print 'mean values loaded from {}'.format(args.mean_file)

    print('Using config:')
    pprint.pprint(cfg)

    # set up caffe
    if args.gpu_id is not None:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    traindb = get_imdb(args.traindb_name)
    valdb = get_imdb(args.valdb_name)
    print 'Loaded dataset `{:s}` for training'.format(traindb.name)
    print 'Loaded dataset `{:s}` for validation'.format(valdb.name)

    imdb = {'train': traindb, 'val': valdb}

    output_dir = get_output_dir(traindb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # parse class_id if necessary
    if args.cls_id is not None:
        class_id = json.loads(args.cls_id)
    else:
        class_id = range(imdb['train'].num_classes)

    # parse cut_points if necessary
    cut_points = None
    if args.cut_points is not None:
        cut_points = json.loads(args.cut_points)

    # load pretrained parameters
    pretrained_params = PretrainedParameter(pretrained_model=args.pretrained_model, use_svd=args.use_svd)
    # if solver file is not specified, dynamically generate one based on options. 
    if args.solver is None:
        # io object
        io = get_io(args.task_name, class_list=class_id, loss_layer=args.loss)
        # create solver and model, update parameters. 
        model, param_mapping, new_branches = get_models(args.model, io=io, model_name=args.model, 
            last_low_rank=args.last_low_rank, use_mdc=args.use_mdc,
            share_basis=args.share_basis, cut_depth=args.cut_depth, cut_points=cut_points)
        pretrained_params.set_param_mapping(param_mapping)
        pretrained_params.set_param_rand(new_branches)
        # the orders in class list might shift if a cut is specified. 
        class_id = [class_id[t] for t in model.list_tasks()]
        # get model parameters
        model_params = ModelParameter(model=model, aff_type=args.aff_type, num_rounds=args.num_rounds)
        # save solver parameters
        solver_path = osp.join(output_dir, 'prototxt')
        solver_params = SolverParameter(path=solver_path, base_lr=args.base_lr, lr_policy=args.lr_policy, 
            gamma=args.gamma, stepsize=args.stepsize, momentum=args.momentum, weight_decay=args.weight_decay, 
            regularization_type=args.regularization_type, clip_gradients=args.clip_gradients, snapshot_prefix=args.snapshot_prefix)
    else:
        # get model parameters
        model_params = ModelParameter()
        # get solver parameters
        solver_params = SolverParameter(solver_prototxt=args.solver)


    print "Class list: {}".format(class_id)
    sw = get_sw(args.task_name, imdb=imdb, output_dir=output_dir, 
        solver_params=solver_params, pretrained_params=pretrained_params, 
        model_params=model_params, cls_id=class_id)
    sw.train_model(args.max_iters, args.base_iter)