#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------

"""Train with branching for multilabel classificaiton """

import _init_paths
from solvers.multilabel_sw import MultiLabelSW
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
from models.factory import get_models
from solvers.solver import SolverParameter
from models.model_io import MultiLabelIO
from models.branch import branch_linear_combination
import caffe
import argparse
import pprint
import numpy as np
import sys, os
import os.path as osp
import cPickle
import json

def parse_args():
    """Parse input arguments """
    parser = argparse.ArgumentParser(description="Train a model for Multilabel Classification")
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
    parser.add_argument('--first_low_rank', dest='first_low_rank',
                        help='the first layer to use low-rank factorization',
                        default=0, type=int)
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
                        default='Sigmoid', type=str)
    # models related to branching
    parser.add_argument('--num_rounds', dest='num_rounds',
                        help='number of branching rounds in training',
                        default=1, type=int)
    # parser.add_argument('--snapshot_file', dest='snapshot_file',
    #                     help='the file containing snapshot information used to resume training', 
    #                     default=None, type=str)

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
        # class_id = [int(id) for id in args.cls_id.split(',')]
        class_id = json.loads(args.cls_id)
    else:
        class_id = range(imdb['train'].num_classes)
    old_class_id = class_id

    # if a solver file is already specified, the training should have only one round. 
    assert (args.solver is None) or (args.num_rounds==1),\
        'Preloaded solver does not support branching'.format()

    solver_file = args.solver
    cur_pretrained_model = args.pretrained_model
    for train_round in xrange(args.num_rounds):
        # if solver file is not specified, dynamically generate one based on options.
        param_mapping = None
        if args.solver is None:
            # io object
            io = MultiLabelIO(class_list=class_id, loss_layer=args.loss)
            # paths, model names and snapshot prefix should clearly specify the rounds
            path = osp.join(output_dir, 'prototxt', 'round_{}'.format(train_round))
            model_name = args.model + '_' + 'round_{}'.format(train_round)
            snapshot_prefix = args.snapshot_prefix + '_' + 'round_{}'.format(train_round)
            # in the first round, we need to create new model
            # in the following rounds, we need to insert branches. 
            if train_round == 0:
                print 'Round {}: Model initialization...'.format(train_round)
                model, param_mapping = get_models(args.model, io=io, model_name=model_name, 
                    path=path, first_low_rank=args.first_low_rank, use_mdc=args.use_mdc,
                    share_basis=args.share_basis)
            else:
                br_idx, br_split = branch_linear_combination(cur_pretrained_model, model)
                # # Make a decision to create k branches at a (layer, col)
                # br_idx, br_split = branch_at_round(cur_pretrained_model, model, 
                #     layer=model.num_layers-train_round+1, col=0, k=args-num_rounds-train_round+1)
                # Update path and model names
                model.set_path(path)
                model.set_name(model_name)
                # idx is a tuple, (layer_idx, col_idx)
                # split is a list of lists, each list is the index into the tops (branches)
                # insert branch. 
                param_mapping = model.insert_branch(br_idx, br_split)
                class_id = [old_class_id[t] for t in model.list_tasks()]
                print 'Round {}: Creating new branches at layer {} branch {}...'.\
                    format(train_round, *br_idx)
                for i in xrange(len(br_split)):
                    print 'Split {}: {}'.format(i, br_split[i])
                for k,v in param_mapping.iteritems():
                    print 'Round {}: Net2Net initialization: {} <- {}'.format(train_round, k[0], v)
            # generate solver
            solver = SolverParameter(path, base_lr=args.base_lr, lr_policy=args.lr_policy, 
                gamma=args.gamma, stepsize=args.stepsize, momentum=args.momentum, weight_decay=args.weight_decay, 
                regularization_type=args.regularization_type, clip_gradients=args.clip_gradients, snapshot_prefix=snapshot_prefix)
            # save files
            model.to_proto(deploy=False)
            model.to_proto(deploy=True)
            solver.to_proto()
            solver_file = solver.solver_file()

            print 'Model files saved at {}'.format(model.path)

        print 'Round {}: Current class list: {}'.format(train_round, class_id)
        sw = MultiLabelSW(imdb, solver_file, output_dir, cur_pretrained_model, param_mapping, args.use_svd, class_id)
        sw.train_model(args.max_iters, args.base_iter)
        cur_pretrained_model = sw.snapshot_name(args.base_iter)
