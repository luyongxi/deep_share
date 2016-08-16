#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------

"""Train N-way classifier """

import _init_paths
from solvers.singlelabel_sw import SingleLabelSW
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
from models.factory import get_models, get_models_dir
from solvers.solver import SolverParameter
from models.model_io import MultiLabelIO
import caffe
import argparse
import pprint
import numpy as np
import sys, os
import os.path as osp
import cPickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a model for N-way classification")
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
                        default='celeba_plus_webcam_cls_train', type=str)
    parser.add_argument('--valdb', dest='valdb_name',
                        help='dataset to validate on',
                        default='celeba_plus_webcam_cls_val', type=str)
    parser.add_argument('--exp', dest='exp_dir',
                        help='experiment path',
                        default=None, type=str)
    parser.add_argument('--base_iter', dest='base_iter',
                        help='the base iteration to train',
                        default=0 ,type=int)
    parser.add_argument('--infix', dest='infix',
                        help='additional infix to add',
                        default='',type=str)

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

    sw = SingleLabelSW(imdb, args.solver, output_dir, args.pretrained_model)
    sw.train_model(args.max_iters, args.base_iter)