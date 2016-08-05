#!/usr/bin/env python

# Written by Yongxi Lu

""" Test clustering of tasks """

import _init_paths
from att.cluster import MultiLabel_CM, ClusterAffinity
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys, os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Test clustering.")
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [None]',
                        default=None, type=int)
    parser.add_argument('--model', dest='model',
                        help='test prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='weights',
                        help='trained caffemodel',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test on',
                        default='celeba_val', type=str)
    parser.add_argument('--method', dest='method',
                        help='similarity matrix used for clustering',
                        default='ecm', type=str)
    parser.add_argument('--cls_id', dest='cls_id',
                        help='comma-separated list of classes to train',
                        default=None, type=str)
    parser.add_argument('--n_cluster', dest='n_cluster',
                        help='number of clusters',
                        default=2, type=int)

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

    print('Using config:')
    pprint.pprint(cfg)

    # set up caffe
    if args.gpu_id is not None:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    # set up the network model
    net = caffe.Net(args.model, args.weights, caffe.TEST)

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for testing'.format(imdb.name)

    # parse class_id if necessary
    if args.cls_id is not None:
        cls_idx = [int(id) for id in args.cls_id.split(',')]
    else:
        cls_idx = None

    if args.method == 'ecm':
        # error correlation matrix
        ecm = MultiLabel_CM(net, imdb=imdb, cls_idx=cls_idx, type='ecm')
        # ClusterAffinity(ecm+1.0, k=args.n_cluster, cls_idx=cls_idx, imdb=imdb)
        ClusterAffinity(abs(ecm), k=args.n_cluster, cls_idx=cls_idx, imdb=imdb)        
    elif args.method == 'lcm':
        # label correlation matrix
        lcm = MultiLabel_CM(net, imdb=imdb, cls_idx=cls_idx, type='lcm')
        # ClusterAffinity(lcm+1.0, k=args.n_cluster, cls_idx=cls_idx, imdb=imdb)
        ClusterAffinity(abs(lcm), k=args.n_cluster, cls_idx=cls_idx, imdb=imdb)