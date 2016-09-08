#!/usr/bin/env python

# Written by Yongxi Lu

""" Test clustering of tasks """

import _init_paths
from evaluation.cluster import MultiLabel_ECM_cluster
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys, os

import json

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Find clusters.")
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
                        help='the method used for clustering',
                        default='ecm_pos', type=str)
    parser.add_argument('--cls_id', dest='cls_id',
                        help='comma-separated list of classes to test',
                        default=None, type=str)
    parser.add_argument('--n_cluster', dest='n_cluster',
                        help='number of clusters',
                        default=2, type=int)
    parser.add_argument('--mean_file', dest='mean_file',
                        help='the path to the mean file to be used',
                        default=None, type=str)

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

    # set up the network model
    net = caffe.Net(args.model, args.weights, caffe.TEST)

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for testing'.format(imdb.name)

    # parse class_id
    classid_name = os.path.splitext(args.weights)[0] + '.clsid'
    with open(classid_name, 'rb') as f:
        class_id = json.loads(f.read())

    if args.method == 'ecm':
        labels=MultiLabel_ECM_cluster(net, k=args.n_cluster, imdb=imdb, 
            cls_idx=class_id, reverse=False)
    elif args.method == 'ecm_reverse':
        labels=MultiLabel_ECM_cluster(net, k=args.n_cluster, imdb=imdb, 
            cls_idx=class_id, reverse=True)

    for i in xrange(args.n_cluster):
        print 'Cluster {} is: {}'.format(i, [class_id[j] for j in np.where(labels==i)[0]])