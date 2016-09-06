#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------

"""Collect labels from individual attributes to be used as soft labels"""

import _init_paths
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import caffe
import sys, os
import json
from evaluation.test import save_softlabels

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Collect evaluation results as soft labels.")
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [None]',
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='datasets to test on',
                        default='celeba_test', type=str)
    parser.add_argument('--mean_file', dest='mean_file',
                        help='the path to the mean file to be used',
                        default=None, type=str)
    parser.add_argument('--test_class', dest='test_class',
                        help='the index to the soft labels requested',
                        default=None, type=int)

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

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    # get imdb
    imdb = get_imdb(args.imdb_name)
    # iterate over classes
    if args.test_class is None:
        class_list = xrange(imdb.num_classes)
    else:
        class_list = [args.test_class]
    for c in class_list:
        # get file name
        score_file = imdb.score_file_name(c)
        # skip if there is no reason for this purpose.
        if os.path.exists(score_file):
            print 'Skipping saving soft labels for {}: {} already exists...'.\
                format(imdb.classes[c], score_file)
            continue

        # find out the caffemodels [caffemodel, score_name, score_idx]
        src_name, labeler = imdb.find_labeler(c)
        weights = labeler[0]
        prototxt = os.path.splitext(labeler[0])[0] + '.prototxt'
        # caffemodel is the first entry
        net = caffe.Net(prototxt, weights, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(weights))[0]
        # list of images that need to be evaluated
        image_list = imdb.image_path_at_inds(imdb.list_incomplete(c))
        print 'Start saving soft labels for {} from {}, using layer {} index {}'.\
            format(imdb.classes[c], weights, labeler[1], labeler[2])
        save_softlabels(net, image_list, score_file, labeler)
