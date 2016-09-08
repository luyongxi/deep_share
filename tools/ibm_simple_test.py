#!/usr/bin/env python

# Written by Yongxi Lu

""" A lightweight testing procedure to evaluate the performance of
    individual attributes, exclusively for the IBMattributes datasest. 
"""

import _init_paths
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from utils.blob import im_list_to_blob
from layers.multilabel_err import compute_mle
import argparse
import pprint
import caffe
import sys, os
import os.path as osp
import numpy as np
import cPickle

def parse_args():
    """
    Parse input arguments
    """
    DATA_DIR = osp.join(osp.dirname(__file__), '..', 'data')

    parser = argparse.ArgumentParser(description="Test a caffemodel on IBM attributes dataset.")
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
    parser.add_argument('--mean_file', dest='mean_file',
                        help='the path to the mean file to be used',
                        default=None, type=str)
    parser.add_argument('--base_folder', dest='base_folder',
                        help='the datapath to the root folder for the images',
                        default=osp.join(DATA_DIR,'imdb_IBMAttributes', 'ValidationData'), type=str)
    parser.add_argument('--folders', dest='folders',
                        help='the folders containing positive examples',
                        default=None, type=str, nargs='*')

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
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.weights))[0]

    image_list = []
    labels = np.zeros((0, ), dtype=np.int64)
    # number of classes
    num_classes = len(args.folders)
    for c in xrange(num_classes):
        src_folder = osp.join(args.base_folder, args.folders[c])
        new_images = [osp.join(src_folder, fn) for fn in os.listdir(src_folder)]
        image_list.extend(new_images)
        new_labels = c*np.ones((len(new_images),), dtype=np.int64)
        labels = np.hstack((labels, new_labels))

    # number of images
    num_images = len(image_list)
    # init error vector
    err = np.zeros((num_images, num_classes)) # in {0,1} format
    for i in xrange(num_images):
        # prepare blobs 
        label_name = "prob"
        fn = image_list[i]
        data = im_list_to_blob([fn], cfg.PIXEL_MEANS, cfg.SCALE)
        net.blobs['data'].reshape(*(data.shape))
        # forward the network
        blobs_out = net.forward(data=data.astype(np.float32, copy=False))
        # get results
        scores = blobs_out[label_name]
        # evaluate the scores
        score_max = np.argmax(scores, axis=1)
        pred = np.zeros((1, num_classes))
        pred[:, score_max] = 1.0
        target = np.zeros((1, num_classes))
        target[:, labels[i]] = 1.0 
        err[i,:] = compute_mle(pred, target)

        # print infos
        print 'Image {}/{}.'.format(i, num_images)

    # print out basic dataset information
    print '---------------------------------------------------------------'
    print '!!! Summary of results.'
    # get error for each class
    class_names = args.folders
    mean_err = np.mean(err, axis=0)
    for i in xrange(len(class_names)):
        print '!!! Error rate for class {} is: {}'.\
            format(class_names[i], mean_err[i])

    print '!!! The average error rate is {}.'.format(mean_err.mean())
    print '---------------------------------------------------------------'
