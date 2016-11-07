#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------

"""Load soft labels for the PersonAttributes dataset"""

import _init_paths
from utils.config import cfg, cfg_from_file, cfg_set_path, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import caffe
import sys, os
import json

from evaluation.test import eval_and_save

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
    parser.add_argument('--model_face', dest='model_face',
                        help='prototxt for face model',
                        default=None, type=str)
    parser.add_argument('--weights_face', dest='weights_face',
                        help='caffemodel for face model',
                        default=None, type=str)
    parser.add_argument('--model_clothes', dest='model_clothes',
                        help='prototxt for clothes model',
                        default=None, type=str)
    parser.add_argument('--weights_clothes', dest='weights_clothes',
                        help='caffemodel for clothes model',
                        default=None, type=str)
    parser.add_argument('--imdb_face', dest='imdb_face',
                        help='dataset to evaluate the face model',
                        default='person_clothes_train', type=str)
    parser.add_argument('--imdb_clothes', dest='imdb_clothes',
                        help='dataset to evaluate the clothes model',
                        default='person_face_train', type=str)
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

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    # save soft labels for face model if the file does not already exists
    imdb = get_imdb(args.imdb_face)
    fn = os.path.join(imdb.data_path, imdb.name+'.pkl')
    if not os.path.exists(fn):
        net = caffe.Net(args.model_face, args.weights_face, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(args.weights_face))[0]
        # parse class_id
        classid_name = os.path.splitext(args.weights_face)[0] + '.clsid'
        with open(classid_name, 'rb') as f:
            class_id = json.loads(f.read())

        eval_and_save(net, imdb, class_id)
    else:
        print '{} already exists!'.format(fn)

    # save soft labels for face model
    imdb = get_imdb(args.imdb_clothes)
    fn = os.path.join(imdb.data_path, imdb.name+'.pkl')    
    if not os.path.exists(fn):
        net = caffe.Net(args.model_clothes, args.weights_clothes, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(args.weights_clothes))[0]
        # parse class_id
        classid_name = os.path.splitext(args.weights_clothes)[0] + '.clsid'
        with open(classid_name, 'rb') as f:
            class_id = json.loads(f.read())
        eval_and_save(net, imdb, class_id)
    else:
        print '{} already exists!'.format(fn)
