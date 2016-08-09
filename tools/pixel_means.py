#!/usr/bin/env python

# ----------------------
# Written by Yongxi Lu
# ----------------------

"""Compute the pixel means of a given dataset, and save the results to a path"""

import _init_paths
from datasets.factory import get_imdb
import numpy as np
import argparse
import sys
import cv2
import cPickle
import os.path as osp

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compute pixel means of imdb')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to compute pixel means',
                        default='celeba_trainval', type=str)
    parser.add_argument('--path', dest='path',
                        help='the path to save the mean file',
                        default=osp.join(osp.dirname(__file__),'..','data','cache'), 
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    imdb = get_imdb(args.imdb_name)
    num_images = imdb.num_images
    
    # means of pixel values, in BGR order
    means = np.zeros((3,))
    num_pixels = 0.0
    
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i)) 
        im_means = im.mean(axis=(0,1))
        im_num_pixels = float(im.shape[0] * im.shape[1])
        means = means * num_pixels / (num_pixels + im_num_pixels) \
            + im_means * im_num_pixels / (num_pixels + im_num_pixels)
        num_pixels = num_pixels + im_num_pixels
        
        if i % 1000 == 0 or i == num_images-1:
            print 'Processing {}/{}, the mean is ({})'.format(i, num_images,means)

    # convert means to (1,1,3) array
    means = means[np.newaxis, np.newaxis, :]

    # save to cache
    mean_file = osp.join(args.path, args.imdb_name+'_mean_file.pkl')
    with open(mean_file, 'wb') as fid:
        cPickle.dump(means, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote mean values to {}'.format(mean_file)