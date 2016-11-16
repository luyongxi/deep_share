#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------


"""Parse a given log, and save to a .mat file. """

import _init_paths
from utils.log import parse_mle
import argparse
import pprint
import numpy as np
import sys, os
import os.path as osp

from scipy.io import savemat

def parse_args():
    """Parse input arguments """
    parser = argparse.ArgumentParser(description="Parse the log file and save it as a .mat file")
    parser.add_argument('--log', dest='log_file',
                        help="name of the log file",
                        default=None, type=str, nargs='*')
    parser.add_argument('--outpath', dest='outpath',
                        help="name of the output path",
                        default='.', type=str)
    parser.add_argument('--split', dest='split',
                        help="the split to parse",
                        default=None, type=str, nargs='*')
    parser.add_argument('--metric', dest='metric',
                        help="the type metric used in evaluation",
                        default='error', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    for log_file in args.log_file:
        params = {}
        filename = osp.splitext(log_file)[0] + '.mat'
        for split in args.split:
            iters, err = parse_mle(log_file, split, args.metric)
            params[split+'iters'] = iters
            params[split+'err'] = err
        savemat(filename, params)