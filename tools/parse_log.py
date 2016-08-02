#!/usr/bin/env python

#-------------------------------
# Written by Yongxi Lu
#-------------------------------


"""
Parse a given log
"""

import _init_paths
from utils.log import parse_mle_and_plot
import argparse
import pprint
import numpy as np
import sys, os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a model for Facial Attribute Classification")
    parser.add_argument('--log', dest='log_file',
                        help="name of the log file",
                        default=None, type=str, nargs='*')
    parser.add_argument('--output', dest='output',
                        help="name of the output file",
                        default='loss.png', type=str)
    parser.add_argument('--run', dest='run',
                        help="run length in smoothing the training set",
                        default=50, type=int)
    parser.add_argument('--max_y', dest='max_y',
                        help="maximum value of y axis",
                        default=None, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    if args.log_file is not None:
        parse_mle_and_plot(args.log_file, ['training','validation'], args.output, 
            run_length=args.run, max_y=args.max_y)