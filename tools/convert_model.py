#!/usr/bin/env python

# ---------------------
# Written by Yongxi Lu
# ---------------------

""" Convert caffemodel into a format that is easier to interpret offline """

import _init_paths
from caffe.proto import caffe_pb2
import numpy as np
import argparse
import cPickle
from scipy.io import savemat
import os.path as osp
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Load caffemodel and save it as a dictionary.")
    parser.add_argument('--caffemodel', dest='caffemodel',
                        help='th caffemodel',
                        default=None, type=str)
    parser.add_argument('--types', dest='layer_types',
                        help="types of layers to save",
                        default=None, type=str, nargs='*')
    parser.add_argument('--output', dest='output',
                        help="name of the output file",
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
    # load files
    print 'Loading caffemodel: {}'.format(args.caffemodel)
    with open(args.caffemodel, 'rb') as f:
        binary_content = f.read()

    protobuf = caffe_pb2.NetParameter()
    protobuf.ParseFromString(binary_content)
    layers = protobuf.layer

    params = {}
    for layer in layers:
        if layer.type in args.layer_types:
            print (layer.name, layer.type)
            params[layer.name+'_w'] = np.reshape(np.array(layer.blobs[0].data), layer.blobs[0].shape.dim) 
            params[layer.name+'_b'] = np.reshape(np.array(layer.blobs[1].data), layer.blobs[1].shape.dim)
            print params[layer.name+'_w'].shape, params[layer.name+'_b'].shape

    # save the layers into a file
    # if the file name is .pkl, save to pickle file.
    # if the file name is .mat, save to mat file.
    # otherwise, report file type not recognized.
    file_type = osp.splitext(args.output)[1]
    if file_type == '.pkl':
        with open(args.output, 'wb') as f:
            cPickle.dump(params, f, cPickle.HIGHEST_PROTOCOL)
            print 'Wrote converted caffemodel to {}'.format(args.output)
    elif file_type == '.mat':
        # for key in params.keys():
        #     params[key.replace('-','_')] = params.pop(key)
        savemat(args.output, params)
        print 'Wrote converted caffemodel to {}'.format(args.output)
    else: 
        print 'The output file type {} is not recognized!'.format(file_type)