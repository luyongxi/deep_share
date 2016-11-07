#!/usr/bin/env python

# ---------------------
# Written by Yongxi Lu
# ---------------------

""" For a caffemodel with BN layers, modify the filter weights and bias so that
    all BN layers can be removed without affecting the performance at deployment time. 
"""

import _init_paths
from caffe.proto import caffe_pb2
import numpy as np
import argparse
import cPickle
from scipy.io import savemat
import os.path as osp
import sys

from utils.config import cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Convert the caffemodel if there is any BN layers.")
    parser.add_argument('--inmodel', dest='inmodel',
                        help='the input caffemodel',
                        default=None, type=str)
    parser.add_argument('--outmodel', dest='outmodel',
                        help='the output caffemodel',
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
    print 'Loading caffemodel: {}'.format(args.inmodel)
    with open(args.inmodel, 'rb') as f:
        binary_content = f.read()

    protobuf = caffe_pb2.NetParameter()
    protobuf.ParseFromString(binary_content)
    layers = protobuf.layer

    _eps = 1e-5
    for layer in layers:
        if layer.type == 'BatchNorm':
            # the layer to be modified. 
            layer_c = [l for l in layers if l.name == layer.name[3:]][0]
            # the parameters fo the computational layer
            w = np.reshape(np.array(layer_c.blobs[0].data), layer_c.blobs[0].shape.dim) 
            b = np.reshape(np.array(layer_c.blobs[1].data), layer_c.blobs[1].shape.dim)
            # load the BN parameters
            factor = 0 if np.array(layer.blobs[2].data) == 0 else 1./np.array(layer.blobs[2].data)
            mean = np.array(layer.blobs[0].data) * factor
            var = np.array(layer.blobs[1].data) * factor

            # display information
            print 'Modifying layer {} based on information from {}'.format(layer_c.name, layer.name)
            # update weights
            if len(w.shape) == 4: 
                w /= (_eps + np.sqrt(var)[:, np.newaxis, np.newaxis, np.newaxis])
            elif len(w.shape) == 2:
                w /= (_eps + np.sqrt(var)[:, np.newaxis])
            # update bias
            b -= mean
            b /= (_eps + np.sqrt(var))
            # save the changes back to the model
            del layer_c.blobs[0].data[:]
            del layer_c.blobs[1].data[:]
            layer_c.blobs[0].data.extend(w.flatten().tolist())
            layer_c.blobs[1].data.extend(b.flatten().tolist())

    # save the model to out model
    new_binary_content = protobuf.SerializeToString()

    print 'Saving caffemodel: {}'.format(args.outmodel)
    with open(args.outmodel, 'wb') as f:
        f.write(new_binary_content)