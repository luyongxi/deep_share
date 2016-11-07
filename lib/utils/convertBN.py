# Written by Yongxi Lu

""" Transform a model that have BN layers, and subsume the normalization into the layers themselves."""


from caffe.proto import caffe_pb2
import numpy as np
import cPickle
from scipy.io import savemat
import os.path as osp
from utils.config import cfg

def convertBN(inmodel, outmodel):
    """ subsume all the BN layers inside inmode to normal layers in the out model """

    # load files
    print 'Loading caffemodel: {}'.format(inmodel)
    with open(inmodel, 'rb') as f:
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

    print 'Saving caffemodel: {}'.format(outmodel)
    with open(outmodel, 'wb') as f:
        f.write(new_binary_content)