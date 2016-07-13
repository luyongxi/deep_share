# Written by Yongxi Lu

"""
Configuration files used in the system

This file specifies default config options
"""

import os
import os.path as osp
import numpy as np
import cPickle
# `pip install easydict` if package not found
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
# 	from utils import cfg
cfg = __C

## ------------------------------------------------------------------------------
#  Options for training
## ------------------------------------------------------------------------------
__C.TRAIN = edict()

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# Perform validation or not
__C.TRAIN.USE_VAL = True

# Sample size for validation (times mini-batch size)
__C.TRAIN.VAL_SIZE = 100

# Sample frequency for validation (number of training iterations)
__C.TRAIN.VAL_FREQ = 400

# Infix to yield te path
__C.TRAIN.SNAPSHOT_INFIX = ''

# Minibatch sizes
__C.TRAIN.IMS_PER_BATCH = 32


## ------------------------------------------------------------------------------

## ------------------------------------------------------------------------------
# Options used in testing
## ------------------------------------------------------------------------------
__C.TEST = edict()

## ------------------------------------------------------------------------------

# standard size uesd in training and testing
__C.SCALE = 224

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

def get_output_dir(imdb, net):
    """ Return the directory that stores experimental results """

    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    """ Merge config dictionary a into dictionary b, clobbering the 
        options in b whenever they are also specified in a.
    """

    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))
   
        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
  
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
  
    _merge_a_into_b(yaml_cfg, __C)

def cfg_set_path(exp_dir):
    """ Set experiment paths """
    if exp_dir is None:
        __C.EXP_DIR = 'default'
    else:
        __C.EXP_DIR = exp_dir

def cfg_print_info():

    def print_values(a, root):
        for k, v in a.iteritems():
            if type(v) is edict:
                print_values(v, k)
            else:
                print '{}->{} is {}, its value is {}'.format(root, k, type(v), v)

    print_values(__C, 'cfg')

if __name__ == '__main__':
  
    print 'The default of the dictionary'
    cfg_print_info()    

