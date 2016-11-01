#!/usr/bin/env python

# Written by Yongxi Lu

"""
Factory method for easily getting imdbs by name.
"""

# import utilities
import sys
import os.path as osp
import numpy.random as npr

if __name__ == '__main__':
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)

    this_dir = osp.dirname(__file__)
    # Add utils to PYTHONPATH
    lib_path = osp.join(this_dir, '..', '..', 'lib')
    add_path(lib_path)
    dataset_path = osp.join(this_dir)
    add_path(dataset_path)

# import datasets
from celeba import CelebA
from celeba_plus_webcam_cls import CelebA_Plus_Webcam_Cls
from IBMattributes import IBMAttributes
from deepfashion import DeepFashion

# dataset functor
__sets = {}

# setup DeepFashion dataset
for split in ['train', 'val', 'test', 'trainval']:
    name = 'deepfashion_{}'.format(split)
    __sets[name] = (lambda split=split:
                    DeepFashion(split))

# setup CelebA dataset
for split in ['train', 'val', 'test', 'trainval']:
    name = 'celeba_{}'.format(split)
    __sets[name] = (lambda split=split:
                    CelebA(split))

# setup CelebA (aligned) dataset
for split in ['train', 'val', 'test', 'trainval']:
    name = 'celeba_{}_align'.format(split)
    __sets[name] = (lambda split=split:
                    CelebA(split, align=True))

# setup CelebA+Webcam dataset
for split in ['train', 'val']:
    name = 'celeba_plus_webcam_cls_{}'.format(split)
    __sets[name] = (lambda split=split:
                    CelebA_Plus_Webcam_Cls(split))

# setup IBMattributes dataset
for split in ['train', 'val']:
    name = 'IBMattributes_{}'.format(split)
    __sets[name] = (lambda split=split:
                    IBMAttributes(split))

def get_imdb(name):
    """ Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """ List all registred imdbs."""
    return __sets.keys()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Usage: ./factory.py imdb-name'
        sys.exit(1)

    imdb = get_imdb(sys.argv[1])
	
    # print out dataset name and confirm the number of classes is correct
    print 'dataset name: {}'.format(imdb.name)
    print 'number of classes {}'.format(imdb.num_classes)
    print 'number of images {}'.format(imdb.num_images)
    print 'cache path: {}'.format(imdb.cache_path)
    print 'data path: {}'.format(imdb.data_path)

    # check few random examples
    idx = npr.choice(imdb.num_images, size=5, replace=False)
    print 'Please check against the dataset to see if the following printed information is correct...'
    for i in idx:
        imdb.print_info(i)
