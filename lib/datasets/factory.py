#!/bin/bash/env python

# Written by Yongxi Lu

"""
Factory method for easily getting imdbs by name.
"""

# import datasets
from celeba import CelebA

# import utilities
import sys
import numpy.random as npr

# dataset functor
__sets = {}

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
    print 'cache path: {}'.format(imdb.cache_path)
    print 'data path: {}'.format(imdb.data_path)

    # check few random examples
    idx = npr.choice(imdb.num_images, size=5, replace=False)
    print 'Please check against the dataset to see if the following printed information is correct...'
    for i in idx:
        imdb.print_info(i)	
