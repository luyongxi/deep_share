""" Person attribute datasets
    It consists of images from CelebA and DeepFashion dataset. 
"""

# import base class
from imdb import Imdb
# import children dataset interfaces
from celeba import CelebA
from deepfashion import DeepFashion

import numpy as np
import os
import os.path as osp
import cPickle
import yaml
from utils.error import compute_mle

class PersonAttributes(Imdb):

    def __init__(self, split, align=False, partition='all'):

        name = 'person_' + partition + '_' + split
        if align and (partition == 'all' or partition == 'face'):
            name += '_align'

        Imdb.__init__(self, name)

        # Load two children dataset wrappers
        self._face = CelebA(split, align=align)
        self._clothes = DeepFashion(split)
        # The class list is a combination of face and clothing attributes
        self._classes = self._face.classes + self._clothes.classes
        self._face_class_idx = range(self._face.num_classes)
        self._clothes_class_idx = range(self._face.num_classes, self._face.num_classes + self._clothes.num_classes)

        # load data path
        self._data_path = os.path.join(self.data_path, 'imdb_PersonAttributes')
        # load the image lists and attributes.
        self._load_dataset(split, align, partition)

    def _load_dataset(self, split, align, partition):
        """ Load dataset from the dataset or face attributes and cltohign attributes.  """

        if partition == 'all':
            self._image_list = self._face.image_list + self._clothes.image_list
            celeba_num = self._face.num_images
            deepfashion_num = self._clothes.num_images
        elif partition == 'face':
            self._image_list = self._face.image_list
            celeba_num = self._face.num_images
            deepfashion_num = 0
        elif partition == 'clothes':
            self._image_list = self._clothes.image_list
            celeba_num = 0
            deepfashion_num = self._clothes.num_images

        self._gtdb = {'attr': -1.*np.ones((self.num_images, self.num_classes), dtype=np.float64)}

        # load labels for celeba images if they are included. 
        if celeba_num > 0:
            self._gtdb['attr'][:celeba_num, self._face_class_idx] = self._face.gtdb['attr']
            # load soft labels for clothes attributes on celeba
            if align:
                fn = osp.join(self.data_path, 'person_'+'face'+'_'+split+'_align.pkl')
            else:
                fn = osp.join(self.data_path, 'person_'+'face'+'_'+split+'.pkl') 
            if osp.exists(fn):
                with open(fn, 'rb') as fid:
                    labels = cPickle.load(fid)
                    self._gtdb['attr'][:celeba_num, self._clothes_class_idx] = labels
            else:
                print 'Dataset {}: Labels for clothes attributes on CelebA are not available! Missing filename: {}. Did you forget to run load_person.py first?'.\
                    format(self.name, fn)

        # load labels for deepfashion images if they are included.
        if deepfashion_num > 0:
            self._gtdb['attr'][celeba_num:, self._clothes_class_idx] = self._clothes.gtdb['attr']
            # load soft labels for face attributes on deepfashion
            fn = osp.join(self.data_path, 'person_'+'clothes'+'_'+split+'.pkl')
            if osp.exists(fn):
                with open(fn, 'rb') as fid:
                    labels = cPickle.load(fid)
                    self._gtdb['attr'][celeba_num:, self._face_class_idx] = labels
            else:
                print 'Dataset {}: Labels for face attributes on Deepfashion are not available! Missing filename: {}. Did you forget to run load_person.py first?'.\
                    format(self.name, fn)

    def evaluate(self, scores, ind, cls_idx=None):
        """ Evaluation: Report classification error rate. 
            "scores" is a (N x C) matrix, where N is the number of samples, 
                C is the number of classes. C=len(cls_idx) if provided.
            "ind" is an array that index into result
        """
        if cls_idx is None:
            cls_idx = np.arange(self.num_classes)

        gt = self.gtdb['attr'][ind, :]
        gt = gt[:, cls_idx]
        err = compute_mle(scores, gt)
        
        return err

    def print_info(self, i):
        """ Output information about the image and some ground truth. """

        im_size = self.image_size(i)
        print 'The path of the image is: {}'.format(self.image_path_at(i))
        print 'width: {}, height: {}'.format(im_size[0], im_size[1])
        
        attr_i = self.gtdb['attr'][i, :]
        print 'The attributes are: {}'.format(','.join([self._classes[i] for i in np.where(attr_i==1)[0]]))
