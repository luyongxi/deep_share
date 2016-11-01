# The person attribute dataset, consists of images
# of CelebA and DeepFashion, with the facial attribute labels
# from CelebA and clothing attribute labels from DeepFashion

# Should consult the implmentation of IBMAttributes.
# Need a good way to test the its correctness as well. 

# In fact it seems a good idea, for to directly use the dataset wrapper from
# celeba and deepfashion (which will largely reduce the chance of making a mistake)
# This class will be mainly responsible for keeping track of which images are from CelebA, 
# which images are from persons, and thus how to load attribute labels when needed. 

# We should give it some methods of loading soft labels if possible. 


# import base class
from imdb import Imdb
import numpy as np
import os
import os.path as osp
import cPickle
import yaml
from utils.error import compute_mle

import CelebA, DeepFashion

class PersonAttributes(Imdb):

    # TODO: we will use the two dataset wrappers to load images and image lists.
    # This function shall provide a way to load soft labels, 
    # and to port the interface from the hard labels (by copying if necessary). 
    def __init__(self, split, align=False):

        name = 'PersonAttributes' + split
        Imdb.__init__(self, name)
        
        # Load two children dataset wrappers
        self._face = CelebA(split, align=align)
        self._clothes = DeepFashion(split)

        # TODO:
        # (1) Load class lists from two datasets. 
        # (2) Load image lists and labels from the two datasets. 

        # attribute classes
        # Load classes by loading 
        # self._classes = ['Bald', 'Hat', 'Hair', 'Blackhair', 'Blondehair', 'Facialhair', 'Asian','Black', 'White', 'NoGlasses', 'SunGlasses', 'VisionGlasses']
        # self._split = split

        # TODO: copies file lists and attribute lists from the dataset files. 
        # TODO: keep track of which images are from which dataset, and provide soft labels whenever necessary. 
        # Finally, can we reuse the save_softlabels function? 
        
        # load image paths and annotations
        self._data_path = osp.join(self.data_path, 'imdb_PersonAttributes') 


        self._load_config()
        self._load_dataset(split)


    def score_file_name(self, cls_id):
        """ The name of the pkl file for soft labels """
        return osp.join(self.data_path, self._split+'_'+self.classes[cls_id]+'.pkl')

    def list_incomplete(self, cls_id):
        """ List images with imcomplete labels at a certain class """
        return np.where(self.gtdb['attr'][:, cls_id]<0)[0]


    # TODO: I think we can greatly simplify
    # Basically, we need two caffemodels, 
    # One is for labeling classes in clothing attributes, 
    # The other is for face attributes. 
    # The config file would just be a two-liner, one tells us that 
    # where is caffemodel for the clothing attributes, and the other
    # tells us where is the caffemodel for the person attributes.
    # We can forget about all the other complex structures.  
    def find_labeler(self, cls_id):
        """ Return infos required to perform labeling at a particular class 
            [caffemodel, score_name, score_idx]
            Note that caffemodel in the config file is relative to the data path
        """
        cls_name = self._classes[cls_id]
        dest_classes = [item[1] for item in self._config]
        label_type = [item[-1] for item in self._config]
        # only "pos" labels are used, under the assumption that negative labels
        # can only be providedd through outputs from the same caffemodel
        item_idx = [i for i in xrange(len(self._config)) if 
            (dest_classes[i]==cls_name) and (label_type[i]=='pos')][0]
        labler = self._config[item_idx][2:5]
        labler[0] = osp.join(self.data_path, labler[0])
        src_name = self._config[item_idx][0]

        return src_name, labler