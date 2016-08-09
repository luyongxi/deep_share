# Written by Yongxi Lu

# import base class
from imdb import Imdb
import numpy as np
import os
import os.path as osp
import cPickle

"""
Class to manipulate CelebA dataset
"""

class CelebA_Plus_Webcam_Cls(Imdb):
    """ CelebA+Webcam data for 3-way classificatoin. """
    
    def __init__(self, split):
        name = 'celeba_plus_webcam_cls_'+split
        Imdb.__init__(self, name)
        
        # object classes    
        self._classes = ['Bald', 'Hat', 'Hair']    
        
        # load image paths and annotations
        self._data_path = osp.join(self.data_path, 'imdb_CelebA+Webcam') 
        self._load_dataset(split)
            
    def _load_dataset(self, split):
        """ Load image path list and ground truths """
        
        # load image path list and ground truths from the cache
        cache_file = osp.join(self.cache_path, self.name+'_dbcache.pkl')
        
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                dbcache = cPickle.load(fid)
                print '{} database cache loaded from {}'.format(self.name, cache_file)
                self._image_list = dbcache['image_list']
                self._gtdb = dbcache['gtdb']
                return

        # load list of images
        self._image_list = self._do_load_filelist(split)   
        # load ground truth labels
        self._gtdb = {'label': self._do_load_labels()}

        dbcache = {'image_list': self.image_list, 'gtdb': self.gtdb}
        # save to cache         
        with open(cache_file, 'wb') as fid:
            cPickle.dump(dbcache, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote database cache to {}'.format(cache_file)       

    def _do_load_filelist(self, split):
        """ Return the absolute paths to image files """

        # images are already separated into folders called "TrainingData" and "ValidationData"
        if split == 'train':
            base_folder = osp.join(self.data_path, 'FinalHairHatBaldBalanced', 'TrainingData')
        elif split == 'val':
            base_folder = osp.join(self.data_path, 'FinalHairHatBaldBalanced', 'ValidationData')

        image_list = []
        for cls_idx in xrange(self.num_classes):
            src_folder = osp.join(base_folder, self.classes[cls_idx])
            image_list.extend([osp.join(src_folder, fn) for fn in os.listdir(src_folder)])

        return image_list

    def _do_load_labels(self):
        """ Load labels of the listed images. 
            Due to the structure of this dataset, we can find
            labels by looking at the folder of the file in the filelist
        """
        labels = np.zeros((self.num_images), dtype=np.int64)
        for i in xrange(self.num_images):
            labels[i] = self.classes.index(osp.basename(osp.dirname(self.image_path_at(i))))

        return labels

    def evaluate(self, scores, ind):
        """ Evaluation: Report classificaiton accuracy.
            The scores is a matrix, where each row is a sample point
            and the columns are scores for each class. 
            A classfication is correct if the argmax in the scores are 
            in fact correct label.
        """
        gt = self.gtdb['label'][ind]
        pred = np.argmax(scores, axis=1)
        acc = float(gt == pred) / len(ind)
        
        return acc

    def print_info(self, i):
        """ Output information about the image and some ground truth. """

        im_size = self.image_size(i)
        print 'The path of the image is: {}'.format(self.image_path_at(i))
        print 'width: {}, height: {}'.format(im_size[0], im_size[1])
        
        label_i = self.gtdb['label'][i]
        print 'The label is {}'.format(self.classes[label_i])