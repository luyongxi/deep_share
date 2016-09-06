# Written by Yongxi Lu

# import base class
from imdb import Imdb
import numpy as np
import os
import os.path as osp
import cPickle
import yaml

"""Class to manipulate IBMattributes dataset """

class IBMAttributes(Imdb):
    """ IBM attribute classification dataset. """
    
    def __init__(self, split):
        name = 'IBMattributes_'+split
        Imdb.__init__(self, name)
        
        # attribute classes    
        self._classes = ['Bald', 'Hat', 'Hair', 'Blackhair', 'Blondehair', 'Facialhair', 'Asian','Black','White']
        self._split = split
        
        # load image paths and annotations
        self._data_path = osp.join(self.data_path, 'imdb_IBMAttributes') 
        self._load_config()
        self._load_dataset(split)

    def score_file_name(self, cls_id):
        """ The name of the pkl file for soft labels """
        return osp.join(self.data_path, self._split+'_'+self.classes[cls_id]+'.pkl')

    def list_incomplete(self, cls_id):
        """ List images with imcomplete labels at a certain class """
        return np.where(self.gtdb['attr'][:, cls_id]<0)[0]

    def find_labeler(self, cls_id):
        """ Return infos required to perform labeling at a particular class 
            [caffemodel, score_name, score_idx]
            Note that caffemodel in the config file is relative to the data path
        """
        cls_name = self._classes[cls_id]
        dest_classes = [item[1] for item in self._config]
        item_idx = dest_classes.index(cls_name)
        labler = self._config[item_idx][2:5]
        labler[0] = osp.join(self.data_path, labler[0])
        src_name = self._config[item_idx][0]

        return src_name, labler

    def _load_config(self):
        """ Load config file that tells us where pre-trained models for soft labels are 
            Config file format:
            [src_name, dest_name, caffemodel, score_name, score_idx, negative]
            ---------------------------------------------------------------------------------------------------
            src_name: the name of a particular visual concept in the source dataset, used to load ground truth
            dest_name: the name of the particular visual concept in the destination dataset
            caffemodel: the caffemodel file associated with this class. If not provide specify "null"
            score_name: the name of the score layer in the caffemodel
            score_idx: the index of the particular score in the score layer in the caffemodel
            negative: If "true", a score True in the src_name indicates a negative score in the dest_name
        """
        self._config = []
        file = os.path.join(self.data_path, 'config.txt')
        with open(file, 'r') as fid:
            next(fid)   # skip the first line            
            for line in fid:
                entry = yaml.safe_load(line)
                self._config.append(entry)

    def _load_dataset(self, split):
        """ Load image path list and ground truths """

        # load image path list and ground truths from the cache
        cache_file = osp.join(self.cache_path, self.name+'_dbcache.pkl')
        # only use caches when there is no missing labels
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                dbcache = cPickle.load(fid)
                print '{} database cache loaded from {}'.format(self.name, cache_file)
                self._image_list = dbcache['image_list']
                self._gtdb = dbcache['gtdb']
                # check if there is missing labels. 
                if np.any([len(self.list_incomplete(idx))>0 for idx in xrange(self.num_classes)]):
                    self._gtdb['attr'] = np.maximum(self._gtdb['attr'], 
                        self._do_load_soft_labels(split))
                return

        # We should first load gt_labels, and then attempt to load soft labels
        # load list of images
        self._image_list = self._do_load_filelist(split)   
        # load ground truth labels
        self._gtdb = {'attr': self._do_load_gt_labels()}
        self._gtdb['attr'] = np.maximum(self._gtdb['attr'], 
            self._do_load_soft_labels(split))

        dbcache = {'image_list': self.image_list, 'gtdb': self.gtdb}
        # save to cache         
        with open(cache_file, 'wb') as fid:
            cPickle.dump(dbcache, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote database cache to {}'.format(cache_file)

    def _do_load_filelist(self, split):
        """ Return the absolute paths to image files """
        # images are already separated into folders called "TrainingData" and "ValidationData"
        if split == 'train':
            base_folder = osp.join(self.data_path, 'TrainingData')
        elif split == 'val':
            base_folder = osp.join(self.data_path, 'ValidationData')

        image_list = []
        # the first entries in the sub-lists of self._config are src class names
        for src_idx in xrange(len(self._config)):
            src_folder = osp.join(base_folder, self._config[src_idx][0])
            image_list.extend([osp.join(src_folder, fn) for fn in os.listdir(src_folder)])

        return image_list

    def _do_load_gt_labels(self):
        """ Load ground truth labels of the listed images. 
            Due to the structure of this dataset, we can find
            labels by looking at the folder of the file in the filelist
        """
        print 'loading ground truth labels...'
        src_classes = [item[0] for item in self._config]
        dest_classes = [item[1] for item in self._config]
        neg = [item[-1] for item in self._config]
        # by default, all the labels are marked "-1", meaning the attribute label is unknown. 
        labels = -1.0 * np.ones((self.num_images, self.num_classes), dtype=np.float32)
        for i in xrange(self.num_images):
            # conversion: first match the folder name to a src_class, then match that src_class to a 
            src_label = src_classes.index(osp.basename(osp.dirname(self.image_path_at(i))))
            labels[i, self.classes.index(dest_classes[src_label])] = 0 if neg[src_label] else 1

        return labels

    def _do_load_soft_labels(self, split):
        """ load soft labels from the dataset """
        # src_classes = [item[0] for item in self._config]
        print 'loading soft labels...'
        dest_classes = [item[1] for item in self._config]
        neg = [item[-1] for item in self._config]
        # by default, all the labels are marked "-1", meaning the attribute label is unknown. 
        labels = -1.0 * np.ones((self.num_images, self.num_classes), dtype=np.float32)

        for c in xrange(self.num_classes):
            score_file_name = self.score_file_name(c)
            if osp.exists(score_file_name) and self.classes[c] in dest_classes:
                img_idx_c = self.list_incomplete(c)
                with open(score_file_name, 'rb') as fid:
                    soft_labels_c = cPickle.load(fid)
                    match_id = dest_classes.index(self.classes[c])
                    print '{} soft label for class {} loaded from {}'.\
                        format('neg' if neg[match_id] else 'pos', self.classes[c], score_file_name)
                    labels[img_idx_c, c] = 1.0-soft_labels_c if neg[match_id] else soft_labels_c

        return labels

    # def evaluate(self, scores, ind, cls_idx=None):
    #     """ Evaluation: Report classification error rate. 
    #         "scores" is a (N x C) matrix, where N is the number of samples, 
    #             C is the number of classes. C=len(cls_idx) if provided.
    #         "ind" is an array that index into result
    #     """
    #     if cls_idx is None:
    #         cls_idx = np.arange(self.num_classes)

    #     err = np.zeros((len(cls_idx), ))
    #     for i in xrange(len(cls_idx)):
    #         gt = self.gtdb['attr'][ind, cls_idx[i]]
    #         valid_ind = np.where(gt>=0.0)[0]
    #         if len(valid_ind) > 0:
    #             err[i] = compute_mle(scores[valid_ind, i], gt[valid_ind])
    #         else:
    #             err[i] = 0.0

    #     return err
    
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