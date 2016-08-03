# Written by Yongxi Lu

# import base class
from imdb import Imdb
import numpy as np
import os
import cPickle
from layers.multilabel_err import compute_mle

"""
Class to manipulate CelebA dataset
"""

class CelebA(Imdb):
    """ Image database for CelebA dataset. """
    
    def __init__(self, split, align=False):
        name = 'celeba_'+split      
        if align is True:
            name += '_align'

        Imdb.__init__(self, name)
        
        # object classes    
        self._classes = \
            ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']    
        
        # load image paths and annotations
        self._data_path = os.path.join(self.data_path, 'imdb_CelebA') 
        self._load_dataset(split, align)
            
    def _load_dataset(self, split, align):
        """ Load image path list and ground truths """
        
        # load image path list and ground truths from the cache
        cache_file = os.path.join(self.cache_path, self.name+'_dbcache.pkl')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                dbcache = cPickle.load(fid)
                print '{} database cache loaded from {}'.format(self.name, cache_file)
                self._image_list = dbcache['image_list']
                self._gtdb = dbcache['gtdb']
                return
        
        # load list of images
        self._image_list, self._index_list = self._do_load_filelist(split, align)   
        
        # load attributes and landmarks data
        self._gtdb = {'attr': np.zeros((self.num_images, self.num_classes), dtype=np.bool), 'lm': np.zeros((self.num_images, 10), dtype=np.float32)}    
        self._gtdb['attr'] = self._do_load_attributes()
        self._gtdb['lm'] = self._do_load_landmarks(align)

        dbcache = {'image_list': self.image_list, 'gtdb': self.gtdb}
        # save to cache         
        with open(cache_file, 'wb') as fid:
            cPickle.dump(dbcache, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote database cache to {}'.format(cache_file)       

    def _do_load_filelist(self, split, align):
        """ Return the absolute paths to image files """
        file = os.path.join(self.data_path, 'Eval', 'list_eval_partition.txt')
    
        # determine the matching id
        if split == 'train':
            sp_idx = ['0']
        elif split == 'val':
            sp_idx = ['1']
        elif split == 'test':
            sp_idx = ['2']
        elif split == 'trainval':
            sp_idx = ['0', '1']
        else:
            raise NameError('Undefined Data Split: {}'.format(split))           

        # determine image folder
        if align:
            basepath = os.path.join(self.data_path, 'Img', 'img_align_celeba') 
        else:
            basepath = os.path.join(self.data_path, 'Img', 'img_celeba')
    
        # find paths to all files
        image_list = []
        idx_list = []
        idx = 0
        with open(file, 'r') as fid:
            for line in fid:
                split = line.split()
                if split[1] in sp_idx:
                    image_list.append(os.path.join(basepath, split[0]))
                    idx_list.append(idx)
                idx = idx + 1
        
        return image_list, idx_list

    def _do_load_attributes(self):
        """ Load attributes of the listed images. """
        file = os.path.join(self.data_path, 'Anno', 'list_attr_celeba.txt')

        attr = np.zeros((self.num_images, self.num_classes), dtype=np.bool)

        base_idx = min(self._index_list)
        end_idx = max(self._index_list)

        with open(file, 'r') as fid:
            for _ in xrange(2+base_idx):    # skip the first two+base_idx lines
                next(fid)
            
            idx = 0
            for line in fid:
                split = line.split()
                if idx <= end_idx-base_idx:
                    attr[idx, :] = np.array(split[1:], dtype=np.float32) > 0
                idx = idx + 1
        
        return attr

    def _do_load_landmarks(self, align):
        """ Load landmarks of the litsed images. """
        if align:
            file = os.path.join(self.data_path, 'Anno', 'list_landmarks_align_celeba.txt')
        else:
            file = os.path.join(self.data_path, 'Anno', 'list_landmarks_celeba.txt')

        lm = np.zeros((self.num_images, 10), dtype=np.float32)
    
        base_idx = min(self._index_list)
        end_idx = max(self._index_list)
            
        with open(file, 'r') as fid:
            for _ in xrange(2+base_idx):    # skip the first two+base_idx lines
                next(fid)
            
        idx = 0
        for line in fid:
            split = line.split()
            if idx <= end_idx-base_idx:
                lm[idx, :] = np.array(split[1:], dtype=np.float32)
            idx = idx + 1

        return lm
    
    def evaluate(self, scores, ind, cls_idx=None):
        """ Evaluation: Report classification error rate. 
            "scores" is a (N x C) matrix, where N is the number of samples, 
                C is the number of classes. C=len(cls_idx) if provided.
            "ind" is an array that index into result
        """
        if cls_idx is None:
            cls_idx = np.arange(self.num_classes)

        gt = self.gtdb['attr'][ind, cls_idx]
        err = compute_mle(scores, gt)
        
        return err

    def print_info(self, i):
        """ Output information about the image and some ground truth. """

        im_size = self.image_size(i)
        print 'The path of the image is: {}'.format(self.image_path_at(i))
        print 'width: {}, height: {}'.format(im_size[0], im_size[1])
        
        attr_i = self.gtdb['attr'][i, :]
        lm_i = self.gtdb['lm'][i, :]

        print 'The attributes are: {}'.format(','.join([self._classes[i] for i in np.where(attr_i==1)[0]]))
        print 'The landmarks points are: {}'.format(lm_i)   
