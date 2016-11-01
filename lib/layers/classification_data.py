# --------------------------------------------------------
# Written by Yongxi Lu
# --------------------------------------------------------

"""Base class for python layers that supplies inputs for classifications in training"""

import caffe
from utils.config import cfg
from utils.blob import im_list_to_blob
import numpy as np
import yaml

class ClassificationData(caffe.Layer):
    """Classification data layer."""

    def _shuffle_img_inds(self):
        """Randomly permute the training images."""
        self._perm = np.random.permutation(np.arange(self._imdb.num_images))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the image indices for the next imnibatch"""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= self._imdb.num_images:
            self._shuffle_img_inds()
        
        db_inds = self._perm[self._cur:self._cur+cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Get a set of minibatches as a blob"""
        db_inds = self._get_next_minibatch_inds()
        blobs = self._get_blobs_from_inds(db_inds)
        return blobs

    def _get_random_val_batch(self):
        """ Get a random batch of samples for validation
            Packed into a blob
        """
        num_samples = min(self._imdb.num_images, 1)
        db_inds = np.random.choice(self._imdb.num_images, size=num_samples, replace=False)
        blobs = self._get_blobs_from_inds(db_inds)
        return blobs

    def _get_blobs_from_inds(self, inds):
        """ Prepare a blob of images given inds """
        filelist = [self._imdb.image_path_at(i) for i in inds]
        im_blob = im_list_to_blob(filelist, cfg.PIXEL_MEANS, cfg.SCALE)

        label_gt = self._label_gt_from_inds(inds)
        blobs = {'data': im_blob, 'label': label_gt}
        return blobs

    def _label_gt_from_inds(self, inds):
        """ Get label gt from inds """
        return NotImplementedError

    def set_imdb(self, imdb):
        """ Set imdb to be use by this layer """
        self._imdb = imdb
        if self._stage == 'TRAIN':  
            self._shuffle_img_inds()

    def set_classlist(self, class_list):
        """ Change the list of classes to test """
        self._class_list = class_list
        assert len(self._class_list) == self._num_classes, \
            'Number of classes does not match class list: {} vs {}'.\
            format(len(self._classlist), self._num_classes)
  
    def setup(self, bottom, top):
        """Setup the AttributeData."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        self._stage = layer_params['stage']

        # load class list if provided, otherwise set it to default
        if ['class_list'] in layer_params.keys():
            self._class_list = np.array(layer_params['class_list'])
            assert len(self._class_list) == self._num_classes, \
                'Number of classes does not match class list: {} vs {}'.\
                format(len(self._classlist), self._num_classes)
        else:
            self._class_list = np.arange(self._num_classes)

        self._shape_output_maps(top)

    def _shape_output_maps(self, top):
        """ reshape the output maps """
        return NotImplementedError

    @property
    def num_classes(self):
        return self._num_classes

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
    
        if self._stage == 'TRAIN':
            blobs = self._get_next_minibatch()
        elif self._stage == 'VAL':
            blobs = self._get_random_val_batch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
