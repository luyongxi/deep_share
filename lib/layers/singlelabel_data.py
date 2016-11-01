# --------------------------------------------------------
# Written by Yongxi Lu
# --------------------------------------------------------

"""A Python layer for single-label classification input """

from utils.config import cfg
from .classification_data import ClassificationData
import numpy as np

class SingleLabelData(ClassificationData):
    """Single label data layer."""

    def _shape_output_maps(self, top):
        """ reshape the output maps """
        self._name_to_top_map = {
            'data': 0,
            'label': 1}

        # data blob: holds a batch of N images, each with 3 channels
        # label blob: holds a batch of labels
        if self._stage == 'TRAIN':
            top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.SCALE, cfg.SCALE)
            top[1].reshape(cfg.TRAIN.IMS_PER_BATCH)
        elif self._stage == 'VAL':
            top[0].reshape(1, 3, cfg.SCALE, cfg.SCALE)
            top[1].reshape(1)

    def _label_gt_from_inds(self, inds):
        """ Get label gt from inds """
        label = self._imdb.gtdb['label'][inds]
        class_list = np.array(self._class_list)
        indexed_label = class_list[label].astype(np.float32, copy=False)

        return indexed_label