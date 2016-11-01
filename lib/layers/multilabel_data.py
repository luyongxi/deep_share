# --------------------------------------------------------
# Written by Yongxi Lu
# --------------------------------------------------------

"""A Python layer for multi-label classification input """

from utils.config import cfg
from .classification_data import ClassificationData
import numpy as np

class MultiLabelData(ClassificationData):
    """Multilabel data layer."""

    def _shape_output_maps(self, top):
        """ reshape the output maps """

        self._name_to_top_map = {
            'data': 0,
            'label': 1}

        # data blob: holds a batch of N images, each with 3 channels
        # label blob: holds a batch of N labels, each with _num_classes labels
        if self._stage == 'TRAIN':
            top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.SCALE, cfg.SCALE)
            top[1].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes)
        elif self._stage == 'VAL':
            top[0].reshape(1, 3, cfg.SCALE, cfg.SCALE)
            top[1].reshape(1, self._num_classes)

    def _label_gt_from_inds(self, inds):
        """ Get label gt from inds """
        attr_all = self._imdb.gtdb['attr'][inds, :].astype(np.float32, copy=False)
        attr_gt = attr_all[:, self._class_list]

        return attr_gt