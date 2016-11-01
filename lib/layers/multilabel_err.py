# --------------------------------------------------------
# Written by Yongxi Lu
# --------------------------------------------------------

"""A Python layer to comupte multi-label error """

import caffe
import numpy as np
import yaml

# def _eval_soft_error(scores, targets):
#     """ Compute multi-label error """
#     num_samples, num_classes = targets.shape
#     err = np.empty((num_samples, num_classes), dtype=np.float32)
#     err[:] = np.nan
#     for c in xrange(num_classes):
#         # negative label is reserved for "unknown", evaluation of those entries are skipped. 
#         valid_ind = np.where(targets[:,c]>=0.0)[0]
#         if len(valid_ind>0):
#             # err[valid_ind, c] = (scores[valid_ind, [c]]>=0.5) != (targets[valid_ind, [c]]>=0.5)
#             # soft errors 
#             pos_labels = np.where(targets[valid_ind, [c]]>=0.5)[0]
#             neg_labels = np.where(targets[valid_ind, [c]]<0.5)[0]
#             err[valid_ind[pos_labels], c] = 1.0 - scores[valid_ind[pos_labels], [c]]
#             err[valid_ind[neg_labels], c] = scores[valid_ind[neg_labels], [c]]
#     return err

def _eval_soft_error(scores, targets):
    """ Compute multi-label error """
    num_samples, num_classes = targets.shape
    targets[np.where(targets[:]<0.0)] = np.nan
    err = np.empty((num_samples, num_classes), dtype=np.float32)
    err[:] = np.nan

    pos_labels = np.where(targets>=0.5)
    neg_labels = np.where(targets<0.5)
    err[pos_labels] = 1.0 - scores[pos_labels]
    err[neg_labels] = scores[neg_labels]

    return err

class MultiLabelErr(caffe.Layer):
    """Multi-label error."""
    
    def setup(self, bottom, top):
        """Setup the layer."""
        top[0].reshape(1, 1)

    def forward(self, bottom, top):
        """Compute multi-label error."""

        scores = bottom[0].data
        targets = bottom[1].data
        err = _eval_soft_error(scores, targets)
      
        top[0].reshape(*(err.shape))
        top[0].data[...] = err.astype(np.float32, copy=False)
   
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens at setup."""
        pass
