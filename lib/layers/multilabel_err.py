# --------------------------------------------------------
# Written by Yongxi Lu
# --------------------------------------------------------

"""A Python layer to comupte multi-label error

"""

import caffe
import numpy as np
import yaml

def compute_mle(pred, target):
    """ Compute multi-label error """
    label = (pred>0.5)
    err = np.mean((label != target), axis=0)
    return err

class MultiLabelErr(caffe.Layer):
    """Multi-label error."""
    
    def setup(self, bottom, top):
        """Setup the layer."""
        top[0].reshape(1, 1)

    def forward(self, bottom, top):
        """Compute multi-label error."""

        pred = bottom[0].data
        target = bottom[1].data
        err = compute_mle(pred, target)
      
        top[0].reshape(*(err.shape))
        top[0].data[...] = err.astype(np.float32, copy=False)
   
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens at setup."""
        pass
