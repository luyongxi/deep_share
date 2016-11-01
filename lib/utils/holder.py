#------------------------------
# Written by Yongxi Lu
#------------------------------

from collections import deque
import numpy as np
from utils.config import cfg

class CircularQueue(object):
    """ Hold a queue with length  """

    def __init__(self, maxlen):
        self._holder = deque([], maxlen=maxlen)

    def append(self, value):
        self._holder.append(value)

    def toMatrix(self):
        return np.concatenate(self._holder, axis=0)