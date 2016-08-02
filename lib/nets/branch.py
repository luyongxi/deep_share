# --------------------------------
# Written by Yongxi Lu
# --------------------------------

""" Generates the prototxt for the branch model based on VGG-M"""

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# TODO: create helper functoins for modules

# Notes:
# To specifiy a parameter, a list represents repeated messages, and a dictionary
# represents a message.
# In other words, if we want fill in a repeated message (e.g. weight filler, multiple 
# parameters), we should use a list of dictionary

def basis_filter(bottom, ks, nout, stride=1, pad=0, group=1):
	pass

def max_pool(botton, ks, stride=1):
	return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# TODO: how to specify output? Is there a smart way?
def branchnet(num_branches, deploy=False, use_bn=False):
	""" Create a network with branches """


if __name__ == '__main__':
	# TODO: perform some testing
