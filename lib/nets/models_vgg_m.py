# --------------------------------
# Written by Yongxi Lu
# --------------------------------

""" Generates the prototxt for the baseline and branch model based on VGG-M"""

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# TODO: create helper functoins for modules

# Notes:
# To specifiy a parameter, a list represents repeated messages, and a dictionary
# represents a message.
# In other words, if we want fill in a repeated message (e.g. weight filler, multiple 
# parameters), we should use a list of dictionary


# TODO: need to at least speicfy the input connections, and how many outputs.
def add_basis_filter(n, name, bottom, ks, num_basis, num_in=None, stride=1, pad=1):
	""" Add a set of basis filters with their linear combinations.
		
		Convolutional filters
		If required, create linear combination at input to adjust dimensions.

		Intput:
			n: handle to network model
			name: name of the layer
			bottom: input of the layer
			ks: kernel size of the basis filters
			num_basis: number of basis filters
			num_in: number of input channels (if None, set to same as input layer)
			stride: stride of the basis filters
			pad: padding for the basis filters
	"""

	



	# TODO: how do we share parameters across basis?
	# Perhaps we should generate all the branches in the same function?
	# TODO: use the initialiation scheme proposed in the ICLR 2016 paper.

	# the name should be something like basis1, basis2 (depending on the layer)
	# the actual name depends on the branch.

	# std = np.sqrt(1/(something in the paper))
	weight_filler=[{type="gaussian", std=std},{type="constant", value=0}]
	
	# branch_name = name + '{}'.format(branch id)
	n[branch_name] = L.Convolution(bottom, kernel_size=kn, num_output=nout, stride=stride, pad=pad, 
		param=param, weight_filler=weight_filler)

	# TODO: perhaps followed by a 

def add_max_pool(n, name, bottom, ks, stride=2):
	n[name] = L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


# TODO: how to specify output? Is there a smart way?
# TODO: as a special case, when num_branches = 1, the model is the baseline
# TODO: need to specify whether we want one output from each branch, or we want
# 		multiple outputs from one or more joint fc layers.
# TODO: need to specify the number of outputs, which could be different from the 
#		number of branches. In fact, considering the application, we should probably
#		define task-specific output layers (or perhaps we should keep things simple for now?)


# This should be the __init__ function of the base class for all networks we are interested in 
# For all those networks, we should specify the number of branches, the number of tasks and the number of feature maps
# But depending on the exact configurations, we might want to specify just one number across all layers, or for each layer
# Another important parameters is the number of layers. However, even if we know the number of layers, it is hard to see 
def branchnet(branches, num_tasks, num_maps, deploy=False, **params):
	""" Create a network with branches 
		Inputs: 
			branches: specify the branche structure
			num_tasks: number of tasks to test
			num_maps: number of feature maps per task
			[deploy]: generate deploy prototxt or not [=False]
			params: other parameters
	"""

	n = caffe.NetSpec()

	# data layer
	# TODO: how to include "phase: TRAIN" and "phase: Test"?
	if deploy == False:
		# input for training phase
		n['data'] = L.Python(python_param={module: "layers.att_data", layer: "AttributeData", 
			param_str: "{num_classes: {}, stage: TRAIN}".format(num_tasks)})
		# input for testing phase
		n['data'] = L.Python(python_param={module: "layers.att_data", layer: "AttributeData", 
		param_str: "{num_classes: {}, stage: VAL}".format(num_tasks)})
	else:
		# TODO: input is simply data
		pass

	# convolution layers (1-5)
	# each convolution layer is some joint filter, some basis filters and some combination filters

if __name__ == '__main__':
	# TODO: build a baseline network (without any branching etc.)
