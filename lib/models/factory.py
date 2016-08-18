# Written by Yongxi Lu

""" Model registry, all valid models are registred in this module. """

from models_low_rank import ModelsLowRank
from model_io import MultiLabelIO
import os.path as osp


# model functor
__models = {}

# NOTE: when counting fully-connected layers, the final output layer is not counted. 

# vgg-m low-rank model
__models['low-vgg-m'] = (lambda io, model_name='low-vgg-m', 
    path='.', **kwargs: 
    _low_vgg_m_gen(io=io, model_name=model_name, path=path, **kwargs))

# vgg-16 low-rank model 
__models['low-vgg-16'] = (lambda io, model_name='low-vgg-16', 
    path='.', **kwargs: 
    _low_vgg_16_gen(io=io, model_name=model_name, path=path, **kwargs))

def get_models(name, **kwargs):
    """ Get an model (network architecture) by name."""
    if not __models.has_key(name):
        raise KeyError('Unknown model: {}'.format(name))
    return __models[name](**kwargs)

def list_models():
    """ List all registred models."""
    return __models.keys()

def _low_vgg_m_gen(io, model_name, path, **kwargs):
    """ Low rank VGG-M with two fully connected layers. """

    if kwargs.has_key('first_low_rank'):
        first_low_rank = kwargs['first_low_rank']
    else:
        first_low_rank = 0

    num_filters = {'conv': {0:12, 1:32, 2:64, 3:64, 4:64}, 
                    'fc': {5: 512, 6: 512}, 'output': {7:512}}
    num_filters['conv'] = {k:v for k,v in num_filters['conv'].iteritems() if k>=first_low_rank}
    num_filters['fc'] = {k:v for k,v in num_filters['fc'].iteritems() if k>=first_low_rank}
    num_filters['output'] = {k:v for k,v in num_filters['output'].iteritems() if k>=first_low_rank}

    model = ModelsLowRank(model_name=model_name, path=path, 
        io=io, num_filters=num_filters)

    param_mapping = {}
    for i in xrange(model.num_layers):
        value = 'conv{}'.format(i+1)
        net_names = model.names_at_i_j(i, 0)
        if i < first_low_rank:
            param_mapping[(model.branch_name_at_i_j_k(i, 0, 0), )] = value
        else:
            param_mapping[(model.col_name_at_i_j(i, 0), \
                model.branch_name_at_i_j_k(i, 0, 0))] = value

    return model, param_mapping

def _low_vgg_16_gen(io, model_name, path, **kwargs):
    """ VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        
        VGG-16 defines the number of outputs at each layer, but it does
        not define the number of filters.

        If the number of filters is set to k=0.9*num_output, then the number
        of parameters is exactly the same as the original model.
        Let's heuristically use 1/8 num_output, so we only have 13.8% of the 
        original number of paramters. 

        first_low_rank specifies the first layer to perform low-rank factorization. 
    """
    if kwargs.has_key('first_low_rank'):
        first_low_rank = kwargs['first_low_rank']
    else:
        first_low_rank = 0

    num_layers = 15     # not counting the last task layer
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:64,8:64,9:64,10:64,11:64,12:64}, 
                    'fc':{13:512,14:512}, 'output': {15:512}}

    # only layers above (including) first_low_rank should have low-rank factorization 
    num_filters['conv'] = {k:v for k,v in num_filters['conv'].iteritems() if k>=first_low_rank}
    num_filters['fc'] = {k:v for k,v in num_filters['fc'].iteritems() if k>=first_low_rank}
    num_filters['output'] = {k:v for k,v in num_filters['output'].iteritems() if k>=first_low_rank}

    # specify number of outputs
    num_outputs = {'conv':{0:64,1:64,2:128,3:128,4:256,5:256,6:256,
                           7:512,8:512,9:512,10:512,11:512,12:512}, 
                    'fc':{13:4096,14:4096}}
    conv_k = [3 for _ in xrange(13)]
    conv_ks = [1 for _ in xrange(13)]
    conv_pad = [1 for _ in xrange(13)]
    # parameters for pooling
    include_pooling = [1,3,6,9,12]
    pool_k = [2 for _ in xrange(13)]
    pool_ks = [2 for _ in xrange(13)]
    pool_pad = [0 for _ in xrange(13)]
    # parameters for dropout
    include_dropout = [13,14]    
    dropout_ratio = [0.5,0.5]
    # no lrn is used
    include_lrn = []

    model = ModelsLowRank(model_name=model_name, path=path, io=io, num_layers=num_layers,
        num_filters=num_filters, num_outputs=num_outputs, conv_k=conv_k, conv_ks=conv_ks, conv_pad=conv_pad,
        include_pooling=include_pooling, pool_k=pool_k, pool_ks=pool_ks, pool_pad=pool_pad,
        include_dropout=include_dropout, dropout_ratio=dropout_ratio,
        include_lrn=include_lrn)

    # find out the appropriate param_mapping for initilizaiton
    def vanila_vgg_16_names(i):
        """ names in the pretrained network """
        if i<=1:    #conv 1
            value = 'conv1_{}'.format(i%2+1)
        elif i<=3:  #conv 2
            value = 'conv2_{}'.format((i-2)%2+1)
        elif i<=6:  #conv 3
            value = 'conv3_{}'.format((i-4)%3+1)
        elif i<=9:  #conv 4
            value = 'conv4_{}'.format((i-7)%3+1)
        elif i<=12: #conv 5
            value = 'conv5_{}'.format((i-10)%3+1)
        else:
            value = 'fc{}'.format(i-7)

        return value

    param_mapping = {}
    for i in xrange(model.num_layers):
        value = vanila_vgg_16_names(i)
        if i < first_low_rank:
            param_mapping[(model.branch_name_at_i_j_k(i, 0, 0), )] = value
        else:
            param_mapping[(model.col_name_at_i_j(i, 0), \
                model.branch_name_at_i_j_k(i, 0, 0))] = value

    return model, param_mapping

# TODO: implement the options of cutting the network. 
# TODO: note that the cutting procedure can start only when the
# initial cut is provided. The function thus takes two inputs.
# One is a number specifying how many cuts to be made, the
# other is a list with two elements that are lists, and 
# the elements represents a group of tasks. 


# names of the arguments
# cut_depth: [default: 0]
# task_split: []

# put these lines of codes before the last line
# return model, param_mapping

# Since this is likely to be used in more than once places
# We should write a function called create_branch_from_plain(cut_depth, model, param_mapping)
# This function should return the new model the param_mapping summarizing the old and the new. 

# if kwargs.has_key('cut_depth'):
#     if kwargs['cut_depth'] > 0:
#         # TODO: assert that kwargs['cut_depth'] < num_layers)

#         mappings = [param_mapping]

#         for d in xrange(kwargs['cut_depth']):
#             # TODO: we to decide the appropriate br_idx and split_idx
#             # br_idx can be determined in a deterministic fashion
#             # We want to have (15, 0), (14, 0), .. all the way towards the end
#             # we start from br_idx_init = num_layers
#             # For the split_idx, the first layer is special, we should have
#             # user-specified branches. 
#             # But starting from the second layer, split_idx should always be [[0],[1]]

#             mappings.append(model.insert_branch())

