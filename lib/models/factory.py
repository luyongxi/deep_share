# Written by Yongxi Lu

""" Model registry, all valid models are registred in this module. """

from .netmodel import reduce_param_mapping
from .models_low_rank import ModelsLowRank
from .model_io import MultiLabelIO, SingleLabelIO
from solvers.classification_sw import MultiLabelSW, SingleLabelSW
import os.path as osp
from collections import OrderedDict
import numpy as np

# solver wrapper functor
def get_sw(name, **kwargs):
    if name == 'multilabel':
        return MultiLabelSW(**kwargs)
    elif name == 'singlelabel':
        return SingleLabelSW(**kwargs)
    else:
        print 'task {} is not registered!'.format(name)
        raise

# io objects functor
def get_io(name, **kwargs):
    if name == 'multilabel':
        return MultiLabelIO(**kwargs)
    elif name == 'singlelabel':
        return SingleLabelIO(**kwargs)
    else:
        print 'task {} is not registered!'.format(name)
        raise

# model functor
__models = {}

# NOTE: when counting fully-connected layers, the final output layer is not counted. 

# vgg-16 low-rank model 
__models['lowvgg16'] = (lambda io, model_name='low-vgg-16', 
    path='.', **kwargs: 
    _low_vgg_16_gen(io=io, model_name=model_name, **kwargs))

# a list of smaller vgg-16 models with varying number of outputs.  
__models['small256-lowvgg16'] = (lambda io, model_name='small256-low-vgg-16', 
    path='.', **kwargs: 
    _small_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=256, num_fc=512, **kwargs))

# a list of smaller vgg-16 models with varying number of outputs.  
__models['small192-lowvgg16'] = (lambda io, model_name='small192-low-vgg-16', 
    path='.', **kwargs: 
    _small_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=192, num_fc=384, **kwargs))

# a list of smaller vgg-16 models with varying number of outputs.  
__models['small128-lowvgg16'] = (lambda io, model_name='small128-low-vgg-16', 
    path='.', **kwargs: 
    _small_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=128, num_fc=256, **kwargs))

__models['small64-lowvgg16'] = (lambda io, model_name='small64-low-vgg-16', 
    path='.', **kwargs: 
    _small_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=64, num_fc=128, **kwargs))

__models['small32-lowvgg16'] = (lambda io, model_name='small32-low-vgg-16', 
    path='.', **kwargs: 
    _small_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=32, num_fc=64, **kwargs))

# a list of tiny vgg-16 models with varying number of outputs.  
__models['tiny256-lowvgg16'] = (lambda io, model_name='tiny256-low-vgg-16', 
    path='.', **kwargs: 
    _tiny_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=256, num_fc=512, **kwargs))

__models['tiny64-lowvgg16'] = (lambda io, model_name='tiny64-low-vgg-16', 
    path='.', **kwargs: 
    _tiny_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=64, num_fc=128, **kwargs))

__models['tiny32-lowvgg16'] = (lambda io, model_name='tiny32-low-vgg-16', 
    path='.', **kwargs: 
    _tiny_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=32, num_fc=64, **kwargs))

__models['tiny16-lowvgg16'] = (lambda io, model_name='tiny16-low-vgg-16', 
    path='.', **kwargs: 
    _tiny_low_vgg_16_gen(io=io, model_name=model_name, 
        max_num_conv=16, num_fc=32, **kwargs))

# some special models
__models['sp-lowvgg16'] = (lambda io, model_name='sp-lowvgg16', 
    path='.', **kwargs: 
    _sp_low_vgg_16_gen(io=io, model_name=model_name, **kwargs))

__models['sp1-lowvgg16'] = (lambda io, model_name='sp1-lowvgg16', 
    path='.', **kwargs: 
    _sp1_low_vgg_16_gen(io=io, model_name=model_name, **kwargs))

def get_models(name, **kwargs):
    """ Get an model (network architecture) by name."""
    if not __models.has_key(name):
        raise KeyError('Unknown model: {}'.format(name))
    return __models[name](**kwargs)

def list_models():
    """ List all registred models."""
    return __models.keys()

def _cut_network(model, param_mapping, cut_depth, cut_points):
    """ Cut the netwrok along multiple sets of tasks """
    # support network with multiple specialist branches
    num_layers = model.num_layers
    if cut_depth > 0:
        mappings = [param_mapping]
        for cur_layer in xrange(num_layers, num_layers-cut_depth, -1):
            # first cut
            if cur_layer == num_layers:
                new_mapping, new_branches = model.insert_branch((cur_layer, 0), cut_points)
            else:
                new_mapping, new_branches = model.insert_branch((cur_layer, 0), 
                    [[i] for i in xrange(len(cut_points))])
            mappings.append(new_mapping)
        # convert to a unified param_mapping
        param_mapping = reduce_param_mapping(mappings)

    return model, param_mapping, new_branches

# TODO: need to add something that tells us max columns and the cost per addint columns!

def _low_vgg_16_gen(io, model_name, **kwargs):
    """  
        The plain VGG-16 model. 

        VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        
        VGG-16 defines the number of outputs at each layer, but it does
        not define the number of filters.

        If the number of filters is set to k=0.9*num_output, then the number
        of parameters is exactly the same as the original model.
        Let's heuristically use 1/8 num_output, so we only have 13.8% of the 
        original number of paramters. 

        last_low_rank specifies the last layer to perform low-rank factorization. 
    """
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:64,8:64,9:64,10:64,11:64,12:64}, 
                    'fc':{13:64,14:64}, 'output': {15:16}}

    # specify number of outputs
    num_outputs = {'conv':{0:64,1:64,2:128,3:128,4:256,5:256,6:256,
                           7:512,8:512,9:512,10:512,11:512,12:512}, 
                    'fc':{13:4096,14:4096}}

    return _low_vgg_16_base(io, model_name, 
        num_filters=num_filters, num_outputs=num_outputs, include_dropout=[15,16], **kwargs)

def _small_low_vgg_16_gen(io, model_name, **kwargs):
    """ 
        The VGG-16 variants with smaller number of outputs at each layer,
        but the same number of basis filters at each layer as the original low
        rank VGG-16 model.  

        We also turn off dropout.

        VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        last_low_rank specifies the last layer to perform low-rank factorization. 
    """
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:32,8:32,9:32,10:32,11:32,12:32}, 
                    'fc':{13:64,14:64}, 'output': {15:16}}

    # specify number of outputs
    num_outputs = {'conv':{0:64,1:64,2:128,3:128,4:256,5:256,6:256,
                           7:512,8:512,9:512,10:512,11:512,12:512}, 
                    'fc':{13:4096,14:4096}}

    max_num_conv = kwargs['max_num_conv']
    num_fc = kwargs['num_fc']
    # change the num_outputs for conv layers
    for k, v in num_outputs['conv'].iteritems():
        num_outputs['conv'][k] = min(max_num_conv, v)
    # change the num_outputs for fc layers
    for k in num_outputs['fc'].keys():
        num_outputs['fc'][k] = num_fc

    return _low_vgg_16_base(io, model_name, 
        num_filters=num_filters, num_outputs=num_outputs, include_dropout=[], **kwargs)

def _tiny_low_vgg_16_gen(io, model_name, **kwargs):
    """ 
        The VGG-16 variants with smaller number of outputs at each layer,
        and the number of basis filter for conv layers are all set to 8, 
        and those for fc layers are all set to 16.   

        We also turn off dropout.

        VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        last_low_rank specifies the last layer to perform low-rank factorization. 
    """
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:8,3:8,4:8,5:8,6:8,
                           7:8,8:8,9:8,10:8,11:8,12:8}, 
                    'fc':{13:16,14:16}, 'output': {15:16}}

    # specify number of outputs
    num_outputs = {'conv':{0:64,1:64,2:128,3:128,4:256,5:256,6:256,
                           7:512,8:512,9:512,10:512,11:512,12:512}, 
                    'fc':{13:4096,14:4096}}

    max_num_conv = kwargs['max_num_conv']
    num_fc = kwargs['num_fc']
    # change the num_outputs for conv layers
    for k, v in num_outputs['conv'].iteritems():
        num_outputs['conv'][k] = min(max_num_conv, v)
    # change the num_outputs for fc layers
    for k in num_outputs['fc'].keys():
        num_outputs['fc'][k] = num_fc

    return _low_vgg_16_base(io, model_name, 
        num_filters=num_filters, num_outputs=num_outputs, include_dropout=[], **kwargs)

def _sp_low_vgg_16_gen(io, model_name, **kwargs):
    """ 
        The VGG-16 variants with smaller number of outputs at each layer,
        and the model is full rank

        We also turn off dropout.

        VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        last_low_rank specifies the last layer to perform low-rank factorization. 
    """
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:32,8:32,9:32,10:32,11:32,12:32}, 
                    'fc':{13:64,14:64}, 'output': {15:16}}

    # specify number of outputs
    num_outputs = {'conv':{0:32,1:64,2:128,3:128,4:224,5:224,6:224,
                           7:224,8:256,9:256,10:256,11:288,12:288}, 
                    'fc':{13:576,14:768}}

    return _low_vgg_16_base(io, model_name, 
        num_filters=num_filters, num_outputs=num_outputs, include_dropout=[], **kwargs)

def _sp1_low_vgg_16_gen(io, model_name, **kwargs):
    """ 
        The VGG-16 variants with smaller number of outputs at each layer,
        and the model is full rank

        We also turn off dropout.

        VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        last_low_rank specifies the last layer to perform low-rank factorization. 
    """
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:32,8:32,9:32,10:32,11:32,12:32}, 
                    'fc':{13:64,14:64}, 'output': {15:16}}

    # specify number of outputs
    num_outputs = {'conv':{0:32,1:32,2:32,3:32,4:32,5:32,6:32,
                           7:64,8:64,9:160,10:160,11:320,12:320}, 
                    'fc':{13:1280,14:1280}}

    return _low_vgg_16_base(io, model_name, 
        num_filters=num_filters, num_outputs=num_outputs, include_dropout=[], **kwargs)   

def _low_vgg_16_base(io, model_name, **kwargs):
    """ VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        last_low_rank specifies the first layer to perform low-rank factorization. 

        This function provides options to control aspects such as basis filters and
        number of parameters in each layer. 
    """

    if kwargs.has_key('last_low_rank'):
        last_low_rank = kwargs['last_low_rank']
    else:
        last_low_rank = 0

    if kwargs.has_key('share_basis'):
        share_basis = kwargs['share_basis']
    else:
        # by default, share basis
        share_basis = True

    if kwargs.has_key('use_bn'):
        use_bn = kwargs['use_bn']
    else:
        use_bn = False

    num_layers = 15     # not counting the last task layer

    assert kwargs.has_key('num_filters'), 'Please specify number of filters.'
    # parameters for convolutional layers
    num_filters = kwargs['num_filters']

    # only layers above (including) last_low_rank should have low-rank factorization 
    num_filters['conv'] = {k:v for k,v in num_filters['conv'].iteritems() if k<last_low_rank}
    num_filters['fc'] = {k:v for k,v in num_filters['fc'].iteritems() if k<last_low_rank}
    num_filters['output'] = {k:v for k,v in num_filters['output'].iteritems() if k<last_low_rank}

    assert kwargs.has_key('num_outputs')
    # specify number of outputs
    num_outputs = kwargs['num_outputs']

    # specify the reference num_outputs, and then compute max_columns
    ref_num_outputs = {'conv':{0:64,1:64,2:128,3:128,4:256,5:256,6:256,
                           7:512,8:512,9:512,10:512,11:512,12:512}, 
                    'fc':{13:4096,14:4096}}

    from math import floor
    max_columns = [1 for _ in xrange(num_layers+1)]
    for name in num_outputs.keys():
        for idx in num_outputs[name].keys():
            max_columns[idx+1] = max(int(floor(ref_num_outputs[name][idx]/num_outputs[name][idx])), 1)
    col_costs = [0., 32./64, 32./64, 16./64, 16./64, 8./64, 8./64, 8./64, 4./64, 4./64, 4./64, 2./64, 2./64, 2./64, 1./64, 1./64]

    # set up col_config
    col_config = {'max_columns': max_columns, 'col_costs': col_costs}

    conv_k = [3 for _ in xrange(13)]
    conv_ks = [1 for _ in xrange(13)]
    conv_pad = [1 for _ in xrange(13)]
    # parameters for pooling
    include_pooling = [1,3,6,9,12]
    pool_k = [2 for _ in xrange(13)]
    pool_ks = [2 for _ in xrange(13)]
    pool_pad = [0 for _ in xrange(13)]
    # parameters for dropout (by default, no dropout is used)
    if kwargs.has_key('include_dropout'):
        include_dropout = kwargs['include_dropout']
    else:
        include_dropout = []
    dropout_ratio = [0.5,0.5]
    # no lrn is used
    include_lrn = []

    model = ModelsLowRank(model_name=model_name, io=io, num_layers=num_layers, col_config=col_config,
        num_filters=num_filters, num_outputs=num_outputs, conv_k=conv_k, conv_ks=conv_ks, conv_pad=conv_pad,
        include_pooling=include_pooling, pool_k=pool_k, pool_ks=pool_ks, pool_pad=pool_pad,
        include_dropout=include_dropout, dropout_ratio=dropout_ratio,
        include_lrn=include_lrn, share_basis=share_basis, use_bn=use_bn)

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
        if i >= last_low_rank:
            param_mapping[(model.branch_name_at_i_j_k(i, 0, 0), )] = value
        else:
            param_mapping[(model.col_name_at_i_j(i, 0), \
                model.branch_name_at_i_j_k(i, 0, 0))] = value

    # specify the number of parameters for the next output
    fit_params = OrderedDict([(vanila_vgg_16_names(i), num_outputs[model.layer_type(i)][i]) for i in xrange(num_layers)])

    # support network with multiple specialist branches
    new_branches = []
    if kwargs.has_key('cut_depth') and kwargs.has_key('cut_points'):
        if kwargs['cut_depth'] > 0 and kwargs['cut_points'] is not None:
            model, param_mapping, new_branches =\
                 _cut_network(model, param_mapping, kwargs['cut_depth'], kwargs['cut_points'])

    return model, param_mapping, new_branches, fit_params
