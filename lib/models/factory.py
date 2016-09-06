# Written by Yongxi Lu

""" Model registry, all valid models are registred in this module. """

from .netmodel import reduce_param_mapping
from .models_low_rank import ModelsLowRank
from .model_io import MultiLabelIO, SingleLabelIO
from solvers.classification_sw import MultiLabelSW, SingleLabelSW
import os.path as osp

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

# vgg-m low-rank model
__models['low-vgg-m'] = (lambda io, model_name='low-vgg-m', 
    path='.', **kwargs: 
    _low_vgg_m_gen(io=io, model_name=model_name, **kwargs))

# vgg-16 low-rank model 
__models['low-vgg-16'] = (lambda io, model_name='low-vgg-16', 
    path='.', **kwargs: 
    _low_vgg_16_gen(io=io, model_name=model_name, **kwargs))

# narrow vgg-16 low-rank model 
__models['narrow-low-vgg-16'] = (lambda io, model_name='narrow-low-vgg-16', 
    path='.', **kwargs: 
    _narrow_low_vgg_16_gen(io=io, model_name=model_name, **kwargs))

def get_models(name, **kwargs):
    """ Get an model (network architecture) by name."""
    if not __models.has_key(name):
        raise KeyError('Unknown model: {}'.format(name))
    return __models[name](**kwargs)

def list_models():
    """ List all registred models."""
    return __models.keys()

def _cut_network(model, param_mapping, cut_depth, cut_points):
    """ Cut the netwrok along two sets of tasks """
    # support network with two specialist branches
    num_layers = model.num_layers
    if cut_depth > 0:
        mappings = [param_mapping]
        for cur_layer in xrange(num_layers, num_layers-cut_depth, -1):
            # first cut
            if cur_layer == num_layers:
                new_mapping, new_branches = model.insert_branch((cur_layer, 0), cut_points)
            else:
                new_mapping, new_branches = model.insert_branch((cur_layer, 0), [[0],[1]])
            mappings.append(new_mapping)
        # convert to a unified param_mapping
        param_mapping = reduce_param_mapping(mappings)

    return model, param_mapping, new_branches

def _low_vgg_m_gen(io, model_name, **kwargs):
    """ Low rank VGG-M with two fully connected layers. """

    if kwargs.has_key('last_low_rank'):
        last_low_rank = kwargs['last_low_rank']
    else:
        last_low_rank = 8

    if kwargs.has_key('use_mdc'):
        use_mdc = kwargs['use_mdc']
    else:
        use_mdc = False

    if kwargs.has_key('share_basis'):
        share_basis = kwargs['share_basis']
    else:
        share_basis = False

    num_filters = {'conv': {0:12, 1:32, 2:64, 3:64, 4:64}, 
                    'fc': {5: 512, 6: 512}, 'output': {7:512}}
    num_filters['conv'] = {k:v for k,v in num_filters['conv'].iteritems() if k<last_low_rank}
    num_filters['fc'] = {k:v for k,v in num_filters['fc'].iteritems() if k<last_low_rank}
    num_filters['output'] = {k:v for k,v in num_filters['output'].iteritems() if k<last_low_rank}

    model = ModelsLowRank(model_name=model_name, io=io, num_filters=num_filters, 
        use_mdc=use_mdc, share_basis=share_basis)

    param_mapping = {}
    for i in xrange(model.num_layers):
        value = 'conv{}'.format(i+1)
        net_names = model.names_at_i_j(i, 0)
        if i >= last_low_rank:
            param_mapping[(model.branch_name_at_i_j_k(i, 0, 0), )] = value
        else:
            param_mapping[(model.col_name_at_i_j(i, 0), \
                model.branch_name_at_i_j_k(i, 0, 0))] = value

    # support network with two specialist branches
    new_branches = []
    if kwargs.has_key('cut_depth') and kwargs.has_key('cut_points'):
        if kwargs['cut_depth'] > 0 and kwargs['cut_points'] is not None:
            model, param_mapping, new_branches =\
                 _cut_network(model, param_mapping, kwargs['cut_depth'], kwargs['cut_points'])

    return model, param_mapping, new_branches

def _low_vgg_16_gen(io, model_name, **kwargs):
    """ VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        
        VGG-16 defines the number of outputs at each layer, but it does
        not define the number of filters.

        If the number of filters is set to k=0.9*num_output, then the number
        of parameters is exactly the same as the original model.
        Let's heuristically use 1/8 num_output, so we only have 13.8% of the 
        original number of paramters. 

        last_low_rank specifies the first layer to perform low-rank factorization. 
    """
    if kwargs.has_key('last_low_rank'):
        last_low_rank = kwargs['last_low_rank']
    else:
        last_low_rank = 0

    if kwargs.has_key('use_mdc'):
        use_mdc = kwargs['use_mdc']
    else:
        use_mdc = False

    if kwargs.has_key('share_basis'):
        share_basis = kwargs['share_basis']
    else:
        share_basis = False

    num_layers = 15     # not counting the last task layer
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:64,8:64,9:64,10:64,11:64,12:64}, 
                    'fc':{13:512,14:512}, 'output': {15:512}}

    # only layers above (including) last_low_rank should have low-rank factorization 
    num_filters['conv'] = {k:v for k,v in num_filters['conv'].iteritems() if k<last_low_rank}
    num_filters['fc'] = {k:v for k,v in num_filters['fc'].iteritems() if k<last_low_rank}
    num_filters['output'] = {k:v for k,v in num_filters['output'].iteritems() if k<last_low_rank}

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

    model = ModelsLowRank(model_name=model_name, io=io, num_layers=num_layers,
        num_filters=num_filters, num_outputs=num_outputs, conv_k=conv_k, conv_ks=conv_ks, conv_pad=conv_pad,
        include_pooling=include_pooling, pool_k=pool_k, pool_ks=pool_ks, pool_pad=pool_pad,
        include_dropout=include_dropout, dropout_ratio=dropout_ratio,
        include_lrn=include_lrn, use_mdc=use_mdc, share_basis=share_basis)

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

    # support network with two specialist branches
    new_branches = []
    if kwargs.has_key('cut_depth') and kwargs.has_key('cut_points'):
        if kwargs['cut_depth'] > 0 and kwargs['cut_points'] is not None:
            model, param_mapping, new_branches =\
                 _cut_network(model, param_mapping, kwargs['cut_depth'], kwargs['cut_points'])

    return model, param_mapping, new_branches


def _narrow_low_vgg_16_gen(io, model_name, **kwargs):
    """ VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        last_low_rank specifies the first layer to perform low-rank factorization. 
    """
    if kwargs.has_key('last_low_rank'):
        last_low_rank = kwargs['last_low_rank']
    else:
        last_low_rank = 0

    if kwargs.has_key('use_mdc'):
        use_mdc = kwargs['use_mdc']
    else:
        use_mdc = False

    if kwargs.has_key('share_basis'):
        share_basis = kwargs['share_basis']
    else:
        share_basis = False

    num_layers = 15     # not counting the last task layer
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:64,8:64,9:64,10:64,11:64,12:64}, 
                    'fc':{13:64,14:64}, 'output': {15:16}}

    # only layers above (including) last_low_rank should have low-rank factorization 
    num_filters['conv'] = {k:v for k,v in num_filters['conv'].iteritems() if k<last_low_rank}
    num_filters['fc'] = {k:v for k,v in num_filters['fc'].iteritems() if k<last_low_rank}
    num_filters['output'] = {k:v for k,v in num_filters['output'].iteritems() if k<last_low_rank}

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

    model = ModelsLowRank(model_name=model_name, io=io, num_layers=num_layers,
        num_filters=num_filters, num_outputs=num_outputs, conv_k=conv_k, conv_ks=conv_ks, conv_pad=conv_pad,
        include_pooling=include_pooling, pool_k=pool_k, pool_ks=pool_ks, pool_pad=pool_pad,
        include_dropout=include_dropout, dropout_ratio=dropout_ratio,
        include_lrn=include_lrn, use_mdc=use_mdc, share_basis=share_basis)

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

    # support network with two specialist branches
    new_branches = []
    if kwargs.has_key('cut_depth') and kwargs.has_key('cut_points'):
        if kwargs['cut_depth'] > 0 and kwargs['cut_points'] is not None:
            model, param_mapping, new_branches =\
                 _cut_network(model, param_mapping, kwargs['cut_depth'], kwargs['cut_points'])

    return model, param_mapping, new_branches