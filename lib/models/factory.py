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
    path='': _vgg_m_gen(io=io, model_name=model_name, path=path))

# vgg-16 low-rank model 
__models['low-vgg_16'] = (lambda io, model_name='low-vgg-16', 
    path='': _vgg_16_gen(io=io, model_name=model_name, path=path))


def get_models(name, kwargs):
    """ Get an model (network architecture) by name."""
    if not __models.has_key(name):
        raise KeyError('Unknown model: {}'.format(name))
    return __models[name](**kwargs)

def list_models():
    """ List all registred models."""
    return __models.keys()

def get_models_dir():
    """ Default paths for all models """
    return osp.abspath(osp.join(osp.dirname(__file__), '..', '..','models','cache'))

def _vgg_m_gen(io, model_name, path):
    """ VGG-M with two fully connected layers. This is the default model """
    model = ModelsLowRank(model_name=model_name, path=path, io=io)
    return model

def _vgg_16_gen(io, model_name, path):
    """ VGG-16 -> 13 fully connected layers and 2 fully connected layers 
        
        VGG-16 defines the number of outputs at each layer, but it does
        not define the number of filters.

        If the number of filters is set to k=0.9*num_output, then the number
        of parameters is exactly the same as the original model.
        Let's heuristically use 1/8 num_output, so we only have 13.8% of the 
        original number of paramters. 
    """
    num_layers = 15     # not counting the last task layer
    # parameters for convolutional layers
    num_filters = {'conv':{0:8,1:8,2:16,3:16,4:32,5:32,6:32,
                           7:64,8:64,9:64,10:64,11:64,12:64}, 
                    'fc':{13:512,14:512}}
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

    return model


if __name__ == '__main__':

    # TODO: test the models, see if they are correct. 
    pass
