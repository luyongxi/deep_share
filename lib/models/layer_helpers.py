# Written by Yongxi Lu

""" Helper functions to add layers to the network. """
import caffe
from caffe import layers as L
from caffe import params as P
from math import sqrt
import yaml

def get_init_params(std):
    if std == 'linear':
        weight_filler = {'type': 'xavier'}            
    elif std == 'ReLu':
        weight_filler = {'type': 'msra'}
    else:
        weight_filler = {'type': 'gaussian', 'std': std}

    bias_filler = {'type': 'constant', 'value': 0}
    return weight_filler, bias_filler

def add_conv(net, bottom, name, param_name, k, ks, pad, nout, lr_factor=1, std=0.01, use_mdc=False):
    """Add a convolutional layer """
    # names of the parameters
    param = [{'name': param_name['weights'], 'lr_mult': lr_factor, 'decay_mult': 1}, 
        {'name': param_name['bias'], 'lr_mult': 2*lr_factor, 'decay_mult': 0}]
    if use_mdc:
        param[0]['local_decay_mult'] = lr_factor
        param[0]['local_decay_type'] = "MDC"
        param[0]['local_beta'] = 10
    # weight filler
    weight_filler, bias_filler = get_init_params(std)
    # set up the layer
    net[name] = L.Convolution(bottom, param=param, convolution_param=dict(kernel_size=k, 
        stride=ks, pad=pad, num_output=nout, bias_term=True, weight_filler=weight_filler, 
        bias_filler=bias_filler))

def add_fc(net, bottom, name, param_name, nout, lr_factor=1, std=0.01, use_mdc=False):
    """Add a fully-connected layer """
    param = [{'name': param_name['weights'], 'lr_mult': lr_factor, 'decay_mult': 1}, 
        {'name': param_name['bias'], 'lr_mult': 2*lr_factor, 'decay_mult': 0}]
    if use_mdc:
        param[0]['local_decay_mult'] = lr_factor
        param[0]['local_decay_type'] = "MDC"
        param[0]['local_beta'] = 10
    # weight filler
    weight_filler, bias_filler = get_init_params(std)
    # set up the layer
    net[name] = L.InnerProduct(bottom, param=param, 
        inner_product_param=dict(num_output=nout, weight_filler=weight_filler, 
        bias_filler=bias_filler))

def add_relu(net, bottom, name, in_place=True):
    """Add ReLu activation """
    net[name] = L.ReLU(bottom, in_place=in_place)

def add_maxpool(net, bottom, name, k, ks, pad):
    """Add max pooling layer """
    net[name] = L.Pooling(bottom, kernel_size=k, stride=ks, pad=pad, 
        pool=P.Pooling.MAX)

def add_lrn(net, bottom, name, local_size, alpha, beta, k):
    """Add local response normalizaiton unit """
    net[name] = L.LRN(bottom, local_size=local_size, 
        alpha=alpha, beta=beta, k=k, in_place=True)

def add_dropout(net, bottom, name, dropout_ratio=0.5, in_place=True):
    """ Add dropout layer """
    net[name] = L.Dropout(bottom, dropout_ratio=dropout_ratio, in_place=in_place)

def add_concat(net, bottom, name, axis):
    """ Add a concatenation layer along an axis """
    net[name] = L.Concat(*bottom, axis=axis)

def add_sigmoid(net, bottom, name, in_place=True):
    """Add Sigmoid activation """
    net[name] = L.Sigmoid(bottom, in_place=in_place)

def add_dummy_layer(net, name):
    """Add a dummy data layer """
    net[name] = L.Layer()

def add_multilabel_data_layer(net, name, phase, num_classes, class_list=None):
    """ Add a MultiLabelData layer """
    include_dict = {'phase': phase}
    param = {'num_classes': num_classes}
    if phase == caffe.TRAIN:
        param['stage'] = 'TRAIN'
    elif phase == caffe.TEST:
        param['stage'] = 'VAL'
    if class_list is not None:
        assert len(class_list) == num_classes, \
            'Length of class list does not match number of classes {} vs {}'.\
            format(len(class_list), num_classes)
        param['class_list'] = class_list

    param_str = yaml.dump(param)
    net[name[0]], net[name[1]] = L.Python(name=name[0], python_param=dict(module='layers.multilabel_data', 
        layer='MultiLabelData', param_str=param_str), include=include_dict, ntop=2)

def add_multilabel_err_layer(net, bottom, name):
    """ Add a MultilabelErr layer """
    net[name] = L.Python(bottom[0], bottom[1],  
        python_param=dict(module='layers.multilabel_err', layer='MultiLabelErr'))

def add_euclidean_loss(net, bottom, name, loss_weight, phase):
    """ Add Euclidean Loss """
    include_dict = {'phase': phase}
    net[name] = L.EuclideanLoss(bottom[0], bottom[1], loss_weight=loss_weight, include=include_dict)

def add_sigmoid_entropy_loss(net, bottom, name, loss_weight, phase):
    """ Add sigmoid entropy loss """
    include_dict = {'phase': phase}
    net[name] = L.SigmoidCrossEntropyLoss(bottom[0], bottom[1], loss_weight=loss_weight, include=include_dict)

if __name__ == '__main__':
    net = caffe.NetSpec()
    net_test = caffe.NetSpec()
    add_multilabel_data_layer(net, name=['data', 'label'], phase=caffe.TRAIN, num_classes=2, class_list=[4,24])
    param_name = {'weights': 'conv1_w', 'bias': 'conv1_b'}
    add_conv(net, bottom=net['data'], name='conv1', param_name=param_name, 
        k=3, ks=1, pad=0, nout=128, lr_factor=1)
    add_relu(net, bottom=net['conv1'], name='relu1')
    param_name = {'weights': 'fc1_w', 'bias': 'fc1_b'}    
    add_fc(net, bottom=net['relu1'], name='fc1-1', param_name=param_name, nout=128, lr_factor=1, std=0.01)
    add_fc(net, bottom=net['relu1'], name='fc1-2', param_name=param_name, nout=128, lr_factor=1, std=0.01)
    add_concat(net, bottom=[net['fc1-1'], net['fc1-2']], name='fc1', axis=1)
    add_sigmoid_entropy_loss(net, bottom=[net['data'],net['fc1']], name='loss', loss_weight=1.0, phase=caffe.TRAIN)
    add_multilabel_err_layer(net, bottom=[net['data'], net['fc1']], name='error')

    add_multilabel_data_layer(net_test, name=['data', 'label'], phase=caffe.TEST, num_classes=2, class_list=[4,24])

    with open('function_test.prototxt', 'w') as f:        
        f.write(str(net_test.to_proto()))
        f.write(str(net.to_proto()))