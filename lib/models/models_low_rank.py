# --------------------------------
# Written by Yongxi Lu
# --------------------------------

""" Models with Low Rank Factorization. """

from netmodel import NetModel
import layer_helpers as lh

# TODO: basis filter parameters should be shared!

class ModelsLowRank(NetModel):
    """ Low rank network """

    # the default network has 5 conv layers, 2 fully-connected layers.
    # default hyperparameters
    # _default_num_filters = {'conv': {0:12, 1:32, 2:64, 3:64, 4:64}, 'fc': {5: 512, 6: 512}, 'output': 512}
    _default_num_filters = {}    
    _default_num_outputs = {'conv': {0:96, 1:256, 2:512, 3:512, 4:512}, 'fc': {5: 4096, 6: 4096}}
    # default parameters for convolutional layers
    _default_conv_k = [7, 5, 3, 3, 3]   # kernel size
    _default_conv_ks = [2, 2, 1, 1, 1]  # kernel stride
    _default_conv_pad = [0, 1, 1, 1, 1] # zero padding
    # default parameters for pooling layers
    _default_pool_k = [3,3,3]   # kernel size
    _default_pool_ks = [2,2,2]   # kernel stride
    _default_pool_pad = [0,0,0]  # zero padding
    # default dropout parameters
    _default_dropout_ratio = [0.5,0.5]
    # layers that requires special transformations
    _default_include_lrn = [0, 1]
    _default_include_pooling = [0,1,4]
    _default_include_dropout = [5,6]

    def __init__(self, model_name, path, io, num_layers=7, 
        num_filters=_default_num_filters, num_outputs=_default_num_outputs, 
        conv_k=_default_conv_k, conv_ks=_default_conv_ks, conv_pad=_default_conv_pad,
        pool_k=_default_pool_k, pool_ks=_default_pool_ks, pool_pad=_default_pool_pad,
        dropout_ratio=_default_dropout_ratio, include_lrn=_default_include_lrn, 
        include_pooling=_default_include_pooling, include_dropout=_default_include_dropout):
        """
            Initialize a network model with low-rank structure.
            Inputs:
                num_filters: a list (number) that specifies number of basis filters at each layer.
                num_outputs: a list (number) that specifies number of outputs at each layer 
                            (for a single branch).
        """
        # initialize base class
        NetModel.__init__(self, model_name, path, io, num_layers)
        # options in the architecture
        self._num_filters = num_filters
        self._num_outputs = num_outputs
        self._conv_k = conv_k
        self._conv_ks = conv_ks
        self._conv_pad = conv_pad
        self._pool_k = pool_k
        self._pool_ks = pool_ks
        self._pool_pad = pool_pad
        self._dropout_ratio = dropout_ratio
        self._include_lrn = include_lrn
        self._include_pooling = include_pooling
        self._include_dropout = include_dropout

    def _layer_type(self, i):
        """Return type of the layer """
        if i == self.num_layers:
            return 'output'
        elif i >= len(self._num_outputs['conv']):
            return 'fc'
        else:
            return 'conv'

    def _add_input_layers(self, net, deploy):
        """ Add the data layers """
        self.io.add_input(net, deploy=deploy)

        if deploy:
            return net[self.io.data_name]
        else:
            return net['train'][self.io.data_name]

    def _add_output_layers(self, net, bottom_dict, deploy):
        """ Add the loss layers """
        new_bottom_dict = {}
        i=self.num_layers
        for j in xrange(self.num_cols_at(i)):
            new_bottom_dict[j] = (self.num_tasks_at(i,j), bottom_dict[j])

        kwargs=dict(bottom_dict=new_bottom_dict, deploy=deploy)
        if self._num_filters.has_key('output'):
            kwargs['num_filters'] = self._num_filters['output']

        self.io.add_output(net, **kwargs)

    def _add_intermediate_layers(self, net, data):
        """ Add intermediate layers to the network. """
        bottom_dict = {0: data}
        for i in xrange(self.num_layers):
            if self._layer_type(i) == 'conv':
                bottom_dict = self._add_conv_layer_i(net, i, bottom_dict)
            elif self._layer_type(i) == 'fc':
                bottom_dict = self._add_fc_layer_i(net, i, bottom_dict)

        return bottom_dict

    def _add_conv_layer_i(self, net, i, bottom_dict):
        """ Add a convolutional layer at layer i. """
        use_lrn = (i in self._include_lrn)
        use_pooling = (i in self._include_pooling)
        new_bottom_dict = {}
        use_basis = self._num_filters.has_key('conv') and self._num_filters['conv'].has_key(i)
        # [basis]+conv+ReLU+[optional]LRN+[optional]Pool
        for j in xrange(self.num_cols_at(i)):
            names = self.names_at_i_j(i,j)
            # use basis filter only if num_filters is specified for this layer. 
            if use_basis:
                blob_name = names['basis']
                blob_param_name = blob_name.split('_')[0]
                filter_names = {'weights': blob_param_name+'_w', 'bias': blob_param_name+'_b'}
                # basis filter
                lh.add_conv(net, bottom=bottom_dict[j], name=blob_name, param_name=filter_names, k=self._conv_k[i], 
                    ks=self._conv_ks[i], pad=self._conv_pad[i], nout=self._num_filters['conv'][i], std='linear')

            for k in xrange(self.num_branch_at(i,j)):
                br_name = names['agg'][k]
                conv_names = {'weights': br_name+'_w', 'bias': br_name+'_b'}
                if use_basis:
                    # linear combination
                    lh.add_conv(net, bottom=net[blob_name], name=br_name, param_name=conv_names, k=1, 
                        ks=1, pad=0, nout=self._num_outputs['conv'][i], std='ReLu')
                else:
                    lh.add_conv(net, bottom=bottom_dict[j], name=br_name, param_name=conv_names, k=self._conv_k[i], 
                        ks=self._conv_ks[i], pad=self._conv_pad[i], nout=self._num_outputs['conv'][i], std='ReLu')

                # bottom set used for the next layer
                new_bottom_dict[self.tops_at(i,j,k)] = net[br_name]
                # add ReLu
                lh.add_relu(net, bottom=net[br_name], name='relu'+self._post_fix_at(i,j,k))
                # add LRN if necessary
                if use_lrn:
                    lh.add_lrn(net, bottom=net[br_name], name='norm'+self._post_fix_at(i,j,k), 
                        local_size=5, alpha=0.0005, beta=0.75, k=2)
                # add max pooling if necessary
                if use_pooling:
                    pool_name = 'pool'+self._post_fix_at(i,j,k)
                    pool_idx = self._include_pooling.index(i)
                    lh.add_maxpool(net, bottom=net[br_name], name=pool_name, 
                        k=self._pool_k[pool_idx], ks=self._pool_ks[pool_idx], 
                        pad=self._pool_pad[pool_idx])
                    # if pooling is used, the feature map has a different name
                    new_bottom_dict[self.tops_at(i,j,k)] = net[pool_name]

        return new_bottom_dict

    def _add_fc_layer_i(self, net, i, bottom_dict):
        """ Add a fully connected layer at layer i. """
        use_dropout = (i in self._include_dropout)
        new_bottom_dict = {}
        use_basis = self._num_filters.has_key('fc') and self._num_filters['fc'].has_key(i)
        # [basis]+conv+ReLU+[optional]LRN+[optional]Pool
        for j in xrange(self.num_cols_at(i)):
            names = self.names_at_i_j(i,j)
            # use basis filter only if num_filters is specified for this layer. 
            if use_basis:
                blob_name = names['basis']
                blob_param_name = blob_name.split('_')[0]
                filter_names = {'weights': blob_param_name+'_w', 'bias': blob_param_name+'_b'}
                # basis filter
                lh.add_fc(net, bottom=bottom_dict[j], name=blob_name, param_name=filter_names, 
                    nout=self._num_filters['fc'][i], std='linear')

            for k in xrange(self.num_branch_at(i,j)):
                br_name = names['agg'][k]
                conv_names = {'weights': br_name+'_w', 'bias': br_name+'_b'}
                if use_basis:
                    # linear combination
                    lh.add_fc(net, bottom=net[blob_name], name=br_name, param_name=conv_names,
                        nout=self._num_outputs['fc'][i], std='ReLu')
                else:
                    lh.add_fc(net, bottom=bottom_dict[j], name=br_name, param_name=conv_names, 
                        nout=self._num_outputs['fc'][i], std='ReLu')                    
                # bottom set used for the next layer
                new_bottom_dict[self.tops_at(i,j,k)] = net[br_name]
                # add ReLu
                lh.add_relu(net, bottom=net[br_name], name='relu'+self._post_fix_at(i,j,k))
                # add Dropout if necessary
                if use_dropout:
                    drop_idx = self._include_dropout.index(i)
                    lh.add_dropout(net, bottom=net[br_name], name='drop'+self._post_fix_at(i,j,k),
                        dropout_ratio=self._dropout_ratio[drop_idx])

        return new_bottom_dict

    def _post_fix_at(self, i, j, k=None):
        """ Output post fix for the names at layer i, column j, [branch k] 
            Use 1-based indexing. 
        """
        if k is None:
            return '{}_{}'.format(i+1,j+1)
        else:
            return '{}_{}_{}'.format(i+1,j+1,k+1)

    def names_at_i_j(self, i, j):
        """ Return the name of the parameters at layer i, column j.
            This function depends on the exact network architecture
            
            Naming conventions: 
            Weights will have a postfix '_w', bias will have a postfix '_b'
            The indexing of the names are 1-based.
        """
        basis_param = 'basis'+self._post_fix_at(i,j)
        if self._layer_type(i) == 'conv':
            conv_param = ['conv'+self._post_fix_at(i,j,k) for k in xrange(self.num_branch_at(i,j))]            
            names = {'basis': basis_param, 'agg': conv_param}
        elif self._layer_type(i) == 'fc':
            fc_param = ['fc'+self._post_fix_at(i,j,k) for k in xrange(self.num_branch_at(i,j))]            
            names = {'basis': basis_param, 'agg': fc_param}
        return names

    def update_deploy_net(self):
        """ Update deploy net. """
        data = self._add_input_layers(self.deploy_net, deploy=True)
        bottom_dict = self._add_intermediate_layers(self.deploy_net, data)
        self._add_output_layers(self.deploy_net, bottom_dict, deploy=True)

    def update_trainval_net(self):
        """ Update trainval net. """
        in_nets = {'train': self.train_net, 'val': self.val_net}
        data = self._add_input_layers(in_nets, deploy=False)
        bottom_dict = self._add_intermediate_layers(self.train_net, data)
        self._add_output_layers(self.train_net, bottom_dict, deploy=False)

if __name__ == '__main__':
    import os
    import os.path as osp
    from model_io import MultiLabelIO

    # Save default model
    io = MultiLabelIO(class_list=[0,1,2,3])
    default = ModelsLowRank(model_name='default_5-layer', path='', io=io)
    default.to_proto(deploy=False)
    default.to_proto(deploy=True)

    # Create a branch
    branch = ModelsLowRank(model_name='branch_5-layer', path='', io=io)
    changes = branch.insert_branch((7,0), [[0,2],[1,3]])
    print changes['blobs']
    print changes['branches']
    # Save new model
    branch.to_proto(deploy=False)
    branch.to_proto(deploy=True)