# --------------------------------
# Written by Yongxi Lu
# --------------------------------

""" Models with Low Rank Factorization. """

from netmodel import NetModel
import layer_helpers as lh

class ModelsLowRank(NetModel):
    """ Low rank network """

    # the default network has 5 conv layers, 2 fully-connected layers.
    # default hyperparameters
    _default_num_filters = {'conv': {0:32, 1:32, 2:32, 3:32, 4:32}, 'fc': {5: 256, 6: 256}}
    _default_num_outputs = {'conv': {0:96, 1:256, 2:512, 3:512, 4:512}, 'fc': {5: 4096, 6: 4096}}
    # default parameters for convolutional layers
    _default_conv_k = [7, 5, 3, 3, 3]   # kernel size
    _default_conv_ks = [2, 2, 1, 1, 1]  # kernel stride
    # layers that requires special transformations
    _default_include_lrn = [0, 1]
    _default_include_pooling = [0,1,4]
    _default_include_dropout = [5, 6]

    def __init__(self, model_name, solver_path, io, num_layers=7, 
        num_filters=_default_num_filters, num_outputs=_default_num_outputs, 
        conv_k=_default_conv_k, conv_ks=_default_conv_ks, include_lrn=_default_include_lrn, 
        include_pooling=_default_include_pooling, include_dropout=_default_include_dropout):
        """
            Initialize a network model with low-rank structure.
            Inputs:
                num_filters: a list (number) that specifies number of basis filters at each layer.
                num_outputs: a list (number) that specifies number of outputs at each layer 
                            (for a single branch).
        """
        # initialize base class
        NetModel.__init__(self, model_name, solver_path, io, num_layers)
        # options in the architecture
        self._num_filters = num_filters
        self._num_outputs = num_outputs
        self._conv_k = conv_k
        self._conv_ks = conv_ks
        self._include_lrn = include_lrn
        self._include_pooling = include_pooling
        self._include_dropout = include_dropout

    def _layer_type(self, i):
        """Return type of the layer """
        if i >= len(self._num_outputs['conv']):
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
        self.io.add_output(net, bottom_dict=bottom_dict, deploy=deploy)

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
        # basis+agg+ReLU+[optional]LRN+[optional]Pool
        for j in xrange(self.num_cols_at(i)):
            names = self.names_at_i_j(i,j)
            blob_name = names['basis']
            filter_names = {'weights': blob_name+'_w', 'bias': blob_name+'_b'}
            # basis filter
            lh.add_conv(net, bottom=bottom_dict[j], name=blob_name, param_name=filter_names, k=self._conv_k[i], 
                ks=self._conv_ks[i], pad=0, nout=self._num_filters['conv'][i])
            for k in xrange(self.num_branch_at(i,j)):
                br_name = names['agg'][k]
                agg_names = {'weights': br_name+'_w', 'bias': br_name+'_b'}
                # linear combination
                lh.add_conv(net, bottom=net[blob_name], name=br_name, param_name=agg_names, k=1, 
                    ks=1, pad=0, nout=self._num_outputs['conv'][i])
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
                    lh.add_maxpool(net, bottom=net[br_name], name=pool_name, k=2, ks=2, pad=0)
                    # if pooling is used, the feature map has a different name
                    new_bottom_dict[self.tops_at(i,j,k)] = net[pool_name]

        return new_bottom_dict

    def _add_fc_layer_i(self, net, i, bottom_dict):
        """ Add a fully connected layer at layer i. """
        use_dropout = (i in self._include_dropout)
        new_bottom_dict = {}
        # basis+agg+ReLU+[optional]LRN+[optional]Pool
        for j in xrange(self.num_cols_at(i)):
            names = self.names_at_i_j(i,j)
            blob_name = names['basis']
            filter_names = {'weights': blob_name+'_w', 'bias': blob_name+'_b'}
            # basis filter
            lh.add_fc(net, bottom=bottom_dict[j], name=blob_name, param_name=filter_names, 
                nout=self._num_filters['fc'][i])
            for k in xrange(self.num_branch_at(i,j)):
                br_name = names['agg'][k]
                agg_names = {'weights': br_name+'_w', 'bias': br_name+'_b'}
                # linear combination
                lh.add_fc(net, bottom=net[blob_name], name=br_name, param_name=agg_names,
                    nout=self._num_outputs['fc'][i])
                # bottom set used for the next layer
                new_bottom_dict[self.tops_at(i,j,k)] = net[br_name]
                # add ReLu
                lh.add_relu(net, bottom=net[br_name], name='relu'+self._post_fix_at(i,j,k))
                # add Dropout if necessary
                if use_dropout:
                    lh.add_dropout(net, bottom=net[br_name], name='drop'+self._post_fix_at(i,j,k))

        return new_bottom_dict

    def _post_fix_at(self, i, j, k=None):
        """ Output post fix for the names at layer i, column j, [branch k] 
            Use 1-based indexing. 
        """
        if k is None:
            return '{}-{}'.format(i+1,j+1)
        else:
            return '{}-{}-{}'.format(i+1,j+1,k+1)

    def names_at_i_j(self, i, j):
        """ Return the name of the parameters at layer i, column j.
            This function depends on the exact network architecture
            
            Naming conventions: 
            Weights will have a postfix '_w', bias will have a postfix '_b'
            The indexing of the names are 1-based.
        """
        basis_param = 'basis'+self._post_fix_at(i,j)
        agg_param = ['agg'+self._post_fix_at(i,j,k) for k in xrange(self.num_branch_at(i,j))]
        names = {'basis': basis_param, 'agg': agg_param}
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

    p1 = osp.join(os.getcwd(), 'default')
    p2 = osp.join(os.getcwd(), 'branch')
    if not osp.isdir(p1):
        os.mkdir(p1)
    if not osp.isdir(p2):
        os.mkdir(p2)

    # Save default model
    io = MultiLabelIO(class_groups=[[0,1],[2,3]])
    default = ModelsLowRank(model_name='5-layer', solver_path=p1, io=io)
    default.save_model_file(deploy=False)
    default.save_model_file(deploy=True)

    # Create a branch
    branch = ModelsLowRank(model_name='5-layer', solver_path=p2, io=io)
    changes = branch.insert_branch((6,0), [[0],[1]])
    print changes['blobs']
    print changes['branches']
    # Save new model
    branch.save_model_file(deploy=False)
    branch.save_model_file(deploy=True)