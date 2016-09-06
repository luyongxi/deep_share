# Written by Yongxi Lu

""" Maintain I/O layers for interested neural net models """

import layer_helpers as lh
import caffe

class ModelIO(object):
    """Base class for an I/O model that are uesd in NetModel class """

    def __init__(self):
        self._num_tasks = 0
        self._data_name = ''
        self._postfix = ''
        # name of the labels
        self._label_names = None
 
    @property
    def label_names(self):
        return self._label_names
    
    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def data_name(self):
        return self._data_name

    @property
    def postfix(self):
        return self._postfix

    def add_input(self, net, deploy=False):
        """ add input layers """
        return NotImplementedError

    def add_output(self, net, bottom_dict, deploy=False, use_mdc=False, share_basis=False):
        """ add output layers """
        return NotImplementedError

    def col_name_at_j(self, j):
        """ provide the name of column """
        return NotImplementedError

    def branch_name_at_j_k(self, j, k):
        """ provide the name of a branch """
        return NotImplementedError

class ClassificationIO(ModelIO):
    """ IO base-class for grouping different outputs in the a classification problem as tasks"""

    def __init__(self, class_list, data_name, label_names, postfix, loss_layer):

        ModelIO.__init__(self)
        self._class_list = class_list
        self._num_tasks = len(class_list)
        self._data_name = data_name
        self._label_names = label_names
        self._postfix = postfix

        self._loss_layer = loss_layer

    @property
    def class_list(self):
        return self._class_list

    @property
    def loss_layer(self):
        return self._loss_layer     

    def add_output(self, net, bottom_dict, num_filters=None, deploy=False, use_mdc=False, share_basis=False):
        """ add output layers """
        # bottom_dict[k] is a tuple (num_tasks_at(i,k), bottom[k])
        # this determines the number of fc layers needed. 

        use_basis = (num_filters is not None)
        task_layer_list = []
        for j in xrange(len(bottom_dict)):
            if use_basis:
                # basis_name = 'score_basis_{}'.format(j+1) + self.postfix
                # basis_name = 'score_basis' + self._post_fix_at(j)
                basis_name = self.col_name_at_j(j)
                if share_basis:
                    blob_param_name = basis_name.split('_')[0]
                else:
                    blob_param_name = basis_name
                param_names = {'weights': blob_param_name+'_w', 'bias': blob_param_name+'_b'}
                lh.add_fc(net, bottom=bottom_dict[j][1], name=basis_name, param_name=param_names, 
                    nout=num_filters, lr_factor=1, std='linear')
                bottom=net[basis_name]
            else:
                bottom=bottom_dict[j][1]

            # each task gets its own layer
            for k in xrange(bottom_dict[j][0]): 
                # blob_name = 'score_fc' + self._post_fix_at(j, k)
                blob_name = self.branch_name_at_j_k(j,k)
                filter_names = {'weights': blob_name+'_w', 'bias': blob_name+'_b'}
                lh.add_fc(net, bottom=bottom, name=blob_name, param_name=filter_names, 
                    nout=1, lr_factor=1, std='ReLu', use_mdc=use_mdc)
                task_layer_list.append(net[blob_name])

        self.add_loss(net, task_layer_list, deploy)

    def add_loss(self, net, task_layer_list, deploy):
        """ Add the loss layers """
        return NotImplementedError

    def _post_fix_at(self, j, k=None):
        """ Output post fix for the names at column j, [branch k] 
            Use 1-based indexing. 
        """
        if k is None:
            return '{}'.format(j+1) + self.postfix
        else:
            return '{}_{}'.format(j+1,k+1) + self.postfix

    def col_name_at_j(self, j):
        """ provide the name of column """
        return "score_basis" + self._post_fix_at(j)

    def branch_name_at_j_k(self, j, k):
        """ provide the name of a branch """
        return 'score_fc' + self._post_fix_at(j, k)

class MultiLabelIO(ClassificationIO):
    """ IO for grouping different outputs in the a multi-label classification problem as tasks  """

    def __init__(self, class_list, data_name='data', label_names='label', postfix='', loss_layer='Sigmoid'):

        ClassificationIO.__init__(self, class_list, data_name, label_names, postfix, loss_layer)

    def add_input(self, net, deploy=False):
        """ add input layers """
        class_list = self.class_list
        num_classes = len(class_list)

        if not deploy:
            train_net = net['train']
            val_net = net['val']
            lh.add_multilabel_data_layer(train_net, name=[self.data_name, self.label_names], 
                phase=caffe.TRAIN, num_classes=num_classes, class_list=class_list)
            lh.add_multilabel_data_layer(val_net, name=[self.data_name, self.label_names], 
                phase=caffe.TEST, num_classes=num_classes, class_list=class_list)

    def add_loss(self, net, task_layer_list, deploy):
        """ Add the loss layers """        
        # concatenate layers in the order specified by task_layer_list, compute the sigmoid
        lh.add_concat(net, bottom=task_layer_list, name='score'+self.postfix, axis=1)
        lh.add_sigmoid(net, bottom=net['score'+self.postfix], name='prob'+self.postfix, in_place=False)
        if not deploy:
            if self.loss_layer == 'Sigmoid':
                lh.add_sigmoid_entropy_loss(net, bottom=[net['score'+self.postfix], net[self.label_names]], 
                    name='loss'+self.postfix, loss_weight=1.0, phase=caffe.TRAIN)
            elif self.loss_layer == 'Square':
                lh.add_euclidean_loss(net, bottom=[net['prob'+self.postfix], net[self.label_names]], 
                    name='loss'+self.postfix, loss_weight=1.0, phase=caffe.TRAIN)
            else:
                print 'The layer type {} is not recognized!'.format(self.loss_layer)
                raise
            
            lh.add_multilabel_err_layer(net, bottom=[net['prob'+self.postfix], net[self.label_names]], 
                name='error'+self.postfix)

class SingleLabelIO(ClassificationIO):
    """ IO for grouping different outputs in the a single-label classification problem as tasks"""

    def __init__(self, class_list, data_name='data', label_names='label', postfix='', loss_layer='Softmax'):

        ClassificationIO.__init__(self, class_list, data_name, label_names, postfix, loss_layer)

    def add_input(self, net, deploy=False):
        """ add input layers """
        class_list = self.class_list
        num_classes = len(class_list)

        if not deploy:
            train_net = net['train']
            val_net = net['val']
            lh.add_singlelabel_data_layer(train_net, name=[self.data_name, self.label_names], 
                phase=caffe.TRAIN, num_classes=num_classes, class_list=class_list)
            lh.add_singlelabel_data_layer(val_net, name=[self.data_name, self.label_names], 
                phase=caffe.TEST, num_classes=num_classes, class_list=class_list)

    def add_loss(self, net, task_layer_list, deploy):
        """ Add the loss layers """        
        # concatenate layers in the order specified by task_layer_list, compute the sigmoid
        lh.add_concat(net, bottom=task_layer_list, name='score'+self.postfix, axis=1)
        lh.add_softmax(net, bottom=net['score'+self.postfix], name='prob'+self.postfix, in_place=False)
        if not deploy:
            if self.loss_layer == 'Softmax':
                lh.add_softmax_loss(net, bottom=[net['score'+self.postfix], net[self.label_names]], 
                    name='loss'+self.postfix, loss_weight=1.0, phase=caffe.TRAIN)
            else:
                print 'The layer type {} is not recognized!'.format(self.loss_layer)
                raise

            lh.add_accuracy_layer(net, bottom=[net['prob'+self.postfix], net[self.label_names]], 
                name='acc'+self.postfix)