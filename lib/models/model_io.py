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

    def add_output(self, net, bottom_dict, deploy=False):
        """ add output layers """
        return NotImplementedError

class MultiLabelIO(ModelIO):
    """ IO for grouping different outputs in the a multi-label classification problem as tasks"""

    def __init__(self, class_groups, data_name='data', label_names='label', postfix='', loss_layer='Sigmoid'):
        """ 'class_groups' is a list of list, where each sub-list contains indexing to classes in
            some dataset. The group of classes inside each sub-list is a task. The indexing of these set of
            tasks follows the indexing of the sublist in 'class_groups'. 

            We assumes the training labels are provided in the order of classes when we flatten class_groups 

        """
        ModelIO.__init__(self)
        self._class_groups = class_groups
        self._num_tasks = len(class_groups)
        self._data_name = data_name
        self._label_names = label_names
        self._postfix = postfix

        self._loss_layer = loss_layer

    @property
    def class_groups(self):
        return self._class_groups

    @property
    def loss_layer(self):
        return self._loss_layer
    
    def add_input(self, net, deploy=False):
        """ add input layers """
        class_list = [cls for task in self.class_groups for cls in task]
        num_classes = len(class_list)

        if not deploy:
            train_net = net['train']
            val_net = net['val']
            lh.add_multilabel_data_layer(train_net, name=[self.data_name, self.label_names], 
                phase=caffe.TRAIN, num_classes=num_classes, class_list=class_list)
            lh.add_multilabel_data_layer(val_net, name=[self.data_name, self.label_names], 
                phase=caffe.TEST, num_classes=num_classes, class_list=class_list)
        else:
            lh.add_dummy_data_layer(net, name=self.data_name, dim=[1,3,224,224])

    def add_output(self, net, bottom_dict, deploy=False):
        """ add output layers """
        # bottom_dict[k] is the bottom of the task k sub-net
        task_layer_list = []
        for k in xrange(self.num_tasks):
            blob_name = 'score_{}'.format(k+1) + self.postfix
            filter_names = {'weights': blob_name+'_w', 'bias': blob_name+'_b'}
            # number of outputs for a task is the number of labels inside the task. 
            lh.add_fc(net, bottom=bottom_dict[k], name=blob_name, param_name=filter_names, 
                nout=len(self.class_groups[k]), lr_factor=1, std=0.01)
            task_layer_list.append(net[blob_name])

        # concatenate layers in the order specified by task_layer_list, compute the sigmoid
        lh.add_concat(net, bottom=task_layer_list, name='score'+self.postfix, axis=1)
        lh.add_sigmoid(net, bottom=net['score'+self.postfix], name='prob'+self.postfix)
        if not deploy:
            if self.loss_layer == 'Sigmoid':
                lh.add_sigmoid_entropy_loss(net, bottom=[net['score'+self.postfix], net[self.label_names]], 
                    name='loss'+self.postfix, loss_weight=1.0, phase=caffe.TRAIN)
            elif self.loss_layer == 'Square':
                lh.add_square_loss(net, bottom=[net['prob'+self.postfix], net[self.label_names]], 
                    name='loss'+self.postfix, loss_weight=1.0, phase=caffe.TRAIN)
            lh.add_multilabel_err_layer(net, bottom=[net[self.label_names], 
                net['prob'+self.postfix]], name='error'+self.postfix)

# TODO: implement a classification IO
# class ClsIO(ModelIO):


# class StitchSameModalityIO(ModelIO):
#     """ Stich different I/O with the same input data together. """

#     def __init__(self, data_name='data', task_groups, input_layer):
#         """task_groups is a dict, keys are names of task group (used as post-fixes), 
#         values are instances of I/O handlers"""
#         self.ModelIO()
#         self._input_layer = input_layer
#     # To allow flexibility in defining the sampling strategy when there more than one task groups, 
#     # we need to define separated input layers for each case. That suggests the programming logics
#     # of stiching different groups of tasks together are different. 

#     def add_input(self, net, deploy=False):
#         """ add input layers """
#         # Instead of calling the functions at each layer, 
#         # we should summarize their information (since we would rather have one input layer).
#         # Note that we should be able to find the appropriate python layer by simply looking at the type of the concatenation.
#         # We probably need to use different sampling policies etc. for different combinations of task sub-groups. 


#     def add_output(self, net, bottom_dict, deploy=False):
#         """ add output layers """
#         # TODO: implement based on self._task_groups (how?)
#         #
