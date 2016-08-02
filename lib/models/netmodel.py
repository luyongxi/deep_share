# Written by Yongxi Lu

""" A class that represents the network models we are interested in """

import sys
import os.path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..', '..','caffe', 'python')
add_path(caffe_path)

import caffe
import os.path as osp

class NetBlob(object):
    """ A structure that encodes a blob  """
    def __init__(self, top_idx, tasks):        
        """
        top_idx: idx to the blob associated with the branch at its 
            layer
        tasks: a list of lists, each sub-list is the tasks associated
            with a branch.
        """
        self._top_idx = top_idx
        self._tasks = tasks
        assert len(self.tasks)==len(self.top_idx),\
            'Inconsistent top number {} vs {}'.\
            format(len(self.tasks), len(self.top_idx))

        assert self.num_tops()>0, 'A blob must have at least one branch.'

    def add_top(self, top_idx, tasks):
        self.top_idx.extend(top_idx)
        self.tasks.extend(tasks)
        assert len(self.tasks)==len(self.top_idx),\
            'Inconsistent top number {} vs {}'.\
            format(len(self.tasks), len(self.top_idx))

    def set_tasks(self, branch_idx, tasks):
        for i in xrange(len(branch_idx)):
            self.tasks[branch_idx[i]] = tasks[i]

    def is_edge(self):
        """ Is the blob at the edge of the network (has branches)?"""
        return self.num_tops()>1
  
    def num_tops(self):
        return len(self.top_idx)
    
    @property
    def top_idx(self):
        return self._top_idx

    @property
    def tasks(self):
        return self._tasks
  
class NetModel(object):
    """ A model for the neural network """

    def __init__(self, model_name, solver_path, io, num_layers):
        """ Initialize a model for neural network
            Inputs:
                model_name: name of the model
                solver_path: path that contains the solver.prototxt
                io: specifies the input/output, must has the following members
                    num_tasks: number of tasks (determine the number of branches 
                           of the last intermediate layers)
                    data_name: name of the data blob
                    add_input(net, deploy): add input layers 
                    add_output(net, bottom_dict, deploy): add output layers
                num_layers: number of intermediate layers 
                            (not counting inputs and task specific 
                            output layers)
        """
        self._model_name = model_name
        self._solver_path = solver_path
        self._io = io
        self._num_layers = num_layers
        self._init_graph(num_layers, self.num_tasks)

        self._net_specs = {'train': caffe.NetSpec(), 'val': caffe.NetSpec(),
            'deploy': caffe.NetSpec()}

    def _init_graph(self, num_layers, num_tasks):
        """ Initilize the connection graph of blobs """
        self._net_graph = [[] for _ in xrange(num_layers)]
        for i in xrange(num_layers):
            j = num_layers-i-1    # start from the top layer
            if j==num_layers-1:
                # for k in xrange(num_tasks):
                #     self._net_graph[j].append(NetBlob(top_idx=[k], tasks=[[k]]))
                self._net_graph[j].append(NetBlob(top_idx=range(num_tasks), 
                    tasks=[[k] for k in range(num_tasks)]))
            else:
                top_idx = [0]
                # flatten the tasks list at the top blob
                tasks = [[t for b in self._net_graph[j+1] for t in b.tasks]]
                self._net_graph[j].append(NetBlob(top_idx=top_idx, tasks=tasks))

    def insert_branch(self, idx, split):
        """ Create a new branch. 
            Inputs:
                idx: a tuple (layer_idx, col_idx), to insert the branch.
                split: a list with two sub-lists, each are indexes into a set of tops
            Outputs:
                changes: a dict with two keys, 'blobs' and 'branches'
                    Only two situations: 
                    (1) we create two new blobs from a single blob
                    (2) We create two new branches from an old branch
        """
        # check if the inputs are valid
        left = split[0]
        right = split[1]
        tops = self._net_graph[idx[0]][idx[1]].top_idx
        assert set(tops)-set(left)==set(right), 'The splitting does not form a partition of tops.'
        assert idx[0]>0, 'Cannot create new branches at the bottom.'

        # insert a new branch, keep track of the parameters.
        bottoms = self._net_graph[idx[0]-1]
        bottom_idx = [i for i in xrange(len(bottoms)) if idx[1] in bottoms[i].top_idx][0]  # a singleton
        branch_idx = bottoms[i].top_idx.index(idx[1])
        # create new blobs at the current layer
        # save original blob
        orig_blob = self._net_graph[idx[0]][idx[1]]
        top_idx = orig_blob.top_idx 
        tasks = orig_blob.tasks
        # left branch
        self._net_graph[idx[0]][idx[1]] = \
            NetBlob(top_idx=[top_idx[i] for i in left], tasks=[tasks[i] for i in left])
        # right branch
        self._net_graph[idx[0]].append(
            NetBlob(top_idx=[top_idx[i] for i in right], tasks=[tasks[i] for i in right]))
        right_idx = len(self._net_graph[idx[0]])-1
        # add a new branch at the bottom layer
        b_blobs = bottoms[bottom_idx]
        b_blobs.set_tasks(branch_idx=[branch_idx], tasks=[tasks[i] for i in left])
        b_blobs.add_top(top_idx=[right_idx], tasks=[tasks[i] for i in right])
        # log changes
        changes = {'blobs':{}, 'branches':{}}
        changes['blobs'] = {(idx[0],idx[1]): [(idx[0],idx[1]), (idx[0],right_idx)]}
        changes['branches'] = {(idx[0]-1, bottom_idx, branch_idx): 
            [(idx[0]-1, bottom_idx, branch_idx), (idx[0]-1, bottom_idx, b_blobs.num_tops()-1)]}

        return changes
        
    def all_edges(self):
        """ Return a list of tuples, each with a layer idx and
            a column idx. The list indexes the blobs considered
            at the edge (candidate for branching).
        """
        edges = []
        for layer_idx in xrange(len(net_graph)):
            for col_idx in xrange(len(net_graph[layer_idx])):
                # if layer idx is 0, then the layer below is the shared input
                if self._net_graph[layer_idx][col_idx].is_edge() and (layer_idx>0):
                    edges += (layer_idx, col_idx)

    def tasks_at(self, i, j, k=None):
        """ Return the indexing into tasks at layer i, column j, [branch k] """
        if k is None:
            return self._net_graph[i][j].tasks
        else:
            return self._net_graph[i][j].tasks[k]

    def tops_at(self, i, j, k=None):
        """Return the indexing into the top blob of layer i, column j, [branch k] """        
        if k is None:
            return self._net_graph[i][j].top_idx
        else:
            return self._net_graph[i][j].top_idx[k]

    def num_branch_at(self, i, j):
        """ Number of branches at a blob """
        return self._net_graph[i][j].num_tops()

    def num_cols_at(self, i):
        """ Returns number of columns at layer i """
        return len(self._net_graph[i])

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_tasks(self):
        return self._io.num_tasks

    @property
    def solver_path(self):
        """ Will save as solver_path/train_val.prototxt, 
            or solver_path/test.prototxt """
        return self._solver_path

    @property
    def deploy_net(self):
        return self._net_specs['deploy']

    @property
    def train_net(self):
        return self._net_specs['train']

    @property
    def val_net(self):
        return self._net_specs['val']

    @property
    def io(self):
        return self._io

    def save_model_file(self, deploy=False):
        """ Need different inputs for deploy or not """
        if deploy == True:
            self.update_deploy_net()
            fn = osp.join(self.solver_path, 'test.prototxt')
            with open(fn, 'w') as f:
                f.write(str(self.deploy_net.to_proto()))
        else:
            self.update_trainval_net()
            fn = osp.join(self.solver_path, 'train_val.prototxt')
            with open(fn, 'w') as f:
                f.write(str(self.val_net.to_proto()))
                f.write(str(self.train_net.to_proto()))   

    def names_at_i_j(self, i, j):
        """ Return the name of the parameters at layer i, column j.
            This function depends on the exact network architecture
            It could return a string or a dict.
        """
        return NotImplementedError

    def update_deploy_net(self):
        """ Update deploy net. """
        raise NotImplementedError

    def update_trainval_net(self):
        """ Update trainval net. """
        raise NotImplementedError