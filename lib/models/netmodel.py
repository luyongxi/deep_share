# Written by Yongxi Lu

""" A class that represents the network models we are interested in """

import os
import os.path as osp
import caffe

def reduce_param_mapping(mappings):
    """ Reduce a sequence of param_mapping to 
        a single mapping procedure
        Input: 
            mappings: a list of param_mapping. 
        Output:
            param_mapping: a param mapping that is
                equivalent of applying the mappings
                on the "mappings" sequentially.
    """
    param_mapping = {}
    for new_mapping in mappings:
        for k, v in new_mapping.iteritems():
            if param_mapping.has_key(v):
                param_mapping[k] = param_mapping[v]
            else:
                param_mapping[k] = v
    return param_mapping

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

    def __init__(self, model_name, path, io, num_layers):
        """ Initialize a model for neural network
            Inputs:
                model_name: name of the model
                path: root path to save the prototxt files
                io: specifies the input/output, must has the following members
                    num_tasks: number of tasks (determine the number of branches 
                           of the last intermediate layers)
                    data_name: name of the data blob
                    add_input(net, deploy): add input layers 
                    add_output(net, bottom_dict, deploy): add output layers
                    col_name_at_j(j): column name at j
                    branch_name_at_j_k(j, k): branch name
                    num_layers: number of intermediate layers 
                            (not counting inputs and task specific 
                            output layers)
        """
        self._model_name = model_name
        self._path = path
        self._io = io
        self._num_layers = num_layers
        self._init_graph(num_layers, self.num_tasks)

    def _init_graph(self, num_layers, num_tasks):
        """ Initilize the connection graph of blobs """
        self._net_graph = [[] for _ in xrange(num_layers+1)]
        for i in xrange(num_layers+1):
            j = num_layers-i    # start from the top layer
            if j==num_layers:
                self._net_graph[j].append(NetBlob(top_idx=range(num_tasks), 
                    tasks=[[k] for k in range(num_tasks)]))
            else:
                top_idx = [0]
                # flatten the tasks list at the top blob
                tasks = [[t for b in self._net_graph[j+1] for ts in b.tasks for t in ts]]
                self._net_graph[j].append(NetBlob(top_idx=top_idx, tasks=tasks))

    def list_tasks(self):
        """ Return a list of idx of the tasks. The idx is in the same order of the
            branches.
        """
        task_list = []
        i = self.num_layers
        for j in xrange(self.num_cols_at(i)):
            for k in xrange(self.num_branch_at(i,j)):
                task_list.extend(self.tasks_at(i, j, k))

        return task_list

    def insert_branch(self, idx, split):
        """ Create a new branch. 
            Inputs:
                idx: a tuple (layer_idx, col_idx), to insert the branch.
                split: a list with two sub-lists, each are indexes into a set of tops (branches)
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
        # create new blobs (columns) at the current layer
        # save original blob
        orig_blob = self._net_graph[idx[0]][idx[1]]
        top_idx = orig_blob.top_idx 
        tasks = orig_blob.tasks
        # left column
        self._net_graph[idx[0]][idx[1]] = \
            NetBlob(top_idx=[top_idx[i] for i in left], tasks=[tasks[i] for i in left])
        # right column
        self._net_graph[idx[0]].append(
            NetBlob(top_idx=[top_idx[i] for i in right], tasks=[tasks[i] for i in right]))
        right_idx = len(self._net_graph[idx[0]])-1
        # add a new branch at the bottom layer
        b_blobs = bottoms[bottom_idx]
        b_blobs.set_tasks(branch_idx=[branch_idx], tasks=[[t for i in left for t in [tasks[i]]]])
        b_blobs.add_top(top_idx=[right_idx], tasks=[[t for i in right for t in [tasks[i]]]])
        # log changes
        changes = {}
        # layer i
        # blobs
        changes[(idx[0], right_idx)] = (idx[0],idx[1])
        # branches
        for k1 in xrange(len(left)):
            changes[(idx[0], idx[1], k1)] = (idx[0], idx[1], tops[left[k1]])
        for k2 in xrange(len(right)):
            changes[idx[0], right_idx, k2] = (idx[0], idx[1], tops[right[k2]])
        # layer i-1
        # branches
        changes[(idx[0]-1, bottom_idx, b_blobs.num_tops()-1)] = (idx[0]-1, bottom_idx, branch_idx)

        return self.to_param_mapping(changes)

    def to_param_mapping(self, changes):
        """ Convert change lists from the insert branch function to 
            param_mapping matching old models to the new model
        """
        return NotImplementedError

    def list_edges(self):
        """ Return a list of tuples, each with a layer idx and
            a column idx. The list indexes the blobs considered
            at the edge (candidate for branching).
        """
        edges = []
        for layer_idx in xrange(1, len(self._net_graph)):
            for col_idx in xrange(len(self._net_graph[layer_idx])):
                # if layer idx is 0, then the layer below is the shared input
                if self._net_graph[layer_idx][col_idx].is_edge():
                    edges += [(layer_idx, col_idx)]

        return edges

    def tasks_at(self, i, j, k=None):
        """ Return the indexing into tasks at layer i, column j, [branch k] """
        if k is None:
            return self._net_graph[i][j].tasks
        else:
            return self._net_graph[i][j].tasks[k]

    def num_tasks_at(self, i, j, k=None):
        """ Return the number of tasks associated with layer i, column j, [branch k] """
        tasks = self.tasks_at(i,j,k)
        return len([t for ts in tasks for t in ts])

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
    def path(self):
        """ Will save as path/train_val.prototxt, 
            or path/test.prototxt """
        return self._path

    def set_path(self, path):
        self._path = path

    def set_name(self, name):
        self._model_name = name

    @property
    def model_name(self):
        return self._model_name

    @property
    def io(self):
        return self._io

    def to_proto(self, deploy=False):
        """ Need different inputs for deploy or not """
        # check if the folder exists, if not create a new folder. 
        folder = self.path
        if not osp.exists(folder):
            os.makedirs(folder)

        if deploy==True:
            fn = osp.join(folder, 'test.prototxt')
        else:
            fn = osp.join(folder, 'train_val.prototxt')

        name_str = 'name: {}\n'.format('"'+self.model_name+'"')
        with open(fn, 'w') as f:
            f.write(name_str+self.proto_str(deploy=deploy))

    def names_at_i_j(self, i, j):
        """ Return the name of the parameters at layer i, column j.
            This function depends on the exact network architecture
            It could return a string or a dict.
        """
        return NotImplementedError

    def col_name_at_i_j(self, i, j):
        """ provide the name of column """
        return NotImplementedError

    def branch_name_at_i_j_k(self, i, j, k):
        """ provide the name of a branch """
        return NotImplementedError

    def proto_str(self, deploy):
        """ Return the prototxt file in string """
        raise NotImplementedError