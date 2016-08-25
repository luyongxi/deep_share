# ------------------------------------------
# Written by Yongxi Lu
# ------------------------------------------


""" Functions for implementation of branching policy """

from caffe.proto import caffe_pb2
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels

def load_branch_params(layers, model, blob):
    """ Return a matrix where each row is the 
        set of linear combination parameters
        associated with a branch.  
        A blob is a tuple (layer_idx, col_idx) 
    """

    # find out all the names, save them to an ordered dictionary.
    names = [] 
    i, j = blob
    for k in xrange(model.num_branch_at(i, j)):
        names.append(model.branch_name_at_i_j_k(i, j, k))

    Xlist = [None for _ in xrange(len(names))]
    # construct a matrix, where each column is a summary of a branch. 
    # print names
    for layer in layers:
        # print 'layer name: {}'.format(layer.name)
        if layer.name in names:
            params = np.reshape(np.array(layer.blobs[0].data), (layer.blobs[0].shape.dim[0], -1))
            params = np.mean(np.abs(params), axis=0)
            Xlist[names.index(layer.name)] = params
    # from Xlist to an array
    X = np.array(Xlist)

    return X

def load_layer_at(layers, net_model, layer, col):
    """
    Load the model parameters at (layer, col)
    """
    # find out all the names, save them to an ordered dictionary.
    names = [] 
    for k in xrange(net_model.num_branch_at(layer, col)):
        names.append(net_model.branch_name_at_i_j_k(layer, col, k))

    Xlist = [None for _ in xrange(len(names))]
    # construct a 3D tensor where the linear combination matrices
    # are concantenated along the axis=2.
    for layer in layers:
        if layer.name in names:
            params = np.reshape(np.array(layer.blobs[0].data), (layer.blobs[0].shape.dim[0], -1))
            Xlist[names.index(layer.name)] = params
    # from Xlist to tensor
    X = np.stack(Xlist, axis=-1)

    return X

def load_caffemodel(caffemodel):
    """ Load the caffemodel"""
    print 'Loading caffemodel: {}'.format(caffemodel)
    with open(caffemodel, 'rb') as f:
        binary_content = f.read()

    protobuf = caffe_pb2.NetParameter()
    protobuf.ParseFromString(binary_content)
    layers = protobuf.layer

    return layers

def dot_abs_kernel(x, y):
    """ Get the dot product of absolute value """
    return np.abs(x).dot(np.abs(y))/(norm(x,2)*norm(y,2))

def matching_dot_abs_kernel(x,y):
    """ Match corresponding elements in the linear combinations """
    return None

def branch_linear_combination(caffemodel, net_model):
    """ Make branching decision based on linear combinations
        Input:
            caffemodel: name to the caffemodel file
            net_model: dynamically generated net model
    """
    # load caffemodel
    layers = load_caffemodel(caffemodel)
    # find out the edges from which we decide on branching.
    edges = net_model.list_edges()
    # construct distance matrix, and perform clustering
    # utilitiy vector
    u = np.zeros((len(edges), ))
    # labels
    labels = [[] for _ in xrange(len(edges))] 
    for i in xrange(len(edges)):
        e = edges[i]
        Xe = load_layer_at(layers, net_model, e[0], e[1])
        # Flatten Xe to Xf, where each row is the set of linear combinations
        # stacked into a vector.
        Xf = Xe.reshape((1,-1,Xe.shape[-1]))
        Xf = Xf[0, :, :]
        Xf = Xf.transpose()
        # flatten the 
        A = pairwise_kernels(Xf, metric=(lambda x, y: 
            dot_abs_kernel(x,y)))
        # initialize spectral clustering object
        spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack',
                                  affinity="precomputed")
        print 'Performing clustering for layer {} branch {}...'.\
            format(*e)
        # store the predicted labels.
        # handle the trivial case
        if A.shape[0] > 2:
            labels[i] = spectral.fit_predict(A)
        else:
            labels[i] = np.array([0,1])

        # compute association terms
        idx0 = np.where(labels[i]==0)[0]
        idx1 = np.where(labels[i]==1)[0]
        ass0 = np.sum(A[idx0,:])
        ass1 = np.sum(A[idx1,:])

        # compute the cut
        L = labels[i]*2.0-1.0
        S = np.dot(L[:, np.newaxis], L[np.newaxis, :])
        Si = 1.0-(S+1.0)/2.0
        cut = np.sum(A*Si)

        # the utility heuristic is the Ncut objective.
        u[i] = (cut/ass0 + cut/ass1)/2.0

        print 'Cluster 0: {}'.format(idx0)
        print 'Cluster 1: {}'.format(idx1)
        print 'cut: {}'.format(cut)
        print 'Ass0: {}'.format(ass0)
        print 'Ass1: {}'.format(ass1)
        print 'Ncut: {}'.format(u[i])

    # return br_idx and br_split associated with the branch with largest cut
    imin = np.argmin(u)
    br_idx = edges[imin]
    br_split = [list(np.where(labels[imin]==0)[0]), list(np.where(labels[imin]==1)[0])]
    
    return br_idx, br_split

# def branch_at_round(caffemodel, net_model, layer, col, k):
#     """
#     Create k branches at (layer, col) on the net_model, by looking at
#     the parameters of the caffemodel
#     """
#     # guard against ill-defined cases
#     assert k>=1, "k must be at least 1"
#     assert k<=net_model.num_branch_at(layer, col), "k cannot exceed number of branches"
#     # handle trivial cases
#     if k==1:
#         print 'Clustering at layer {} column {}, trivial case k=1'.format(layer, col)
#         return (layer, col), [[0]]
#     elif k==net_model.num_branch_at(layer, col):
#         print 'Clustering at layer {} column {}, trivial case k=num_branches={}'.\
#             format(layer, col, k)
#         return (layer, col), [[i] for i in xrange(k)]

#     # load caffemodel
#     layers = load_caffemodel(caffemodel)
#     X = load_layer_at(layers, net_model, layer, col)

#     # Flatten X to Xf, where each row is the set of linear combinations
#     # stacked into a vector.
#     Xf = X.reshape((1,-1,X.shape[-1]))
#     Xf = Xf[0, :, :]
#     Xf = Xf.transpose()

#     # numerator of the distance matrix
#     Xn = pairwise_kernels(Xf, metric=(lambda x, y: 
#         norm(x-y, ord=2)))/Xf.shape[1]
#     # compute affinity matrix using RBF kernel
#     delta = 1.0
#     A = np.exp(- Xn ** 2 / (2. * delta ** 2))
#     spectral = SpectralClustering(n_clusters=k, eigen_solver='arpack',
#                               affinity="precomputed")

#     print 'Performing clustering for layer {} column {}...'.\
#         format(layer, col)
#     labels = spectral.fit_predict(A)

#     for i in xrange(k):
#         idxi = np.where(labels==i)[0]
#         print 'Cluster {}: {}'.format(i, idxi)

#     br_idx = (layer, col)
#     br_split = [list(np.where(labels==i)[0]) for i in xrange(k)]

#     return br_idx, br_split