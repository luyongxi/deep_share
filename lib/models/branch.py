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

    Xlist = [[] for _ in xrange(len(names))]
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

def load_caffemodel(caffemodel):
    """ Load the caffemodel"""
    print 'Loading caffemodel: {}'.format(caffemodel)
    with open(caffemodel, 'rb') as f:
        binary_content = f.read()

    protobuf = caffe_pb2.NetParameter()
    protobuf.ParseFromString(binary_content)
    layers = protobuf.layer

    return layers

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
        Xe = load_branch_params(layers, net_model, e)
        # numerator of the distance matrix
        # Xn = pairwise_kernels(Xe, metric=(lambda x, y: 
        #     np.abs(np.abs(x)-np.abs(y)).mean()))
        Xn = pairwise_kernels(Xe, metric=(lambda x, y: 
            norm(x-y, ord=1)))
        # Xd = np.mean(Xe)
        # compute affinity matrix using RBF kernel
        delta = 1.0
        # A = np.exp(- (Xn/Xd) ** 2 / (2. * delta ** 2))
        A = np.exp(- Xn ** 2 / (2. * delta ** 2))
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

        # use 1-norm of group-wise mean difference
        idx0=np.where(labels[i]==0)[0]
        idx1=np.where(labels[i]==1)[0]
        xe0 = Xe[idx0, :].mean(axis=0)
        xe1 = Xe[idx1, :].mean(axis=0)
        # take one norm of the difference of mean
        u[i] = norm(xe0-xe1, ord=1)/Xe.shape[1]

        # num_pos = np.sum(labels[i]==0)
        # num_neg = np.sum(labels[i]==1)

        # L = labels[i] * 2.0 - 1.0
        # S = np.dot(L[:, np.newaxis], L[np.newaxis, :])
        # Si = 1.0-(S+1.0)/2.0
        # # based on the clustering result, fill in u
        # u[i] = np.sum(Xn/Xd*Si)/(num_pos*num_neg)

        # print 'heuristic of layer {}, branch {}'.format(*edges[i])
        # print 'L: {}'.format(L)
        # print 'S: {}'.format(S)
        # print 'Si: {}'.format(Si)
        # print 'Xn/Xd: {}'.format(Xn/Xd)
        # print 'num_pos: {}'.format(num_pos)
        # print 'num_neg: {}'.format(num_neg)
        # print 'u[i]: {}'.format(u)

    # return br_idx and br_split associated with the branch with largest cut
    imax = np.argmax(u)
    br_idx = edges[imax]
    br_split = [list(np.where(labels[imax]==0)[0]), list(np.where(labels[imax]==1)[0])]
    
    return br_idx, br_split
