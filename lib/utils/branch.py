# ------------------------------------------
# Written by Yongxi Lu
# ------------------------------------------


""" Functions for implementation of branching policy """

from caffe.proto import caffe_pb2
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigvalsh
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels

from utils.config import cfg


def compute_normalized_cut(A, labels):
    """ compute normalized cut """
    idx0 = np.where(labels==0)[0]
    idx1 = np.where(labels==1)[0]
    ass0 = np.sum(A[idx0,:])
    ass1 = np.sum(A[idx1,:])

    # compute the cut
    signed_labels = labels*2.0-1.0
    expanded_labels = np.dot(signed_labels[:, np.newaxis], signed_labels[np.newaxis, :])
    indicator_labels = 1.0-(expanded_labels+1.0)/2.0
    cut = np.sum(A*indicator_labels)

    # the utility heuristic is the Ncut objective.
    ncut = (cut/ass0 + cut/ass1)/2.0

    print 'Cluster 0: {}'.format(idx0)
    print 'Cluster 1: {}'.format(idx1)
    print 'cut: {}'.format(cut)
    print 'Ass0: {}'.format(ass0)
    print 'Ass1: {}'.format(ass1)
    print 'Ncut: {}'.format(ncut)

    return ncut

def _sym_KL_kernel(x,y):
    """ Compute exp(-KLs(x,y)), where KLs(x,y) is the
        symmetric KL divergence of x and y
    """

    KL1 = x.dot(np.log((x+cfg.EPS)/(y+cfg.EPS)))
    KL2 = y.dot(np.log((y+cfg.EPS)/(x+cfg.EPS)))
    KL = KL1 + KL2
    return np.exp(-KL)

def aff_mean_KL(X):
    """ Take absolute value, compute mean across rows, 
    normalize, then symmetric KL as distance metric. 
    """
    # compute the mean
    Xn = np.abs(X)
    Xf = Xn.mean(axis=0).transpose()
    Xf /= Xf.sum(axis=1)[:, np.newaxis]

    A = pairwise_kernels(Xf, metric=(lambda x, y: 
        _sym_KL_kernel(x,y)))

    return A, Xf

def aff_max_KL(X):
    """ Take absolute value, compute max across rows, 
    normalize, then symmetric KL as distance metric. 
    """
    # compute the max
    Xn = np.abs(X)
    Xf = Xn.max(axis=0).transpose()
    Xf /= Xf.sum(axis=1)[:, np.newaxis]

    A = pairwise_kernels(Xf, metric=(lambda x, y: 
        _sym_KL_kernel(x,y)))
    
    return A, Xf

def aff_dot(X):
    """ Flatten the maatrix and then take dot product. """
    Xf = np.reshape(Xf, (-1, Xf.shape[-1]))
    A = pairwise_kernels(Xf, metric=(lambda x, y: 
        x.dot(y)/(norm(x,2)*norm(y,2))))

    return A, Xf