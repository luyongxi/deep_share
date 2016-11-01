# Written by Yongxi Lu

""" Truncated SVD used to initialize low-rank factorization """

import numpy as np
from numpy.linalg import svd

def truncated_svd(W, k):
    """ Given input filters, return a set of basis and the linear combination
        required to approximate the original input filters
        Input: 
            W: [dxc] matrix, where c is the input dimension, 
                d is the output dimension
        Output:
            B: [kxc] matrix, where c is the input dimension, 
                k is the maximum rank of output filters
            L: [dxk] matrix, where k is the maximum rank of the
                output filters, d is the output dimension

        Note that k <= min(c,d). It is an error if that is encountered.
    """
    d, c = W.shape
    assert k <= min(c,d), 'k={} is too large for c={}, d={}'.format(k,c,d)
    # S in this case is a vector with len=K=min(c,d), and U is [d x K], V is [K x c]
    u, s, v = svd(W, full_matrices=False)
    # compute square of s -> s_sqrt
    s_sqrt = np.sqrt(s[:k])
    # extract L from u
    B = v[:k, :] * s_sqrt[:, np.newaxis]
    # extract B from v
    L = u[:, :k] * s_sqrt

    return B, L
