# Written by Yongxi Lu

""" Simultaneously orthogonal matching pursuit algorihtm 
    We implement the naive form of this algorithm, as documented in:
    Slide 7 of http://users.cms.caltech.edu/~jtropp/slides/Tro05-Simultaneous-Sparsity-Talk.pdf

    As well as the efficient implementation based on Inverse Cholesky Factorization, as documented in
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6692175
    
    It is used to solve the following problem.

    Let S be a signal matrix, where each column is an observation.
    Let D be a dictionary matrix, where each column is a basis.

    We want to solve argmin |Y - TS|, where S is a sparse matrix with |S|_0_\infty <= K, which means
    each column has at most K non-zeros. In other words, no more than K columns of T are used as 
    basis to approximate signal Y.
"""

import numpy as np
import numpy.random as npr
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import lstsq

from utils.config import cfg

def somp_cholesky(Y, T, K, p=2):
    """ Inverse Cholesky implementation of SOMP algorihtm
        Input:
            Y: The signal matrix
            T: the dictionary matrix (each column is an atom)
            K: the cardinality of the basis
            p: use p-norm in finding the columns 
        Output:
            w: the non-zero entries
            S: the linear combination
    """

    # the initial residual is Y itself
    R = Y
    N = T.shape[1]
    F = Y.shape[1]
    I = []
    # gram matrix
    G = np.dot(np.transpose(T), T)
    # norm of colums of T
    norm_T = norm(T, ord=2, axis=0)
    # initialize D and a
    a = np.zeros((0, F))
    D = np.zeros((N, 0))
    
    for k in xrange(K):
        # the remaining set of basis
        I_cmp = [i for i in range(N) if i not in I]
        if k == 0:
            gamma = np.dot(np.transpose(R), T)
        else:
            gamma = gamma - np.dot(Dk[:, np.newaxis], ak[np.newaxis, :])
        obj = norm(gamma[:, I_cmp], ord=p, axis=0)/norm_T[I_cmp]
        # update the set of matching basis
        ik = I_cmp[np.argmax(obj)]
        I.append(ik)
        # iteration updates
        if k >= 1:
            w = D[ik, :]
            if G[ik, ik] - np.dot(w,w) <= 0:
                break
            lam = 1/(np.sqrt(G[ik, ik] - np.dot(w,w)))
            Dk = lam*(G[:,ik]-np.dot(D,w))
        else:
            lam = 1/np.sqrt(G[ik, ik])
            Dk = lam*(G[:, ik])

        ak = lam*gamma[:, ik]
        a = np.vstack((a, ak))
        D = np.hstack((D, Dk[:, np.newaxis]))

    if len(I) < K:
        I_rand = npr.choice(I_cmp, K-len(I), replace=False)
        I.extend(I_rand)

    return I

def somp_naive(Y, T, K, p=2):
    """ Naive implementation of SOMP algorihtm
        Input:
            Y: The signal matrix
            T: the dictionary matrix (each column is an atom)
            K: the cardinality of the basis
            p: use p-norm in finding the columns 
        Output:
            w: the non-zero entries
            S: the linear combination
    """

    # the initial residual is Y itself
    R = Y
    N = T.shape[1]
    I = []

    # norm of columns of T
    norm_T = norm(T, ord=2, axis=0)
    
    for _ in xrange(K):
        # the remaining set of basis
        I_cmp = [i for i in range(N) if i not in I]
        # project columns of R to columns of T, find the column in T that maximizes the 
        # sum of the absolute values of inner product.
        # gamma(i,j) = <y_i, t_j>/||t_j||, where the appropriate inner product is the dot product
        gamma = np.dot(np.transpose(R), T[:, I_cmp])
        obj = norm(gamma, ord=p, axis=0)/norm_T[I_cmp]
        # update the set of matching basis
        ik = I_cmp[np.argmax(obj)]
        I.append(ik)
        # update residual by subtracting the projection
        S = lstsq(T[:,I], Y)[0]
        R = Y - np.dot(T[:, I], S)

    return I

# def somp_pair(S1, S2, D1, D2, T):    
#     """ Solve a pair-wise SOMP problem 
#         Input:
#             S1, S2: Signal matrices
#             D1, D2: Dictionary matrices
#             T: the cardinality of the basis
#         Output:
#             A: the sparse linear combination
#     """

#     # the initial residual is S itself
#     R1 = S1
#     R2 = S2

#     # Check if the number of columns of D1 and D2 are the same
#     assert D1.shape[0]==D2.shape[0], 'Mismatch in number of columns: {} vs {}.'.\
#         format(D1.shape[0], D2.shape[0])
#     d = D1.shape[1]
#     w = []
#     for _ in xrange(T):
#         # the remaining set of basis
#         w_cmp = [i for i in range(d) if i not in w]
#         # project columns of R to columns of D, find the column in D that maximizes the 
#         # sum of the absolute values of inner product.
#         # inner(i,j) = <s_i, d_j>, where the appropriate inner product is the dot product
#         inner1 = np.dot(np.transpose(R1), D1[:, w_cmp])
#         inner2 = np.dot(np.transpose(R2), D2[:, w_cmp])
#         obj = np.abs(inner1).sum(axis=0) + np.abs(inner2).sum(axis=0)
#         # update the set of matching basis
#         w.append(w_cmp[np.argmax(obj)])
#         # update residual by subtracting the projection
#         A1 = lstsq(D1[:,w], S1)[0]
#         A2 = lstsq(D2[:,w], S2)[0]
#         R1 = S - np.dot(D1[:, w], A1) 
#         R2 = S - np.dot(D2[:, w], A2)

#     return w