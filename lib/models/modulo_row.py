# Modified by Yongxi Lu
# Based on source code provided by Abhishek Kumar

import cvxpy as cvx
from cvxpy import Variable, Minimize, Problem, sum_entries, CVXOPT, SCS
import numpy as np
import sys

def l1dist_matrices_rowperm(A,B,tol):
    """
     min_P ||A - PB||_1 s.t. P is a permutation matrix
     LP formulation:
     min 1'Z1 s.t. A-PB <= Z, A-PB>=-Z, P'1 = 1, 1'P = 1', P>=0
    """

    m, n = A.shape

    Z = Variable(m,n)
    P = Variable(m,m)
    ones_m = np.ones((m,1))
    
    #objective = Minimize(sum_entries(Z))
    #constraints = [A-P*B <= Z,
    #               P*B-A <= Z,
    #               P >= 0,
    #               P*ones_m == ones_m,
    #               P.T * ones_m == ones_m]
    
    objective = Minimize(cvx.norm(A-P*B,1))
    constraints = [P >= 0, P*ones_m == ones_m, P.T * ones_m == ones_m]
    
    prob = Problem(objective, constraints)
    prob.solve(verbose=True, solver=SCS, eps=tol)
    #prob.solve(verbose=True, solver=CVXOPT, abstol=1e-6)

    P = P.value
    dist = A - P.dot(B)
    return dist.sum(), P
        
if __name__=='__main__':
    if len(sys.argv)<3:
        print 'Usage:', sys.argv[0], ' rowsize colsize'
        print 'Solves a problem of specified size with random matrices'
        sys.exit()
    
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    A = np.random.rand(m,n)
    p = np.random.permutation(m)
    P = np.zeros((m,m)) #permutation matrix
    for i in range(m):
        P[i,p[i]] = 1
    B = P.dot(A)

    d, Q = l1dist_matrices_rowperm(A,B,1e-3)
    d_naive = np.abs(A - B)
    print 'L1 distance (naive):', d_naive.sum()
    print 'L1 distance (modulo row permutations):', d
    PQ = np.abs(P-Q)
    print PQ.sum()

