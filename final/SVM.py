"""
LFD Final pset

Code for questions on SVM and Radio Basis Functions
"""

import numpy as np
from sklearn.svm import SVC
from sklearn import cross_validation

import cvxopt

"""
Problem 12,

Apply hard-margin SVM algorithm with second-order polynomial transformation,
on dual-problem on dataset of 7 points
"""

# (x1,x2,y)
points = np.array([ [1,0,-1],[0,1,-1],[0,-1,-1],[-1,0,1],[0,2,1],[0,-2,1],[-2,0,1]])
def problem12():
    """
    libSVM just for fun
    """
    C = np.inf
    Q = 2
    clf = SVC(C=C, kernel='poly', degree=Q, gamma=1, coef0=1)
    clf.fit(points[:, 0:-1], points[:,2])
    print "Number of support vectors from sklearn is %d" % sum(clf.n_support_)

    # Approach using quadratic programming
    w, b, nSupVec = SVM_kernel(points, poly2_kernel)
    print "Number of support vectors with cvxopt is %d"  % nSupVec
    return (w, b, nSupVec)


def poly2_kernel(v1,v2):
    return np.square((1 + np.dot(v1,v2)))

def SVM_kernel(data, kernel_fn=None):
    """
    Use Quadratic Programming to solve for SVM from given data - (x1,x2,y)
    kernel_fn(v1,v2) gives new output from two input vectors.

    By default, kernel_fn is just the dot product between the two vectors

    Using CVXOPT, from user guide:
        minimize: (1/2)x'Px + q'x
        subj to:  Gx <= h, Ax=b
    Uses the command: cvxopt.solvers.qp(P,q, G, h, A, b, solvers, initvals)

    SVM dual problem can be reduced to (from LFD ch. 8):
        minimize: (1/2)alpha'*QD*alpha - ones(N)'*alpha
        subj to:  Ad*alpha >= zeros(N+2), where N=number of data points.

    Therefore we have the substitution:
        X -> alpha
        P -> QD
        q -> -ones(N)
        G -> -AD
        h -> zeros(N+2)
    """
    tol = 1e-5
    if kernel_fn is None:
        kernel_fn = np.dot

    # calculate signed data-matrix to get QD
    N,M = data.shape
    QD = np.zeros((N,N))
    for i in xrange(N):         # row
        for j in xrange(N):     # col
            QD[i,j] = data[i,-1]*data[j,-1]*kernel_fn(data[i,0:-1], data[j,0:-1])

    # calculate AD
    AD = np.zeros((N+2, N))
    AD[0,:] = data[:,-1].T
    AD[1,:] = -data[:,-1].T
    AD[2:,:] = np.eye(N)

    # get alpha
    cvxopt.solvers.options['feastol'] = 1e-6    # slightly relaxed from default to avoid singular KKT msgs
    cvxopt.solvers.options['abstol'] = 1e-9     # gives good accuracy on final result
    cvxopt.solvers.options['show_progress'] = False

    alpha = cvxopt.solvers.qp(cvxopt.matrix(QD), \
                              cvxopt.matrix(-np.ones(N)),\
                              cvxopt.matrix(-AD),\
                              cvxopt.matrix(np.zeros(N+2)))
    alpha = np.array(alpha['x'])

    # find idx of supp vectors
    supVecIdx = np.where(alpha >= tol)[0]

    # construct the w
    w = np.zeros(M-1)
    for i in xrange(len(supVecIdx)):
        w = w + alpha[supVecIdx[i]] * data[supVecIdx[i],-1] * data[supVecIdx[i], 0:-1]
    w = w.T # col vector now
    
    # construct b
    b = data[supVecIdx[0], -1] - np.dot(w.T, data[supVecIdx[0], 0:-1])

    return (w, b, len(supVecIdx))

