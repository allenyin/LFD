"""
Comparing PLA vs. SVM performance, question 8-10 for LFD pset7

For question 9, in order to get the answer that SVM outperforms PLA 70% of the time,
we need to increase the number of test points to at least 10,000, which yielded 
SVM outperforming around 65% of the time, with on average 3 support vectors. Took 17min to run though.

Also, need to change the alpha tolerance to 1e-5 used to detect non-zero alphas.
"""

import sys
import os
LFDpath = "/".join(os.path.dirname(__file__).split('/')[0:-1])
if not LFDpath in sys.path:
    sys.path.append(LFDpath)
from pset1.PLA import *
import cvxopt

# PLA include the code needed to generate 2D test data, target function, and running PLA

def runComparison(N):
    """
    For training-sample size N, run experiment 1000 times, where in each iteration:
        1. Create target function and training data.
        2. Train PLA and SVM
        3. Create ntest testing points, and compare the misclassification performance
           as percentage of disagreements.
    """
    ntest = 1000    # number of out-of-sample dataset points
    PLA_miss = np.ones(1000)*np.NAN
    SVM_miss = np.ones(1000)*np.NAN
    SVM_nsupp = np.ones(1000)*np.NAN
    for i in xrange(1000):
        f = generate_targetFn()
        trainingSet = generate_dataPoints_fromFn(N, f)
        while all_same_sign(trainingSet):
            trainingSet = generate_dataPoints_fromFn(N, f)
        testingSet = generate_dataPoints_fromFn(ntest, f)
        while all_same_sign(testingSet):
            testingSet = generate_dataPoints_fromFn(ntest, f)

        # train PLA
        step_lim = 10000
        classifier = PLA(trainingSet)
        classifier.train(lim = step_lim)
        # test PLA
        (results, misclassified) = classifier.classify(testingSet)
        PLA_miss[i] = misclassified/(ntest*1.0)

        # train SVM
        (w, b, SVM_nsupp[i]) = SVM(trainingSet)
        # test SVM
        (results, misclassified) = apply_SVM(w, b, testingSet)
        SVM_miss[i] = misclassified/(ntest*1.0)

    # find how often SVM is better than PLA in out-of-sample performance
    print "For N=%d, SVM is better than PLA %0.3f of the times" % (N, sum(np.less(SVM_miss, PLA_miss))/1000.0)
    print "For N=%d, average number of support vectors is %0.3f" % (N, np.mean(SVM_nsupp))
        
def all_same_sign(data):
    """
    Given data with row in the form of (x1,x2,y),
    check if all the y's have the same sign
    """
    s = np.sign(data[0,-1])
    for i in xrange(1, data.shape[0]):
        if np.sign(data[i,-1]) != s:
            return False
    return True


def SVM(data):
    """
    Use Quadratic Programming to solve for SVM from given data - (x1,x2,y)

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

    # calculate signed data-matrix to get QD
    N,M = data.shape
    Xs = np.zeros((N, M-1))
    for i  in xrange(N):
        Xs[i,:] = data[i,-1] * data[i,0:-1]
    QD = np.dot(Xs, Xs.T)

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

def apply_SVM(w, b, x):
    """
    Apply SVM defined by (w,b) on data point given by x = (x1,x2,.., xn, y)
    and return the classification results, and number of misclassification
    """
    results = np.zeros(x.shape[0])
    misclassified = 0
    for i in xrange(x.shape[0]):
        results[i] = np.sign( np.dot(w, x[i,0:-1]) + b)
        if results[i] != x[i,-1]:
            misclassified += 1
    return (results, misclassified)



