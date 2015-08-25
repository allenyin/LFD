"""
For LFD pset 6, problem 2-6 on Regularization with Weight Decay
"""

import numpy as np
trainingData = np.loadtxt('in.dta')
testData = np.loadtxt('out.dta')

def LG(x, y):
    """
        Do one-step linear regression.
        x: Each row is [x1, x2, ... xn]. Total of N rows.
        y: Each row is the value corresponding to y=f(x1,x2,...xn). Total of N rows.
    """
    N = x.shape[0]
    X = np.hstack((np.ones((N,1)), x))
    Xdagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    weights = np.dot(Xdagger, y)
    return weights

def LG_weight_decay(x, y, lambda_C):
    """
        Do one-step linear regression with weight-decay regularization applied.
        x: Each row is [x1, x2, ... xn]. Total of N rows.
        y: Each row is the value corresponding to y=f(x1,x2,...xn). Total of N rows.
        lambda_C: lambda value used in the penalty term of Ein -- Ein(w)+lambda_C*wT*w
    """
    N,M = x.shape    
    X = np.hstack((np.ones((N,1)), x))
    Xdagger = np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_C*np.eye(M+1)), X.T)   # modified with penalty term
    weights = np.dot(Xdagger, y)
    return weights


def classification_errors(weights, x, y):
    """
        Apply linear-regression weights on data x. Check
        errors against y.

        x does not include the constant terms, so need to augment with 1's.

        Return classification errors out of total data points.
    """
    N = x.shape[0]*1.0
    X = np.hstack((np.ones((N,1)), x))
    yhat = np.dot(X, weights)
    return (1.0 - np.count_nonzero(np.sign(yhat) == np.sign(y))/N)


def problem2(trainingData, testData):
    """
        Do linear transformation:
            phi(x1, x2) = (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|)
        But since LG handles the appending by 1 for us, only return 7 columns of transformed data
    """
    transformedX = np.array([ [x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)] for (x1,x2,y) in trainingData])
    weights = LG(transformedX, trainingData[:,-1])
    Ein = classification_errors(weights, transformedX, trainingData[:,-1])
    
    transformedXtest = np.array([ [x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)] for (x1,x2,y) in testData])
    Eout = classification_errors(weights, transformedXtest, testData[:,-1])

    print "Problem 2: Ein=%0.3f, Eout=%0.3f" % (Ein, Eout)

def problem3_to_5(trainingData, testData, k):
    """
        Same as problem 2, except use linear-regression with weight decay for weights
    """
    lambda_C = 10**k

    # in-sample
    transformedX = np.array([ [x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)] for (x1,x2,y) in trainingData])
    weights = LG_weight_decay(transformedX, trainingData[:,-1], lambda_C)
    Ein = classification_errors(weights, transformedX, trainingData[:,-1])

    # out-sample
    transformedXtest = np.array([ [x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)] for (x1,x2,y) in testData])
    Eout = classification_errors(weights, transformedXtest, testData[:,-1])

    print "k=%d, Ein=%0.3f, Eout=%0.3f" % (k, Ein, Eout)
    return (Ein, Eout)

def problem6():
    """
        Sweep k values to vary amount of weight-decay regularization, keeping k as integer.
        Report Eout

        k from [-5, 5] -> 11 values
    """
    Ein_array = np.zeros((11, 1))
    Eout_array = np.zeros((11, 1))
    for i in xrange(11):
        k = -5+i
        Ein_array[i], Eout_array[i] = problem3_to_5(trainingData, testData, k)
