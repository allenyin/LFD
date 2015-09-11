"""
For LFD pset 7, problem 1-6 on validation
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

def trans_k3(data):
    return np.array([ [x1,x2] for (x1,x2,y) in data])

def trans_k4(data):
    return np.array([ [x1, x2, x1*x1] for (x1,x2,y) in data])

def trans_k5(data):
    return np.array([ [x1, x2, x1*x1, x2*x2, x1*x2] for (x1,x2,y) in data])

def trans_k6(data):
    return np.array([ [x1, x2, x1*x1, x2*x2, x1*x2, abs(x1-x2)] for (x1,x2,y) in data])

def trans_k7(data):
    return np.array([ [x1, x2, x1*x1, x2*x2, x1*x2, abs(x1-x2), abs(x1+x2)] for (x1,x2,y) in data])


def problem1_2():
    """
        Do linear transformation from (x1, x2) to:
            k = 3: (1, x1, x2)
            k = 4: (1, x1, x2, x1^2)
            k = 5: (1, x1, x2, x1^2, x2^2, x1*x2)
            k = 6: (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|)
            k = 7: (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|)
        Train on trainigSet, and compare on validationSet, and testData
    """
    trainingSet = trainingData[0:25, :]
    validationSet = trainingData[25:, :]

    # k=3
    transformedX = trans_k3(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k3(validationSet) 
    Eval_k3 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k3(testData) 
    Eout_k3 = classification_errors(weights, transformedXtest, testData[:,-1])
    
    # k = 4
    transformedX = trans_k4(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k4(validationSet) 
    Eval_k4 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k4(testData)
    Eout_k4 = classification_errors(weights, transformedXtest, testData[:,-1])

    # k = 5
    transformedX = trans_k5(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k5(validationSet) 
    Eval_k5 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k5(testData)
    Eout_k5 = classification_errors(weights, transformedXtest, testData[:,-1])

    # k = 6
    transformedX = trans_k6(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k6(validationSet) 
    Eval_k6 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k6(testData)
    Eout_k6 = classification_errors(weights, transformedXtest, testData[:,-1])

    # k = 7
    transformedX = trans_k7(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k7(validationSet) 
    Eval_k7 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k7(testData)
    Eout_k7 = classification_errors(weights, transformedXtest, testData[:,-1])

    print "Problem 1 - Validation Error:\n \
           k=3: %0.3f\n \
           k=4: %0.3f\n \
           k=5: %0.3f\n \
           k=6: %0.3f\n \
           k=7: %0.3f\n" % (Eval_k3, Eval_k4, Eval_k5, Eval_k6, Eval_k7)

    print "Problem 2 - Eout:\n \
           k=3: %0.3f\n \
           k=4: %0.3f\n \
           k=5: %0.3f\n \
           k=6: %0.3f\n \
           k=7: %0.3f\n" % (Eout_k3, Eout_k4, Eout_k5, Eout_k6, Eout_k7)

def problem3_4():
    """
        Do linear transformation from (x1, x2) to:
            k = 3: (1, x1, x2)
            k = 4: (1, x1, x2, x1^2)
            k = 5: (1, x1, x2, x1^2, x2^2, x1*x2)
            k = 6: (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|)
            k = 7: (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|)
        Train on trainigSet, and compare on validationSet, and testData
    """
    validationSet = trainingData[0:25, :]
    trainingSet = trainingData[25:, :]

    # k=3
    transformedX = trans_k3(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k3(validationSet) 
    Eval_k3 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k3(testData) 
    Eout_k3 = classification_errors(weights, transformedXtest, testData[:,-1])
    
    # k = 4
    transformedX = trans_k4(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k4(validationSet) 
    Eval_k4 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k4(testData)
    Eout_k4 = classification_errors(weights, transformedXtest, testData[:,-1])

    # k = 5
    transformedX = trans_k5(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k5(validationSet) 
    Eval_k5 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k5(testData)
    Eout_k5 = classification_errors(weights, transformedXtest, testData[:,-1])

    # k = 6
    transformedX = trans_k6(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k6(validationSet) 
    Eval_k6 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k6(testData)
    Eout_k6 = classification_errors(weights, transformedXtest, testData[:,-1])

    # k = 7
    transformedX = trans_k7(trainingSet) 
    weights = LG(transformedX, trainingSet[:,-1]);
    transformedXval = trans_k7(validationSet) 
    Eval_k7 = classification_errors(weights, transformedXval, validationSet[:,-1])
    transformedXtest = trans_k7(testData)
    Eout_k7 = classification_errors(weights, transformedXtest, testData[:,-1])

    print "Problem 3 - Validation Error:\n \
           k=3: %0.3f\n \
           k=4: %0.3f\n \
           k=5: %0.3f\n \
           k=6: %0.3f\n \
           k=7: %0.3f\n" % (Eval_k3, Eval_k4, Eval_k5, Eval_k6, Eval_k7)

    print "Problem 4 - Eout:\n \
           k=3: %0.3f\n \
           k=4: %0.3f\n \
           k=5: %0.3f\n \
           k=6: %0.3f\n \
           k=7: %0.3f\n" % (Eout_k3, Eout_k4, Eout_k5, Eout_k6, Eout_k7)

def const_fit(points):
    p1,p2 = points
    return np.mean((p1[1], p2[1]))

def linear_fit(points):
    p1,p2 = points
    a = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1]-p2[0]*a
    return np.array([a,b])

def leave_one_out_idx(N):
    idx = np.arange(N)
    idx = idx[1:]-(idx[:,None]>=idx[1:])
    return idx


def problem7():
    """
        Do leave-one-out cross-validation for 3 points on two models:
            1. h(x) = b, constant function.
            2. h(x) = ax+b, linear
        Find for what value of rho will the xvalidation error be tied, where
        the three data points are:
            (-1,0), (rho, 1), (1,0)
    """
    rhos = np.array([np.sqrt(np.sqrt(3)+4), \
                     np.sqrt(np.sqrt(3)-1), \
                     np.sqrt(9 + 4*np.sqrt(6)), \
                     np.sqrt(9 - np.sqrt(6))])
    # leave-one-out idx for points
    idx = leave_one_out_idx(3)
    for i in xrange(len(rhos)):
        points = np.array(( [-1,0],\
                            [rhos[i], 1],\
                            [1,0] ))
        h1_error = np.zeros((3,1))
        h2_error = np.zeros((3,1))
        for j in range(3):
            h1 = const_fit(points[idx[j,:],:])
            h2 = linear_fit(points[idx[j,:],:])
            h1_error[j] = (h1 - points[j,1])**2
            h2_error[j] = (h2[0]*points[j,0]+h2[1] - points[j,1])**2

        if h1_error.mean() == h2_error.mean():
            print "Problem 7: rho=%0.3f gives same cross-validation error for both models" % rhos[i]
            return

    print "Problem 7: None of the given rhos result in same cross-validation error"



