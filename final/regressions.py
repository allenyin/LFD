"""
LFD Final pset

Code for questions on Regularized Linear Regression
"""
import numpy as np

# Data set, format is (digit, intensity, symmetry)
trainingData = np.loadtxt("../pset8/features.train")
testData = np.loadtxt("../pset8/features.test")

"""
Implement regularized least-squares linear regression for classification that minimizes the augmented
error with weight decay.

Set lambda = 1, no feature transform, find 1 vs all classifier with the least Ein
"""
def problem7():
    lambda_C = 1
    print "Problem 7"

    # For each 1 vs. all classifier
    for i in np.arange(5,10):
        # pick data
        data = np.array([ [1 if p[0]==i else -1, p[1], p[2]] for p in trainingData])
        
        # get regression
        W_reg = LG_weight_decay(data[:,1:], data[:,0], lambda_C)
                
        # find Ein
        Ein = classification_errors(W_reg, data[:,1:], data[:,0])
        
        print "%d vs. all has Ein=%0.2f" % (i, Ein)
    return

"""
Implement feature transform z=(1, x1, x2, x1*x2, x1^2, x2^2), set lambda=1. Get Eout
"""
def problem8():
    lambda_C = 1
    transformedX = np.array([ [x1, x2, x1*x2, x1**2, x2**2] for (y,x1,x2) in trainingData])
    transformedXtest = np.array([ [x1, x2, x1*x2, x1**2, x2**2] for (y,x1,x2) in testData])
    print "Problem 8"
    for i in np.arange(0,5):
        # transform y to binary data
        y = np.array([ [1 if p[0]==i else -1] for p in trainingData])

        # get regression
        W_reg = LG_weight_decay(transformedX, y, lambda_C)

        # find Eout
        yout = np.array([ [1 if p[0]==i else -1] for p in testData])
        Eout = classification_errors(W_reg, transformedXtest, yout)

        print "%d vs. all has Eout=%0.2f" % (i, Eout)
    return

"""
Compare Ein and Eout for no transform vs transform, for 0 vs all, and 9 vs all classifiers
"""
def problem9():
    lambda_C = 1
    transformedX = np.array([ [x1, x2, x1*x2, x1**2, x2**2] for (y,x1,x2) in trainingData])
    transformedXtest = np.array([ [x1, x2, x1*x2, x1**2, x2**2] for (y,x1,x2) in testData])
    print "Problem 9"
    for i in [0, 5, 9]:
        # transform y to binary data
        y = np.array([ [1 if p[0]==i else -1] for p in trainingData])
        yout = np.array([ [1 if p[0]==i else -1] for p in testData])
        # regressions
        W_reg = LG_weight_decay(trainingData[:,1:], y, lambda_C)
        W_regTrans = LG_weight_decay(transformedX, y, lambda_C)
        
        # Ein and Eout, no transform
        Ein = classification_errors(W_reg, trainingData[:,1:], y)
        Eout = classification_errors(W_reg, testData[:,1:], yout)

        # Ein and Eout, transformed
        Ein_trans = classification_errors(W_regTrans, transformedX, y)
        Eout_trans = classification_errors(W_regTrans, transformedXtest, yout)

        print "\n%d vs. all:" % i
        print "No Transform Ein = %0.3f" % Ein
        print "No Transform Eout = %0.3f" % Eout
        print "Transform Ein = %0.3f" % Ein_trans
        print "Transform Eout = %0.3f" % Eout_trans
    return

"""
Train 1 vs. 5 classifier with transform. Lambda=0.1 and Lambda=0.01. Compare Ein and Eout
If digit is 1, then label as 1
If digit is 5, then label as -1
Else, dont use data.
"""
def problem10():
    lambda_C = [0.01, 1]
    data_train = np.array([p for p in trainingData if p[0]==1 or p[0]==5])
    data_test = np.array([p for p in testData if p[0]==1 or p[0]==5])

    transformedX = np.array([ [x1, x2, x1*x2, x1**2, x2**2] for (y,x1,x2) in data_train])
    transformedXtest = np.array([ [x1, x2, x2*x2, x1**2, x2**2] for (y,x1,x2) in data_test])
    y = np.array([ [1 if p[0]==1 else -1] for p in data_train])
    yout = np.array([ [1 if p[0]==1 else -1] for p in data_test])

    print "Problem 10"
    for l in lambda_C:
        W_reg = LG_weight_decay(transformedX, y, l)
        Ein = classification_errors(W_reg, transformedX, y)
        Eout = classification_errors(W_reg, transformedXtest, yout)

        print "\nlambda = %0.2f" % l
        print "     Ein = %0.3f" % Ein
        print "     Eout = %0.3f" % Eout
    return



def LG_weight_decay(x, y, lambda_C):
    """
        Do one-step linear regression with weight decay regularization applied
        x: Each row is [x1, x2, ... xm]. Total of N rows. Does not contain the constant term yet.
        y: Each row is the value corresponding to y=f(x1,x2,...xm). Total of N rows
        lambda_C: lambda value used in the penalty term of Ein -- Eaug(w)=Ein + lambda_C*wT*w
    """
    N, M = x.shape
    X = np.hstack((np.ones((N,1)), x))  # add the constant terms to X
    W_reg_dagger = np.dot(np.linalg.inv((np.dot(X.T, X) + lambda_C*np.eye(M+1))), X.T)
    W_reg = np.dot(W_reg_dagger, y)
    return W_reg

def classification_errors(weights, x, y):
    """
        Apply linear-regression weights on data x. Check errors against y.

        Given x does not include the constant terms, so augment with 1's.
        
        Returns classification errors out of total data points.
    """
    N = x.shape[0]*1.0
    X = np.hstack((np.ones((N,1)), x))
    Yhat = np.dot(X, weights)
    return (1.0 - np.count_nonzero(np.sign(Yhat)==np.sign(y))/N)

