"""
Code for the bias-and-variance questions for Homework 4

Question 4-7
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6)
np.set_printoptions(threshold=10)

def fit_ax(p1, p2):
    """
        Given two points, fit h(x)=ax, minimizing mean-squared error.
        
        The fitting/learning algorithm follows that of the linear regression, where the learned w is:
            
            w = X_dagger*y
        where w is replaced with the vector of a, an Nx1 vector

        Return the value of a
    """
    X = np.array([p1[0], p2[0]])
    Y = np.array([p1[1], p2[1]])
    Xsq = np.dot(X.T, X)
    if len(Xsq.shape) < 2:
        Xdagger = X.T/Xsq*1.0
    else:
        Xdagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    a = np.dot(Xdagger, Y)
    return a

def fit_ax_b(p1, p2):
    """
        Given two points, fit h(x)=ax+b, minimizing mean-squared error.

        return the value of (a,b)
    """
    X = np.array(([1, p1[0]], [1, p2[0]]))
    Y = np.array([p1[1], p2[1]])
    Xdagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    a = np.dot(Xdagger, Y)
    return a

def fit_b(p1, p2):
    return (p1[1]+p2[1])/2.0


def pick_points_on_sin(N):
    """
    Pick N random points on f(x) = sin(pi*x), x uniform on [-1,1]
    """
    X = np.random.uniform(-1, 1, N)
    Y = np.array([np.sin(np.pi*x) for x in X])
    return np.vstack((X, Y)).T

def problem4_5_6():
    N = 10000
    data = pick_points_on_sin(2*N)
    a_array = np.array([fit_ax(data[i,:], data[i+1,:]) for i in np.arange(N, step=2)])
    gbar = np.mean(a_array)
    print "Problem 4: gbar(x) for h(x)=ax has a=%0.2f" % (gbar)

    """
    Calculate bias: Ex[(gbar(x)-f(x))^2]
    """
    gbar_of_x = np.array([gbar*data[i,0] for i in np.arange(2*N)])
    bias = np.mean((gbar_of_x - data[:,1])**2)
    print "Problem 5: bias for h(x)=ax is %0.2f" % bias

    """
    Calculate variance: Ex[ E_D[ (gD(x)-gbar(x))^2]]
    """
    diffs = np.zeros((2*N, N))
    for i in xrange(len(a_array)):
        diffs[:,i] = (a_array[i]*data[:,0] - gbar_of_x)**2
    var = diffs.mean()
    print "Problem 6: var for h(x)=ax is %0.2f" % var

def problem7():
    """
    Calculate the expected Eout for different hypothesis
    """

    N = 10000
    data = pick_points_on_sin(2*N)
    data_xsq = np.array([(p[0]**2, p[1]) for p in data])

    # fit the hypothesis
    h1_array = np.array([fit_b(data[i,:], data[i+1,:]) for i in np.arange(N, step=2)])          # b
    h2_array = np.array([fit_ax(data[i,:], data[i+1,:]) for i in np.arange(N, step=2)])         # ax
    h3_array = np.array([fit_ax_b(data[i,:], data[i+1,:]) for i in np.arange(N, step=2)])       # ax+b
    
    h4_array = np.array([fit_ax(data_xsq[i,:], data_xsq[i+1,:]) for i in np.arange(N, step=2)])    # ax^2
    h5_array = np.array([fit_ax_b(data_xsq[i,:], data_xsq[i+1,:]) for i in np.arange(N, step=2)]) # ax^2+b

    # get the out-sample-results
    diffs1 = np.zeros((2*N, N))
    diffs2 = np.zeros((2*N, N))
    diffs3 = np.zeros((2*N, N))
    diffs4 = np.zeros((2*N, N))
    diffs5 = np.zeros((2*N, N))

    for i in xrange(len(h1_array)):
        diffs1[:,i] = (h1_array[i]-data[:,1])**2
        diffs2[:,i] = ((h2_array[i]*data[:,0]) - data[:,1])**2
        diffs3[:,i] = ((h3_array[i,0] + h3_array[i,1]*data[:,0]) - data[:,1])**2
        diffs4[:,i] = (h4_array[i]*(data[:,0]**2) - data[:,1])**2
        diffs5[:,i] = ((h5_array[i,0] + h5_array[i,1]*(data[:,0]**2)) - data[:,1])**2
    
    print "Problem 7, errors for:\n\
                h(x)=b: %0.3f\n\
                h(x)=ax: %0.3f\n\
                h(x)=ax+b: %0.3f\n\
                h(x)=ax^2: %0.3f\n\
                h(x)=ax^2+b: %0.3f" % (diffs1.mean(), diffs2.mean(), diffs3.mean(), diffs4.mean(), diffs5.mean())

