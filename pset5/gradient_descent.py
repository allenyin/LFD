"""
Code for gradient descent algorithm, for LFD homework 5
"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
np.set_printoptions(threshold=10)

class gradient_descent(object):
    """
    Fixed learning rate gradient descent algorithm:
        1. Initialize the weights at time step t=0 to w(0)
        2. for t = 0,1,2, ..., do:
        3.      Compute the gradient gt = del(Ein(w(t)))
        4.      Set the direction to move, vt = -gt
        5.      Update the weights: w(t+1) = w(t) + eta*vt
        6.      Iterate to the next step until it is time to stop
        7.  Return the final weights.

    Inputs:
        - dE: function that calculates the gradient at w(t).
        - w0: initial guess. dE should take w0 as an argument.
        - eta: fixed learning rate.
        - t_fun: function to evaluate whether to terminate given w(t).
                 Take current weight and steps as input.
        - stepfun: optional function argument that is called rather than naive step().
                   Should take one argument:
                       - w: current weights
                   Returns:
                       - w: updated weights
    """

    def __init__(self, dE, w0, eta, t_fun, stepfun=None):
        self.dE = dE
        self.w = w0
        self.eta = eta
        self.t_fun = t_fun
        self.stepfun = stepfun
        self.steps = 0

    def step(self):
        if self.stepfun is not None:
            self.w = self.stepfun(self.w)
            return

        else:
            vt = -self.dE(self.w)   # this is likely a vector
            self.w = self.w + vt*self.eta
            return

    def doGD(self):
        while not self.t_fun(self.steps, self.w):
            self.step()
            self.steps += 1
        print "Gradient descent finished!"
        return self.steps

    def __str__(self):
        """
            Print the weights for this instance of gradient_descent
        """
        return str(self.w)

################################################################################
# Problem 4, 5, 6, and 7
################################################################################

def dEuv(weights):
    """
        Implementing the partial derivative of E(u,v)
    """
    u, v = weights*1.0  # make sure it's floating point
    dEdu = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u))
    dEdv = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    return np.array([dEdu, dEdv])

def t_fun_p5(steps, weights):
    """
        Implementing the termination function for problem 5 -
        Terminate when the error is less than 10^-14
    """
    u, v = weights*1.0
    E = (u*np.exp(v) - 2*v*np.exp(-u))**2
    return E < 1e-14

def problem5():
    eta = 0.1
    w0 = np.array([1,1])
    GD1 = gradient_descent(dEuv, w0, eta, t_fun_p5)
    GD1.doGD()
    print "Problem 5 finished with %d steps, weights=%s" % (GD1.steps, GD1)

def coordinate_descent(weights):
    """
        Substitute the naive gradient descent stepping function with
        coordinate descent
    """
    eta = 0.1
    u, v = weights*1.0

    # optimize in u-direction
    dEdu = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u))
    u = u + (-1)*dEdu*eta
   
    # optimize in v-direction
    dEdv = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    v = v + (-1)*dEdv*eta

    return np.array([u,v])

def t_fun_p7(steps, weights):
    # terminate after 15 iterations
    return steps is 15

def problem7():
    eta = 0.1
    w0 = np.array([1,1])
    GD1 = gradient_descent(dEuv, w0, eta, t_fun_p7, stepfun=coordinate_descent)
    GD1.doGD()
    u,v = GD1.w
    E = (u*np.exp(v) - 2*v*np.exp(-u))**2
    print "Problem 7 finished with error %g" % (E)

#################################################################################
# Problem 8,9: Logistic Regression
#################################################################################
def generate_targetFn():
    """ 
        Generate a 1D target Fn by selecting two random points. Then
        finding the line that goes through them.
    """

    points = np.random.uniform(-1, 1, (2,2))
    diffs = np.diff(points, axis=0)
    slope = diffs[0,1]/diffs[0,0]
    intercept = points[0,1]-slope*points[0,0]
     
    return (slope, intercept)

def generate_dataPoints_fromFn(N, fn):
    """
        Generate N test points, with their category (1 or -1) using fn.
        Return in a Nx3 array: (x,y,sign)
    """
    data = np.zeros((N, 3))
    slope = fn[0]
    intercept = fn[1]

    i = 0
    while i < N:
        coords = np.random.uniform(-1, 1, 2)
        if coords[1] > (slope*coords[0] + intercept):
            data[i,:] = [coords[0],coords[1],1]
            i += 1
        elif coords[1] < (slope*coords[0] + intercept):
            data[i,:] = [coords[0],coords[1],-1]
            i += 1
        else:
            continue
    return data

""" The naive gradient descent would be unwieldy to extend to stochastic gradient descent
"""

def CEE_gradient(p, w):
    """
        Given a data-point (x, y, sign), return the cross-entropy error gradient 
        on that point, with current weights w
    """
    p = np.hstack((1,p))    # now it's (1, x, y, sign), 1 = x0 by default
    return -np.array( (p[3]*p[0:3])/(1 + np.exp(p[3]*np.dot(w, p[0:3]))) )

def CEE(p, w):
    """
        Given a set of weights, and a data point (x, y, sign)
        return the cross-entropy error for that point
    """
    p = np.hstack((1,p))    # now it's (1, x, y, sign), 1 = x0 by default
    return np.log(1+np.exp(-p[3]*np.dot(w ,p[0:3])))

def stochastic_GD(training_data, test_data):
    """
        data is Nx3 array, each row is (x,y,sign)
    """
    eta = 0.01
    N, dim = training_data.shape
    idx = np.arange(N)
    
    cur_w = np.zeros((1, dim))
    prev_w = np.ones((1, dim))
    epochs = 0

    while np.linalg.norm(prev_w - cur_w) >= 0.01:
        np.random.shuffle(idx)  # new stochastic gradient descent data order
        prev_w = cur_w
        for i in xrange(N):
            gt = CEE_gradient(training_data[i,:], cur_w)
            vt = -gt
            cur_w = cur_w + eta*vt
        epochs += 1
    
    Eout = np.array([CEE(p, cur_w) for p in test_data])
    Eout = Eout.mean()
    return (Eout, epochs)

def problem8_9():
    runs = 100
    N_train = 100
    N_test = 1000
    epoch_array = np.zeros([runs, 1])
    Eout_array = np.zeros([runs, 1])

    for i in xrange(runs):
        f = generate_targetFn()
        training_data = generate_dataPoints_fromFn(N_train, f)
        test_data = generate_dataPoints_fromFn(N_test, f)
        Eout_array[i], epochs_array[i] = stochastic_GD(training_data, test_data)

    print "Problem 8: Average Eout = %0.3f" % (Eout_array.mean())
    print "Problem 9: Average epochs = %0.3f" % (epoch_array.mean())




    
