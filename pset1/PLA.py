"""
Code for Perceptron problem for EdX Learning from Data homework 1,
questions 7-10
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
LFDpath = "/".join(os.path.dirname(__file__).split('/')[0:-1])
if not LFDpath in sys.path:
    sys.path.append(LFDpath)

np.set_printoptions(precision=4)    # set printing precision
np.set_printoptions(threshold=10)    # print only 5 nums per array

class PLA(object):
    """
        PLA uses the perceptron learning algorithm to create a
        classifier to separate data into two groups.

        Atributes:
            w: Weights of the classifier, sign(w[0]+w[1]*x+w[2]*y) determines
               the classification of the dataponit (x,y). By default is 0
            N: Number of training data available.
            data: Training data available to the classifier. Modified from the
                  input data to be: (1, x, y, correct_sign, cur_sign), where
                  cur_sign is the sign given to the point with the current weights
            
        Initialization inputs:
            data: Nx3 array. Each row=data point--(x,y,sign). sign= +1 or -1.
  """

    def __init__(self, data, weights = np.array([0,0,0])):
        """
            Initialize with all weight=0
        """
        
        self.w = weights
        # reformat data to include the constant term for easier computation
        self.N = data.shape[0]
        self.data = np.ones((self.N, 5))
        self.data[:, 1:4] = data
        self.data[:, [4]] = np.zeros((self.N, 1))

    def has_converged(self):
        """
            Check using the training data to see if the classifier
            has converged - i.e. no misclassified point.
        """
        return all(self.data[:,3]==self.data[:,4])

    def step(self):
        """
            If not converged, use the PLA update algorithm to 
            modify the classifier weights according to a randomly
            choosen misclassified point.

            Return true if a step is taken.
        """
        if self.has_converged():
            print "Classifier has already converged given training data!"
            return False
        else:
            missed = self.data[self.data[:,3] != self.data[:,4], :]
            chosen_point = np.random.choice(range(missed.shape[0]))

            # update with w = w+y_n*x_n, where y_n is the correct sign
            self.w = self.w + missed[chosen_point, 0:3]*missed[chosen_point, 3]
            
            # re-classify the data points with the new w
            (self.data[:, [4]],misclassified) = self.classify(self.data[:, 1:4])
            return True

    def classify(self, testdata):
        """
            Using the current weights to classify the given points,
            return the classification in Nx1 array.

            Points is Nx3: (x,y,classification)
        """

        n = testdata.shape[0]
        results = np.zeros((n,1))
        misclassified = 0
        for i in range(n):
            #print i, testdata[i,0:2], self.w[1:3]
            results[i] = np.sign(self.w[0] + np.inner(self.w[1:3], testdata[i, 0:2]))
            if results[i] != testdata[i,2]:
                misclassified += 1
        #print "%d points misclassified." % (misclassified)
        return (results, misclassified)

    def train(self, lim = 10000):
        """
            Train the classifier until convergence or steps exceeding lim.
            Return the number of steps taken
        """
        steps = 0
        while steps < lim:
            if self.has_converged():
                break
            else:
                if self.step():
                    steps += 1
        return steps
        
    def __str__(self):
        """
            Print the weights for this instance of PLA
        """
        s = "w = [%0.4f, %0.4f, %0.4f]" % (self.w[0], self.w[1], self.w[2])
        return s
####################################################################

def generate_dataPoints(N):
    """ 
        Generate N test points, with their category (1 or -1). Return
        in a Nx3 array. First two columns represent coordinates
    """
    data = np.random.uniform(-1, 1, (N, 2))
    signs = np.random.choice([-1, 1], N).reshape(N,1)
    data = np.hstack( (data,signs) )
    return data

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

def plot_fn_and_data(fn, data, fign=1):
    """
        Given the data and function (line), plot them in the [-1,1]x[-1,1] space
    """
    plt.figure(fign)
    plt.subplot(111)
    xx = np.linspace(-1,1,1000)
    yy = xx*fn[0]+fn[1]
    plt.plot(xx, yy)    # plot the line

    pos = data[data[:,2]==1, :]
    neg = data[data[:,2]==-1, :]

    plt.plot(pos[:,0], pos[:,1], '+', markersize=20, markeredgewidth=5, color='red')
    plt.plot(neg[:,0], neg[:,1], '_', markersize=20, markeredgewidth=5, color='blue')

    topline = np.ones(len(yy))
    botline = topline*-1
    plt.fill_between(xx, y1=yy, y2=topline, interpolate=True, color='LightCoral')
    plt.fill_between(xx, y1=yy, y2=botline, interpolate=True, color='LightBlue')

    plt.axis([-1,1,-1,1])
    plt.show()
    
################################################################

"""
Choose random line, generate N points, and run PLA until convergence. Find 
how many steps it takes, and average them for problem 7 and 9.

Also, for each line, generate large dataset of points to test misclassification rate.
Average the number of classification errors for problem 8 and 10.
"""
if __name__ == "__main__":

    step_lim = 10000
    ntest = 5000
    N = 100
    convergence_steps = np.ones((1000,1))*np.NAN
    misclassification_rate = np.ones((1000,1))*np.NAN
    for i in range(1000):
        fn = generate_targetFn()
        training_data = generate_dataPoints_fromFn(N, fn)

        # find number of steps for this trained classifier
        classifier = PLA(training_data)
        convergence_steps[i] = classifier.train(lim = step_lim)

        # make testing data
        testing_data = generate_dataPoints_fromFn(ntest, fn)
        (results, misclassified) = classifier.classify(testing_data)
        misclassification_rate[i] = misclassified/(ntest*1.0)

    print "average convergence steps = %0.3f" % (convergence_steps.mean())
    print "average misclassification = %0.3f" % (misclassification_rate.mean())

########################################################################
#Initial testing code
"""
N = 10
fn = generate_targetFn()
data = generate_dataPoints_fromFn(N, fn)

classifier1 = PLA(data)
steps = 0
lim = 10000

while steps < lim:
    if classifier1.has_converged():
        break
    else:
        if classifier1.step():
            steps += 1

print "Converged after %d steps" % (steps)

fign = 1
classifier1_fn = (-classifier1.w[1]/classifier1.w[2], -classifier1.w[0]/classifier1.w[2])
plot_fn_and_data(classifier1_fn, data, fign)

fign = 2
plot_fn_and_data(fn, data, fign)
"""
