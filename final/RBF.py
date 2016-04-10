"""
LFP Final pset

Code for questions on Radio Basif Functions, question 13-18

SVM - use libsvm first, then use custom SVM code.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import cross_validation
from numpy.linalg import inv

import matplotlib.pyplot as plt


def generateData(n=100):
    """
    Input space is [-1,1]x[-1,1] with uniform probability.
    Target is f(x) = sign(x2 - x1 + 0.25*sin(pi*x1))

    Each data set has 100 training points.

    Data format should be [x1,x2, f(x1,x2)]
    """
    data = np.random.uniform(-1, 1, (n,2))
    signs = np.array( [np.sign(p[1]-p[0]+0.25*np.sin(np.pi*p[0])) for p in data]).reshape(n,1)
    data = np.hstack( (data,signs) )
    return data

def plot_fn_and_data(data, fign=1):
    """
        Plot the [x1,x2,f(x1,x2)] data along with f(x) in the [-1,1]x[-1,1] space
    """
    plt.figure(fign)
    xx = np.linspace(-1,1,1000)
    yy = xx - 0.25*np.sin(np.pi*xx)
    plt.plot(xx, yy, 'k')

    pos = data[data[:,2]==1, :]
    neg = data[data[:,2]==-1, :]
    plt.plot(pos[:,0], pos[:,1], '+', markersize=5, color='red')
    plt.plot(neg[:,0], neg[:,1], 'o', markersize=5, color='blue')

    topline = np.ones(len(yy))
    botline = topline*-1
    plt.fill_between(xx, y1=yy, y2=topline, interpolate=True, color='LightCoral')
    plt.fill_between(xx, y1=yy, y2=botline, interpolate=True, color='LightBlue')

    plt.axis([-1,1,-1,1])
    plt.show(block=False)

def lloyd_clustering(data, K, plot=True):
    """
    Lloyd's algorithm for K-means clustering given a data set, and the number of 
    centers K.

    Follows LFD e-chapter 6.3, page 32 algorithm. Greedily initialize the k-centers.
    """
    # first augment data matrix with the cluster numbering -- 0:K-1
    N = data.shape[0]
    data = np.hstack( (data, -1*np.ones((N,1))) )
    # data is now [x1,x2,f(x1,x2),cluster]
    centers = greedy_choose_center(data, K)
    if plot:
        plot_centers_in_data(data, centers, fign=1)

    Ein = np.infty
    Ej = np.zeros( (K,1) )
    done = False
    
    iters = 0
    while iters < 100:
        iters = iters + 1
        # label each data point with that of the closest center
        for i in np.arange(1,N):
            p = data[i,:]
            # calculate dist to each cluster center
            dist = np.array([ (p[0]-q[0])**2 + (p[1]-q[1])**2 for q in centers])
            # update point's label
            data[i, 3] = np.argmin(dist)

        # update centers to centroid of each cluster
        for j in np.arange(1,K):
            Sj = np.array( [p for p in data if p[3]==j] )
            centers[j] = np.sum(Sj[:,0:2],0)/Sj.shape[0]
            # calculate new Ej
            Ej[j] = sum([(p[0]-centers[j][0])**2 + (p[1]-centers[j][1])**2 for p in Sj])
       
        Enew = sum(Ej)[0]
        #print "iter %d, Ein=%0.5f" % (iters, Enew)
        if Ein is np.infty:
            Ein = Enew
        elif sum(Ej)-Ein < 1e-2:
            Ein = Enew
        else:
            break
    if plot:
        plot_centers_in_data(data, centers, fign=2)
        print 'Lloyd clustering finished in %d iterations, Ein=%0.5f' % (iters, Ein)
    
    return centers

def plot_centers_in_data(data, centers, fign=1):
    plt.figure(fign)
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], marker='x', color='red')
    plt.axis([-1,1,-1,1])
    plt.show(block=False)

def greedy_choose_center(data, K):
    """
    data has rows: [x1,x2, f(x1,x2), cluster number]
    Re-use the last col to hold sum of distance of the current point from the
    current centers
    """
    N = data.shape[0]
    centers = np.zeros( (K,2) )
    # randomly choose first point
    center_idx = np.random.choice(N,1)
    centers[0,:] = data[center_idx, 0:2]

    # take away this point from the dataset
    data = np.delete(data, center_idx, 0)
    N = data.shape[0]

    for i in range(1,K):
        # choose the point furthest from all chosen points
        for j in range(N):
            px,py = data[j, 0:2]
            dist = np.array([np.sqrt((px-c[0])**2 + (py-c[1])**2) for c in centers[0:i,:]])
            data[j,3] = sum(dist)
        center_idx = np.argmax(data[:,3])
        centers[i,:] = data[center_idx, 0:2]
        data = np.delete(data, center_idx, 0)
        N = N-1
    return centers


def libSVM_RBF(data, gamma):
    """
    Use RBF kernel with hard-margin SVM to classify data
    Data has format [x1,x2, f(x1,x2)]

    libSVM approach first
    """
    C = 1e10    # big C for hard-margin
    clf = SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(data[:, 0:2], data[:,2])
    #print "Number of support vectors from sklearn is %d" % sum(clf.n_support_)
    return clf

def problem13(M):
    """
    gamma=1.5, run hard-margin SVM-RBF on 100-points training set for M times

    Count how many times we get non-separable data
    """
    gamma = 1.5
    nonseparable = 0.0
    for i in np.arange(1,M):
        data = generateData(n=100)
        clf = libSVM_RBF(data, gamma)
        if not all(clf.predict(data[:, 0:2]) == data[:,2]):
            nonseparable = nonseparable + 1
    print "Problem 13: %0.2f%% of time data set inseparable" % (nonseparable / M * 100)
    return (nonseparable/M)

def RBF_network(data, gamma, K):
    """
    Train regular form RBF model: k-centers with Lloyd and linear regression pseudo inverse
    Data has format [x1,x2, f(x1,x2)]

    gamma = RBF kernel param
    K = number of centers
    """

    N = data.shape[0]
    centers = None
    while centers is None:
        try:
            # centers[i] has format[x1,x2]
            centers = lloyd_clustering(data, K, plot=False) 
        except IndexError:
            pass

    # make the feature matrix Z
    Z = np.ones( (N, K+1) )
    Z[:, 1:] = np.array([ [RBF_kernel([p[0],p[1]], mu, gamma) for mu in centers] for p in data])
    # fit the linear model by using pseudo-inverse
    weights = np.dot(np.dot(inv(np.dot(Z.T, Z)), Z.T), data[:,2])
    return (weights,centers)

def RBF_kernel(x, mu, gamma):
    """
    Apply the RBF kernel to point x=[x1,x2], given
    a center mu=[x1',x2']
    """
    d = np.sqrt( (x[0]-mu[0])**2 + (x[1]-mu[1])**2 )
    return np.exp(-(d * gamma))

def apply_RBF(data, weights, centers, gamma):
    """
    Apply the RBF classifier on a data set.

    data[i] = [x1,x2,f(x1,x2)]

    w has K elements equal to number of centers used in classifier.
    w[i] = [w0,w1,..., wk]

    centers has k rows equal to the number of centers used in classifier.
    centers[i] = [x1,x2]

    gamma = RBF kernel param
    """
    N = data.shape[0]
    K = centers.shape[0]
    # make the feature matrix Z
    Z = np.ones( (N, K+1) )
    Z[:, 1:] = np.array([ [RBF_kernel([p[0],p[1]], mu, gamma) for mu in centers] for p in data])
    # get results from weights
    results = np.dot(weights, Z.T)
    return np.sign(results)

def problem14_15(M, gamma=1.5, K=9):
    """
    Train a SVM-classifier, and regular-RBF classifier. Train M of them
    Test on the same generated test dataset.

    Count how many times SVM form beats regular form
    """
    total = M
    SVM_wins = 0.
    Eout_SVM = []
    Eout_RBF = []
    testData = generateData(n=10000)
    for i in np.arange(M):
        trainingData = generateData(n=100)
        clf = libSVM_RBF(trainingData, gamma)
        if not all(clf.predict(trainingData[:, 0:2]) == trainingData[:,2]):
            # if we get inseparable data by SVM, skip this iteration
            total = total - 1
            continue
        weights, centers = RBF_network(trainingData, gamma, K)

        Eout_SVM.append( sum(clf.predict(testData[:, 0:2]) != testData[:,2]) )
        Eout_RBF.append( sum(apply_RBF(testData, weights, centers, gamma) != testData[:,2]) )
        if Eout_SVM[i] < Eout_RBF[i]:
            SVM_wins = SVM_wins+1
    print "SVM wins %0.3f%% of the time" % (SVM_wins/total*100)
    return np.array(Eout_SVM), np.array(Eout_RBF)

def runRBF_network(M, gamma, K):
    """
    Train M regular RBF-classifier, test on the same generated test dataset
    Save both Ein and Eout
    """
    Eout = np.ones( (M,1) )
    Ein = np.ones( (M,1) )
    testData = generateData(n=10000)
    for i in np.arange(M):
        trainingData = generateData(n=100)
        weights, centers = RBF_network(trainingData, gamma, K)

        Ein[i] = sum(apply_RBF(trainingData, weights, centers, gamma) != trainingData[:,2])/100.
        Eout[i] = sum(apply_RBF(testData, weights, centers, gamma) != testData[:,2])/10000.
    return Ein, Eout

def problem16(M=1000):
    """
    run RBF network M times, each time with K=9, then K=12; gamma=1.5
    compare how Ein and Eout changes
    """
    gamma = 1.5
    Ein_k9, Eout_k9 = runRBF_network(M, gamma, 9)
    Ein_k12, Eout_k12 = runRBF_network(M, gamma, 12)

    E_diff = np.hstack( (np.sign(Ein_k12-Ein_k9), np.sign(Eout_k12-Eout_k9)) )
    print "Problem 16:"
    print "k=9 -> k=12, Ein down and Eout up %0.3f%% of the time" % (sum((E_diff==(-1,1)).all(axis=1))*100./M)
    print "k=9 -> k=12, Ein up and Eout down %0.3f%% of the time" % (sum((E_diff==(1,-1)).all(axis=1))*100./M)
    print "k=9 -> k=12, Both Ein and Eout up %0.3f%% of the time" % (sum((E_diff==(1,1)).all(axis=1))*100./M)
    print "k=9 -> k=12, Both Ein and Eout down %0.3f%% of the time" % (sum((E_diff==(-1,-1)).all(axis=1))*100./M)
    print "k=9 -> k=12, Both Ein and Eout same %0.3f%% of the time" % (sum((E_diff==(0,0)).all(axis=1))*100./M)
    return (Ein_k9, Eout_k9, Ein_k12, Eout_k12)

def problem17(M=1000):
    """
    run RBF network M times, each time with gamma=1.5, gamma=2; K=9
    compare how Ein and Eout changes
    """
    K=9
    Ein_1p5, Eout_1p5 = runRBF_network(M, 1.5, K)
    Ein_2, Eout_2 = runRBF_network(M, 2, K)

    E_diff = np.hstack( (np.sign(Ein_2-Ein_1p5), np.sign(Eout_2-Eout_1p5)) )
    print "Problem 17:"
    print "gamma=1.5 -> gamm=2, Ein down and Eout up %0.3f%% of the time" % (sum((E_diff==(-1,1)).all(axis=1))*100./M)
    print "gamma=1.5 -> gamma=2,  Ein up and Eout down %0.3f%% of the time" % (sum((E_diff==(1,-1)).all(axis=1))*100./M)
    print "gamma=1.5 -> gamma=2, Both Ein and Eout up %0.3f%% of the time" % (sum((E_diff==(1,1)).all(axis=1))*100./M)
    print "gamma=1.5 -> gamma=2, Both Ein and Eout down %0.3f%% of the time" % (sum((E_diff==(-1,-1)).all(axis=1))*100./M)
    print "gamma=1.5 -> gamma=2, Both Ein and Eout same %0.3f%% of the time" % (sum((E_diff==(0,0)).all(axis=1))*100./M)

    return (Ein_1p5, Eout_1p5, Ein_2, Eout_2)

def problem18(M=1000):
    """
    Find percentage of time that regular RBF achieves Ein=0 with K=9 and gamma=1.5
    """
    gamma = 1.5
    K = 9
    Ein, Eout = runRBF_network(M, gamma, K)
    print "Regular RBF with K=9, gamma=1.5 achieves Ein=0, %0.3f%% of the time" % (sum(Ein==0)*100./M)[0]
    return Ein,Eout

"""
Given libsvm result and data set, plot the separating plane, margin and the data points
"""
# TODO: visualize SVM results
def plotSVM(data, clf):
    return









        

