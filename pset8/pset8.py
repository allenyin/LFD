"""
LFD pset8

Using scikit learn's SVM package for the computation.
scikit learn's SVM package is based on libSVM.

Want to implement SVM with soft-margin. This is using libsvm/scikit-learn's
SVC method.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import cross_validation

# Data set, format is (digit, intensity, symmetry)
trainingData = np.loadtxt("features.train")
testData = np.loadtxt("features.test")

"""
Problem 2-6: Polynomial kernels.
Polynomial kernel: (1 + x'x)^Q
For SVC: (gamma*(x'x)+r)^d*d --> gamma=1, r=coef0=1, degree=d=Q
"""

def problem2_3_4():
    """
    Implement class-1 vs. all classifiers,
    with C = 0.01 and Q = 2 and compare performances.
    """
    C = 0.01
    Q = 2
    clf = SVC(C=C, kernel='poly', degree=Q, gamma=1, coef0=1)
    Ein = []
    Eout = []
    nSV = []

    # fit 10 classifiers for each digit
    for i in xrange(10):
        y = trainingData[:,0]
        y = np.array([1 if j==i else -1 for j in y])
        clf.fit(trainingData[:, 1:], y)
        nSV.append(sum(clf.n_support_))
        # get Ein
        Ein.append(1 - clf.score(trainingData[:, 1:], y))

        # get Eout
        y = testData[:,0]
        y = np.array([1 if j==i else -1 for j in y])
        Eout.append(1 - clf.score(testData[:, 1:], y))

    idx_maxEin = np.argmax(Ein)
    idx_minEin = np.argmin(Ein)
    print "%d vs. all has the highest Ein" % idx_maxEin
    print "%d vs. all has the lowest Ein" % idx_minEin
    print "Classifier (%d vs all) had %d more support vectors than Classifier (%d vs all)" % (idx_maxEin, nSV[idx_maxEin]-nSV[idx_minEin], idx_minEin)

def problem5_6():
    """
    Implement class-1 vs. class-5 classifier
    with C = 0.001, 0.01, 0.1, 1
    and Q = 2.

    By definition of hard-constraint SVM, max C achieves the lowest Ein
    """
    C = [0.0001, 0.001, 0.01, 0.1, 1.0]
    Q = [2, 5]
    param_list = [(c,q) for c in C for q in Q]
    param_list.sort(key = lambda tup: tup[1])
    stats = {}
    
    training = trainingData[np.logical_or(trainingData[:,0]==1, trainingData[:,0]==5), :]
    test = testData[np.logical_or(testData[:,0]==1, testData[:,0]==5), :]
    # keep track of each combination of (c,q) in terms of the [Ein, Eout, nSV]
    for c,q in param_list:
        curStat = [0,0,0]
        clf = SVC(C=c, kernel='poly', degree=q, gamma=1, coef0=1)
        
        clf.fit(training[:, 1:], training[:,0])
        # number of support vectors
        curStat[-1] = sum(clf.n_support_)
        
        # Ein
        curStat[0] = 1 - clf.score(training[:, 1:], training[:,0])
        
        # Eout
        curStat[1] = 1 - clf.score(test[:, 1:], test[:,0])

        stats[(c,q)] = curStat
    
    print "{:6s} {:2s} {:7s} {:7s} {:5s}".format('C', 'Q', 'Ein', 'Eout', 'nSV')
    for k in param_list:
        print "{:6.4f} {:2d} {:7.4f} {:7.4f} {:5d}".format(k[0], k[1], stats[k][0], stats[k][1], stats[k][2])
        
"""
10-fold cross-validation with polynomial kernel.
Try 100-runs with different partitions.

Problem compares the 1 vs. 5 classifier with Q=2, choosing between C values
"""
def problem7_8():
    C = [0.0001, 0.001, 0.01, 0.1, 1]
    Q = 2
    folds = 10

    training = trainingData[np.logical_or(trainingData[:,0]==1, trainingData[:,0]==5), :]

    Ecv = np.ones((len(C), 100))*np.NAN

    for i in xrange(100):
        kf = cross_validation.KFold(training.shape[0], n_folds=folds)
        for j in xrange(len(C)):
            clf = SVC(C=C[j], kernel='poly', degree=Q, gamma=1, coef0=1)
            curScore = []
            for train_idx, test_idx in kf:
                clf.fit(training[train_idx, 1:], training[train_idx, 0])
                curScore.append(1 - clf.score(training[test_idx, 1:], training[test_idx,0]))
            Ecv[j,i] = np.mean(curScore)

    winningC_idx = np.argmax(np.bincount(np.argmin(Ecv,0)))
    winningC = C[winningC_idx]
    winningEcv = np.mean(Ecv[winningC_idx,:])
    print "For 1 vs. 5 classifier, C=%0.4f wins, with Ecv=%0.4f" % (winningC, winningEcv)
    return Ecv
    
"""
Problem 9, 10: RBF soft-constraint SVM comparions.

RBF: exp(-|xn-xm|^2)
SVC RBF: exp(-gamma*|x-x'|^2) --> gamma=1
"""
def problem9_10():
    C = [0.01, 1, 100, 1e4, 1e6]

    training = trainingData[np.logical_or(trainingData[:,0]==1, trainingData[:,0]==5), :]
    test = testData[np.logical_or(testData[:,0]==1, testData[:,0]==5), :]

    Ein = []
    Eout = []

    for c in C:
        clf = SVC(C=c, kernel='rbf', gamma=1)
        clf.fit(training[:, 1:], training[:, 0])
        Ein.append(1 - clf.score(training[:, 1:], training[:, 0]))
        Eout.append(1 - clf.score(test[:, 1:], test[:, 0]))

    print "For 1 vs. 5 classifier: \nC=%0.4f has lowest Ein, \nC=%0.4f has lowest Eout" % (C[np.argmin(Ein)], C[np.argmin(Eout)])
    return (Ein, Eout)










        


        

