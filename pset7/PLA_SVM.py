"""
Comparing PLA vs. SVM performance, question 8-10 for LFD pset7
"""

import sys
import os
LFDpath = "/".join(os.path.dirname(__file__).split('/')[0:-1])
if not LFDpath in sys.path:
    sys.path.append(LFDpath)
from pset1.PLA import *

# PLA include the code needed to generate 2D test data, target function, and running PLA

def runComparison(N):
    """
    For training-sample size N, run experiment 1000 times, where in each iteration:
        1. Create target function and training data.
        2. Train PLA and SVM
        3. Create 1000 testing points, and compare the misclassification performance
           as percentage of disagreements.
    """
    PLA_miss = np.zeros((1000,1))
    SVM_miss = np.zeros((1000,1))
    ntest = 1000
    for i in range(1000):
        f = generate_targetFn()
        trainingSet = generate_dataPoints_fromFn(N, f)
        testingSet = generate_dataPoints_fromFn(ntest, f)

        # train PLA
        step_lim = 10000
        classifier = PLA(trainingSet)
        classifier.train(lim = step_lim)
        (results, misclassified) = classifer.classify(testingSet)
        PLA_miss[i] = misclassified/(ntest*1.0)


def SVM(data):
    """
    Use Quadratic Programming to solve for SVM from given data




