"""
Code for calculating different generalization error bounds for Homework 4

Question 2-3
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6)
np.set_printoptions(threshold=10)

dvc = 50.0
delta = 0.05

def m_H(N, dvc):
    """
        Approximate growth function m_H(N) by N^dvc
    """
    return N**dvc

def original_VC(N, dvc, delta):
    """
        Given sample size, vc-dimension, and confidence level,
        return the generalization error.
        
        Inputs:
            N: sample size
            dvc: vc-dimension
            delta: confidence level.

        Outputs:
            epsilon: generalization error bound
    """
    return np.sqrt((8.0/N) * np.log1p(4*m_H(2*N, dvc)/delta))

def Rademacher(N, dvc, delta):
    a = np.sqrt((2.0/N) * np.log1p(2*N*m_H(N, dvc)))
    b = np.sqrt((2.0/N) * np.log1p(1/delta))
    c = 1.0/N
    return a+b+c

def Parronda(N, dvc, delta):
    """
        Implicit bound in epsilon, use iterative method
    """
    tol = 1e-5
    prev_result = 0.5
    result = np.sqrt((2*prev_result + np.log1p(6*m_H(2*N, dvc)/delta)/N))
   

    while abs(result - prev_result) > tol:
        prev_result = result
        result = np.sqrt((2*prev_result + np.log1p(6*m_H(2*N, dvc)/delta))/N)

    return result
        
def Devroye(N, dvc, delta):
    tol = 1e-5
    prev_result = 0.5
    result = np.sqrt((4*prev_result*(1+prev_result + np.log1p(4/delta) + np.log1p(N)*2*dvc))/(2*N))
    while abs(result - prev_result) > tol:
        prev_result = result
        result = np.sqrt((4*prev_result*(1+prev_result + np.log1p(4/delta) + np.log1p(N)*2*dvc))/(2*N))
    return result

def problem2():
    n = 10000
    print "For N = 10,000,\n\
           original_VC = %0.5f,\n\
           rademacher = %0.5f, \n\
           parronda = %0.5f, \n\
           devroye = %0.5f" \
          % (original_VC(n, dvc, delta), Rademacher(n, dvc, delta), \
             Parronda(n, dvc, delta), Devroye(n, dvc, delta))
    
def problem3():
    n = 5
    print "For N = 5,\n\
           original_VC = %0.5f,\n\
           rademacher = %0.5f, \n\
           parronda = %0.5f, \n\
           devroye = %0.5f" \
          % (original_VC(n, dvc, delta), Rademacher(n, dvc, delta), \
             Parronda(n, dvc, delta), Devroye(n, dvc, delta))
