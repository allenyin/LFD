"""
Code for Question 1 and 2, coin flip questions related to Hoeffding Inequality
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(threshold=10)

def flip_1000_coins():
    """
    Flip 1000 coins 10 times and record number of heads in array,
    return the final array.
    """
    results = np.ones((1000,1))*np.NAN
    for i in range(1000):
        results[i] = sum(np.random.choice([1,0],10))
    return results

def get_wanted_coins(results):
    """
    From the result of the number of heads from flipping 1000 coins 10 times, get
    1) c1 = # of heads from 1st coin flipped.
    2) c_rand = # of heads from a random coin.
    3) c_min = minimum # of heads among the 1000.

    return them in tuple.
    """

    c1 = results[0]
    c_rand = results[np.random.choice(range(len(results)))]
    c_min = min(results)
    
    return (c1, c_rand, c_min)

"""
Now run the flip_1000_coins 100,000 times. And save the wanted_coins result to get distribution
"""
N = 100000
c1_results = np.ones((N,1))*np.NAN
c_rand_results = np.ones((N,1))*np.NAN
c_min_results = np.ones((N,1))*np.NAN

for i in range(N):
    exp_result = flip_1000_coins()
    (c1_results[i], c_rand_results[i], c_min_results[i]) = get_wanted_coins(exp_result)

print "Average value of nu_1 = %0.3f" % (np.mean(c1_results)/10)
print "Average value of nu_rand = %0.3f" % (np.mean(c_rand_results)/10)
print "Average value of nu_min = %0.3f" % (np.mean(c_min_results)/10)
