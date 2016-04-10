The [LFD final](https://work.caltech.edu/homework/final.pdf)
The [LFD final solutions](https://work.caltech.edu/homework/final_sol.pdf)

Notes on the questions and running the code.

**Problem 7-10**: Questions on Regularized Linear Regressions, see `regressions.py`

Straight forward, run `problem7()`, `problem8()`, `problem9()` and `problem10()` to run tests and print results.

**Problem 11-12**: Questions on Support Vector Machine, see `SVM.py`

`problem11()` plot the transformed points, as needed by problem 11.

problem 12 requires using SVM to find a decision boundary for the same 7 point data set after transforming into Z-space, using the specified kernel.

Run `problem12()` to train the classifier and give the number of support vectors. Note that within this function, two classifiers will be trained. One is using Python's sklearn, an implementation of libsvm (easy to use). The second one uses the handrolled SVM code (`SVM_kernel()`) that uses cvxopt for Quadratic-Programming, and returns the weights, bias, and the number of support vectors.

**Problem 13-18**: Questions on Radial Basis Functions (RBF), see `RBF.py`

These series of questions compares using RBF for classification:

1. The regular way - select k centers from training data set through Lloyd's clustering algorithm, then train the classifier as a regression problem, with the basis as RBF centered arouund these centers.
2. The SVM way - Use hard-margin SVM with RBF kernel. Implemented using `sklearn` module.

Utility functions:

1. `generateData(n=100)` to generate number of desired data points for these problem, format `[x1, x2, f(x1,x2)]`
2. `plot_fn_and_data(data, fign=1)` plots the data points along with the sampling function.
3. `lloyd_clustering(data, K, plot=True)` finds K centers for the given data set. Initialize the centers prior to the first iteration via greedy algorithm rather than randomly select points. This is for faster convergence. Iteration upper limit set to 100. Option `plot`, when set to `True`, will plot the data points with the initial centers in the beginning, and the data points with the converged centers at the end, as well as printing the number of iterations taken and the error.

Problem 13 - Run `problem13()`

Problem 14, 15 - Run `problem14_15(M, gamma=1.5, K)`, where K=9 for problem 14 and K=12 for problem 15. For each of these two problems, 1000 comparisons were ran. Each comparison consisted of training a RBF-SVM and a regular-RBF classifier on the same randomly generated 100-points training data set. If the data set is inseparable by SVM, it is skipped (and total number of runs subtracted by 1). The resulting classifiers are then tested on a single 10,000 points test data set.

For both tests, RBF-SVM beats regular-RBF Eout performance over 90% of the time. The solution has that percentage in (60%, 90%] for problem 15.

Problem 16 - Run `problem16()`. Default is M = 1000 comparisons of running regular-RBF with gamma=1.5, and K={9,12}. Scheme is training (separately) M classifiers of a particular type on different 100-point training sets, and then testing all of those classifiers (of that type) on a 10,000 points test set. It was noticed setting M to 200 is enough for stable answer, which takes around 2 minutes to finish.

Problem 17 - Run `problem17()`. Same test setup as previously, with different parameters. Again, setting M to 200 is enough, runs in 2 minutes.

Problem 18 - Run `problem18()`. Same test setup as previous two. Set M=200, runs in 2 minutes.

`
