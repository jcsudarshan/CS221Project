##############################################################################
##############################################################################
# Baseline sentiment classifier
# Gradient descent using hinge loss
# Token level features

import random
import pdb
import collections
import math
import copy
import sys
from collections import Counter
from util_baseline import *

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    words = x.split()
    sparse_vec = collections.Counter(words)
    return dict(sparse_vec)

def learnPredictor(trainExamples, testExamples, featureExtractor, eta=0.1, numIters=10):
    #pdb.set_trace()
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    numIters refers to a variable you need to declare. It is not passed in.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    def takeGradient(w, i):
        x, y = trainExamples[i]                    # raw string data and label
        phi_x = featureExtractor(x)                # feature vector
        margin = dotProduct(w, phi_x) * y          # calculate margin

        if margin > 0:
            return dict({})
        else:
            keys = phi_x.keys()
            for k in keys:
                phi_x[k] *= -y
            return phi_x

    def stochasticGradientDescent(takeGradient, w, n):
        # parameters defined
        # ------------------
        # w                     --> weight vector
        # takeGradient()        --> gradient function (hinge loss)
        # n                     --> num training observations

        # hyperparams to fit
        # ------------------
        # eta, total updates    --> eta = \sqrt( # of updates made )
        # numIters              --> # of iterations (must be <= 20)        
        
        random.seed(7)                              # for random.randint
        w.update({'intercept':1})                   # add intercept
        for t in range(numIters):
            iters = range(0, n)
            random.shuffle(iters)
            for i in iters:
                gradient = takeGradient(w, i)
                increment(w, -eta, gradient)
        
        return w

    weights = stochasticGradientDescent(takeGradient, weights, len(trainExamples))
    return weights