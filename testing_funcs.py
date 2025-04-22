"""
File: testing_funcs.py
Author: Aidan Clark
Date: 4/22/25
Description: A few useful functions for generating/testing datasets for outlier detection algorithms
"""
import random
import numpy as np

from IncrementalLOF import IncrementalLOF
from dilof import DILOF

from SlidingWindow import KDEWRStreaming

from milof import milof

def generate_dataset(n_samples, centers, stdevs, center_weights, outlier_center, outlier_spread, outlier_prop, outlier_sequence = False, outlier_sequence_length = 5):
    dimension = len(centers[0])

    X = np.empty((n_samples, dimension))
    labels = np.empty(n_samples)

    i = 0
    while i < n_samples:
        #Do a biased coin flip to decide whether we have an outlier or not
        if random.random() < outlier_prop:
            #Then we have an outlier
            sample = np.random.multivariate_normal(mean = outlier_center,
                                                   cov = np.diag([outlier_spread] * dimension))
            
            X[i] = sample
            labels[i] = -1 #Outlier
            i += 1

            if outlier_sequence:
                seq_length = 1
                while random.random() < 0.3 and seq_length <= outlier_sequence_length:
                    #Then get another outlier
                     sample = np.random.multivariate_normal(mean = outlier_center,
                                                   cov = np.diag([outlier_spread] * dimension))
            
                     X[i] = sample
                     labels[i] = -1 #Outlier
                     i += 1
                     seq_length += 1
        else:
            #Not an outlier. Sample from regular centers
            c = random.choices(range(len(centers)), weights = center_weights)[0]
            sample = np.random.multivariate_normal(mean = centers[c],
                                                   cov = np.diag([stdevs[c]] * dimension))
            
            X[i] = sample
            labels[i] = c #Center index

            i += 1

    
    return (X, labels)

def test(data, method, k = 30, threshold = 1, window_size = 200, skipping_enabled = False, num_clusters = 10):
    """
    Takes an array/list of tuples, data, and a outlier detection method (IncrementalLOF, DILOF, MILOF, KDEWR)
    and runs that algorithm on data. Returns a list of the same length as data of the decisions made by the algo.
    0 for non-outlier, -1 for outlier
    """
    match method:
        case "IncrementalLOF":
            alg = IncrementalLOF(k = k, threshold = threshold)
        case "DILOF":
            alg = DILOF(k = k,
                        threshold = threshold,
                        window_size = window_size,
                        step_size = 0.1,
                        regularizer = 0.001,
                        max_iter = 50,
                        skipping_enabled = skipping_enabled)
        case "MILOF":
            alg = milof(
                    k = k,
                    b = window_size,
                    threshold = threshold,
                    c = num_clusters
            )
        case "KDEWR":
            alg = KDEWRStreaming(windowSize = window_size, 
                                 r = threshold)
    
    classification = []
    scores = []
    for d in data:
        (decision, score) = alg.insert(d)
        scores.append(score)
        classification.append(decision)
    
    return (classification, scores)

def f1_score(estimated, actual):
    """
    Assume they're both bool lists of equal length. True = outlier, False = not outlier
    """
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    for i in range(len(estimated)):
        if estimated[i] == actual[i]:
            if actual[i] == True:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if actual[i] == True:
                #Then estimated didn't pick up on it
                false_negatives += 1
            else:
                false_positives += 1
    
    f1_score = 2*true_positives / (2*true_positives + false_positives + false_negatives)

    return f1_score