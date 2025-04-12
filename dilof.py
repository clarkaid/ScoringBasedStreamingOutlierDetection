"""
File: dilof.py
Author: Aidan Clark
Updated: 4/11/2025
Description: An implementation of the Density summarizing Incremental Local Outlier Factor (DILOF) algorithm,
as described in this paper:
https://dl-acm-org.ezproxy.bu.edu/doi/pdf/10.1145/3219819.3220022
"""

from item import Item
from IncrementalLOF import IncrementalLOF

import heapq #Heap, good for getting n smallest items in list
import math
import statistics


class DILOF(IncrementalLOF):
    """
    A class for running the Density summarization Incremental LOF algorithm (DILOF)

    This is an extension of the Incremental LOF algorithm.
    """
    def __init__(self, k:int, window_size:int, threshold:float, step_size:float, regularizer:float, max_iter:int, skipping_enabled = False, print_outliers = False):
        """
        Inputs:
        k = Number of nearest neighbors to consider
        window_size = Maximum number of points to store at any given time
        threshold = LOF boundary for deciding whether a point is an outlier or not
        step_size, regularizer, max_iter: used for gradient descent
        skipping_enabled = Should we attempt to skip sequences of outliers
        print_outliers = If True, prints a message each time an outlier is detected
        """
        
        self.data = [] #List of Item objects
        self.k = k #Number of nearest neighbors to consider

        self.outliers = [] #List of outliers detected

        self.window_size = window_size #Maximum window size-- this is how many points we'll store!
        self.threshold = threshold #LOF Threshold for outlier scores. Above this threshold = outlier

        self.skipping_enabled = skipping_enabled #Whether we use skipping scheme in lod()
        self.try_skip_next = False #Set to True when we see an outlier because it might be the start of a sequence

        #Parameters for gradient descent in NDS
        self.step_size = step_size
        self.regularizer = regularizer
        self.max_iter = max_iter

        self.print_outliers = print_outliers

    def insert(self, tuple):
        """
        Takes a list/tuple representing a point in Euclidean space.
        Converts tuple into an Item object and then executes one iteration of the DILOF algorithm:
            -- We run the LOD algorithm, which is very similar to one iteration of the Incremental LOF algorithm
            -- If we have stored our maximum window size, we run the NDS algorithm
                    -- This summarizes the oldest W/2 points using a gradient descent density sampling strategy
        
        This method is the same as Algorithm 1: DILOF in the paper linked in the heading comment.
        """
        item = Item(tuple)

        res = self.lod(item)
        if res >= 0:
            #Then we aren't skipping this point
            #The original algo has > 0, but I changed it to >= because we set LOF to 0 when there's only one point
            self.data.append(item) #item's stats and the stats of its neighbors/reverse_knn's should be already updated by lod
            if len(self.data) == self.window_size:
                z_summary = self.nds() #Summarize last W/2 points

                self.data = self.data[self.window_size // 2:] #Remove oldest W/2

                self.data = z_summary + self.data #Add summary (W/4 points) to front

    
    def lod(self, item):
        """
        Described in Algorithm 4 of the above paper.
        Essentially does the same thing as incremental LOF, with an added skipping option

        Returns the LOF of the point OR -1 if the point should be skipped
        """

        #Should we skip this point? i.e. is it part of a sequence of outliers?
        if self.skipping_enabled:
            if self.skipping_scheme():
                return -1

        #Compute KNN and RKNN of p
        self.KNN_and_RKNN(item)

        #See this article for more info: https://ieeexplore.ieee.org/document/4221341
        x_update = item.reverse_knn.copy() #List of Items that we're going to have to update
        x_update_lrd = x_update

        for j in x_update:
            #The k-distances of each of these points should have already been updated
            #when we calculated the RKNN. Although I could have done it more efficiently.
            for i in j.neighbors:
                if i == item:
                    continue
                else:
                    if j in i.neighbors and j not in x_update_lrd:
                        x_update_lrd.append(i)

        x_update_lof = x_update_lrd

        for m in x_update_lrd:
            m.set_lrd(self.k)
            for y in m.neighbors:
                if y not in x_update_lof:
                    x_update_lof.append(y)


        for l in x_update_lof:
            l.set_lof(self.k)
            
        #All of the other necessary Items have been updated.
        #Now, calculate p's stats
        item.set_lrd(self.k)
        item.set_lof(self.k)

        #Make decision about whether this NEW point is an outlier or not
        #Notice that, while we recompute old LOF's as necessary, we do not reevaluate outlier decisions

        if item.lof > self.threshold:
            self.outliers.append(item)
            if self.print_outliers:
                print("Outlier Detected:", item.tuple, "with a LOF of", item.lof)

            #There's more stuff to do with skipping here. Implement Later.

        #Note that the item isn't actually inserted here.
        return item.lof
    

    def skipping_scheme(self, item):
        skip_insertion = False
        if self.try_skip_next:
            #Need to compute d1(x)_bar --> the average distance from a point to its first nearest neighbor
            sum_d1 = 0
            for x in self.data:
                sum_d1 += x.dist(x.neighbors[0])

            d1_bar = sum_d1 / len(self.data)

            if item.dist(self.outliers[-1]) < d1_bar:
                #i.e. if this new point is closer to the most recent outlier than average distance
                #Then it is probably part of a sequence of outliers.
                #Add to outlier list and do not insert it
                self.outliers.append(item)
                print("Outlier Detected:", item, "as part of a sequence")

                skip_insertion = True
        
        return skip_insertion

    
    def nds(self):
        """
        Returns a list of self.window_size / 4 points to summarize the oldest self.window_size / 2 points.

        When the number of points stored reaches the maximum window size W, we want to summarize the
        oldest W/2 points (call this set X) by selecting W/4 of them (call this set Z) that best "represent" X.
        To put it more formally, we want to select Z (a subset of X) s.t. the density difference between
        X and Z is minimized. The effect of this is that we select more points in higher density clusters,
        which helps reduce noise and assists with outlier detection as later points arrive.

        The way we achieve this is a bit complicated, so buckle up. As the paper describes, they use
        a nonparametric density sampling technique, made possible with a gradient descent algorithm.
        In each iteration, we update decision variables y_i using the update formula. This formula can involve
        recalculating the k nearest neighbors in every iteration; to avoid this, the paper describes a couple of
        heuristics for approximating some necessary values for the formula. After we reach the maximum
        number of iterations (a hyperparameter of the algorithm), we pick the W/4 highest y_i's (i.e. project
        them into binary domain) and those are our Z.

        One should note that this requires quite a lot of time and calculations. Also, the success of this 
        partially depends on the choice of hyperparameters-- in particular, the maximum number of iterations,
        the step size, and the regularization constant. Some tuning may be required to make this work successfully.
        """

        #X = W/2 oldest points.
        X = self.data[0:(self.window_size // 2)]

        #Z = Subset of X of size W/4 that we'll ultimately select

        #We need vk(x) for each x in X, where
        #vk(x) = distance between x and its kth nearest neighbor in X. This is NOT the same as the
        #k-distance because X is only the W/2 oldest points. (Although, what would happen if we just used
        #the k-distance)?
        #To calculate this, I need pairwise distances.

        #Calculate pairwise distances for X
        pairwise_distances = [[x.dist(y) for y in X] for x in X]

        v_k, X_knn = self.run_knn(pairwise_distances, X)

        #In the exact algorithm, C_k(n) (the set of data points that have x_n as their nearest neighbor in Z)
        # has to be recomputed after each iteration. The authors offer a heuristic way of computing it
        #From what I can tell, this only has to be computed once and is then used throughout.
        C_k = self.approximate_Ck(X, v_k)

        #Similarly, to avoid recomputing p_k(n) every iteration, we use a heuristic
        #Again, I believe this does not depend on the current value of our decision variables,
        #so it can be computed just once
        p_k = self.approximate_pk(v_k, pairwise_distances, X)
        
        #Now let's start working with our decision variables
        Y = [0.5 for _ in range(len(X))]

        cur_step_size = self.step_size
        for t in range(self.max_iter):
            old_y = Y.copy()
            cur_step_size *= 0.95 #Shrinks step size each iteration (why?)
            for n in range(len(X)):
                #Update each decision variable-- i.e. going from time t to time t+1
                update = sum([old_y[i] for i in C_k[n]])
                update += (p_k[n] / v_k[n])
                update -= math.exp(X[n].lof)
                update += self.psi_prime(old_y[n])
                update += self.regularizer * (sum(old_y) - self.window_size / 4)
                update *= cur_step_size

                Y[n] = old_y[n] - update

        #Decision variables have been calculated. Now, pick the W/4 highest and report those
        #Since |X| = W/2, W/4 highest are all that are higher than median

        med = statistics.median(Y)
        Z = [X[i] for i in range(len(X)) if Y[i] >= med]

        return Z




    def run_knn(self, pairwise_distances, X):
        """
        Computes v_k and the k nearest neighbors in X. Returns them as a tuple.
        """
        v_k = [0 for _ in range(len(pairwise_distances))]
        X_knn = [[] for _ in range(len(pairwise_distances))]

        for r in range(len(pairwise_distances)):
            with_i = [[pairwise_distances[r][i], i] for i in range(len(pairwise_distances))]

            #I expect k to be small compared to W, so a heap seems like a good way of getting k smallest
            nearest_neighbors = heapq.nsmallest(n = self.k, iterable = with_i, key = lambda x: x[0])
            X_knn[r] = [X[n[1]]for n in nearest_neighbors] #Get Item objects, so it's a list of k-nearest neighbors in X
            v_k[r] = nearest_neighbors[-1][0] #Distance to k-nearest neighbor in X

        return (v_k, X_knn)
    

    def approximate_Ck(self, X, v_k):
        """
        Approximates Ck using the method described in algorithm 3

        C_k[n] = the set of indices of data points that have x_n as their nearest neighbor in Z
        """
        C_k = [[] for _ in range(len(X))]

        s = [0 for _ in range(len(X))] #Used for approximating KNN

        for n in range(len(X)):
            for q in X[n].neighbors:
                s[n] += math.exp(sigmoid(q.lof))

        s_bar = sum(s) / len(s)

        for i in range(len(X)):
            if s[i] > s_bar:
                for n in range(len(X)):
                    if v_k[i] < X[i].dist(X[n]) < 2 * math.exp(sigmoid(X[i].lof)) * v_k[i]:
                        C_k[n].append(i)
                        break #Why break?
        
        return C_k

    def approximate_pk(self, v_k, pairwise_distances, X):
        """
        Returns approximate pk using equations 12 and 13 from the paper
        """
        p_k = v_k.copy() #p_k is at least v_k, with an additional shift

        beta_denom = sum([math.exp(sigmoid(x.lof)) for x in X ])

        for i in range(len(X)):
            beta_numer = sum([math.exp(sigmoid(x.lof)) for x in X[i].neighbors])
            beta = beta_numer / beta_denom
            update = beta * (max(pairwise_distances[i]) - v_k[i])
            p_k[i] += update

        return p_k
    
    def psi_prime(self, y):
        """
        Computes the derivative of psi applied to y
        Psi is defined in equation equation 8
        """
        if y > 1:
            return 2 * (y - 1)
        elif y < 0:
            return 2 * y
        else:
            return 0

def sigmoid(x):
        """
        Compute the sigmoid function of x.
        """
        return 1 / (1 + math.exp(-1 * x))