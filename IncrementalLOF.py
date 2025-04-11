"""
File: IncrementalLOF.py
Author: Aidan Clark
Updated: 4/10/2025
Description: An implementation of the Incremental Local Outlier Factor algorithm, as described in
this paper: https://ieeexplore.ieee.org/document/4221341
"""

from item import Item

class IncrementalLOF():
    """
    A class for running the Incremental LOF algorithm.
    We follow a streaming model.
    """
    def __init__(self, k):
        """
        k is the number of nearest neighbors to consider in our calculations.
        This can be tuned for the dataset.

        We store a list of Item objects representing all of the items that have been received so far.
        """
        self.data = [] #List of Item objects
        self.k = k

    def __repr__(self):
        return str(self.data)

    def insert(self, point):
        """
        Takes a list/tuple representing a point in Euclidean space.
        Converts that point into an Item object
        Then, it inserts that item into our stored data, in the following way:
            -- We calculate the k-nearest neighbors and the reverse k-nearest neighbors of the point
            -- We update the k-distances, LRD's, and LOF's of any items that are affected by this insert
            -- Finally, we calculate and set the k-distance, LRD, and LOF of the new item
        """
        p = Item(point)

        #Compute KNN and RKNN of p
        self.KNN_and_RKNN(p)

        #See this article for more info: https://ieeexplore.ieee.org/document/4221341
        x_update = p.reverse_knn.copy() #List of Items that we're going to have to update
        x_update_lrd = x_update

        for j in x_update:
            #The k-distances of each of these points should have already been updated
            #when we calculated the RKNN. Although I could have done it more efficiently.
            for i in j.neighbors:
                if i == p:
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
        p.set_lrd(self.k)
        p.set_lof(self.k)

        #Insert p
        self.data.append(p)



    def KNN_and_RKNN(self, p: Item):
        """
        An (inefficient) implementation of finding the k-nearest neighbors and reverse KNN of p
        We update the neighbors and reverse_knn fields of each point in self.data as needed

        This is inefficient because it involves scanning through every point seen so far.
        Note that in the streaming model, this is unacceptable. However, since we will be extending
        this implementation to work for other algorithms that only store a small window, this shouldn't
        make our further algorithms too inefficient.
        """
        for x in self.data:
            #See if x is in p's neighborhood
            if p.insert_into_neighborhood(x, self.k):
                x.reverse_knn.append(p)

            #See if p is in x's neighborhood
            if x.insert_into_neighborhood(p, self.k):
                p.reverse_knn.append(x)

    def batch_insert(self, l):
        """
        l = a list of points to be inserted
        Inserts multiple points at a time
        """
        for x in l:
            self.insert(x)