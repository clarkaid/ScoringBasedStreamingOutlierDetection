"""
File: item.py
Author: Aidan Clark
Last Updated: 4/9/2025
"""

import bisect

class Item:
    """
    A class that represents items in a data stream
    For use by outlier detection algorithms that use an incremental LOF strategy
    """
    def __init__(self, tuple):
        """
        We take a tuple/list/whatever that represents a point in the Euclidean space

        We store the k-nearest neighbors and the reverse KNN of the point.
        We also cache the k_distance, LRD, and LOF of the point
        We assume that the algorithm actually performing the insertion of the
        point makes the necessary method calls to update these values!!
        """
        self.tuple = tuple

        #Neighborhood information
        self.neighbors = [] #List of k-nearest neighbors
        self.reverse_knn = [] #List of points whose neighborhoods include self

        #Cached Values
        self.k_distance = None
        self.lrd = None
        self.lof = None

    def __repr__(self):
        return "Item: " + str(self.tuple)
    
    def dist(self, other):
        """
        Simple Euclidean distance between two Item objects
        """

        sum = 0
        for i in range(len(self.tuple)):
            sum += (self.tuple[i] - other.tuple[i]) ** 2

        return sum ** (1 / 2)
    
    def insert_into_neighborhood(self, new, k):
        """
        Takes a new item and a value k and tries to insert new into the neighborhood of self

        If this succeeds, the neighbors field AND the k_distance field are updated and we return True.

        Otherwise, we return False.

        When inserting, we maintain the following constraints:
            - The neighbors field is in increasing sorted order by distance from self
            - At most k-1 items in neighbors have distances strictly less than self.k_distance
            - At least k items in neighbors have distances at most self.k_distance
        """

        distance = self.dist(new)

        if len(self.neighbors) < k or self.k_distance == None or distance <= self.k_distance:
             #Then we're adding to neighbors
             if self.k_distance == None or distance > self.k_distance:
                 self.k_distance = distance
                 self.neighbors.append(new)
             elif distance == self.k_distance:
                 #Then k_distance won't change. We just need to add to neighbors
                 self.neighbors.append(new)
             else:
                 #Then the k-distance will become the distance to the k-1th neighbor
                   n = [[self.dist(x), x] for x in self.neighbors]
                   bisect.insort(n, [distance, new], key = lambda x: x[0])

                   #Adjust rKNN's of any Items we remove
                   for i in range(k, len(n)):
                       elem = n[i][1]
                       if self in elem.reverse_knn:
                           elem.reverse_knn.remove(self)
                   
                   n = n[0:k]

                   self.neighbors = [x[1] for x in n]
                   self.k_distance = self.dist(self.neighbors[-1])

             return True

        else:
            return False
        
    def reach_dist(self, other):
        """
        From Wikipedia: the reachability distance of an object A from B is the 
        true distance between the two objects, but at least the k-distance of B

        Note that we assume that both objects have already had their k-distances calculated and set.
        This occurs when we update the neighbors of each.
        """
        distance = self.dist(other)
        return max(distance, other.k_distance)
    
    def set_lrd(self, k):
        """
        Sets Local reachability density for self

        Note: If the dataset only has one point (or one kind of point), LRD is set to 0, but it's really undefined.
        """
        sum = 0
        for x in self.neighbors:
            sum += self.reach_dist(x)

        if sum == 0:
            #raise Exception("Can't take inverse of 0. Does this Item have a neighborhood?")
            self.lrd = 0 #Trying to avoid an exception

        else:
            self.lrd = ((1 / k) * sum) ** (-1)
    
    def set_lof(self, k):
        """
        Sets Local outlier factor for self

        Note: If the dataset only consists of 1 point, LOF is set to 0, but it's really undefined
        """
        #lrd's should already be set
        sum = 0

        if self.lrd == 0:
            self.lof = 0 #Filler
        else:
            for x in self.neighbors:
                sum += (x.lrd / self.lrd)

            self.lof = (1 / k) * sum


#Tests
"""
x = Item([2])
print(x.neighbors)
y = Item([3])
x.insert_into_neighborhood(y, 2)
print("The neighborhood of x is", x.neighbors)
y.insert_into_neighborhood(x, 2)

z = Item([6])
w = Item ([5])
x.insert_into_neighborhood(z, 2)
print("The neighborhood of x is", x.neighbors)
x.insert_into_neighborhood(w, 2)
print("The neighborhood of x is", x.neighbors)

a = Item([5])
x.insert_into_neighborhood(a, 2)
print("The neighborhood of x is", x.neighbors)
"""