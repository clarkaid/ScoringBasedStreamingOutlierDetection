"""
Class for a generic "item" that have distances between them-- i.e. points in the Euclidean space

"""

import bisect #For sorted array inserts

class Item:

    def __init__(self, tuple):
        self.tuple = tuple
        self.neighborhood = []
        #The k-distance neighborhood is every Item in our dataset whose distance to p
        # is not greater than k-dist(p)
        #According to the way I wrote self.neighborhood, this is
        #a list of >=k Item objects in sorted order by increasing distance from self
        
    def dist(self, other):
        """
        Simple Euclidean distance. I suppose we could override this if we wanted something different
        """
        sum = 0
        for i in range(len(self.tuple)):
            sum += (self.tuple[i] - other.tuple[i]) ** 2

        return sum ** (1 / 2)
    
    def __repr__(self):
        return self.tuple
    
    def set_k_neighborhood(self, D, k):
    
        """
        self = currently inserted point
        D = our database of points-- list of Item objects
        k = number of neighbors to consider (parameter of overall algo)

        Updates k nearest neighborhood parameter of self
        """

        def insert_and_trim (distance, item, cur_lis):
            """
            Insert the new item into the neighborhood, maintaining sorted order by increasing distance
            Trim list to maintain constraints.
            i..e. we want no more than k-1 items to have distances strictly less than k-dist(item)
            and we want at least k items to have distances <= k-dist(item)
                -- so we preserve duplicates of the kth item's distance!
            """
            cur_lis = bisect.insort([distance, item], cur_lis, key = lambda x : x[0])

            #Trim list s.t. we have at most k - 1 items whose distances are strictly less than
            #the highest distance in the list-- per original paper on LOF
            num_seen = 0
            most_recent_dist = cur_lis[0]
            for i in range(len(cur_lis)):
                if i < k - 1:
                    num_seen += 1 #I want to count each of the k-1 first items
                elif i == k - 1:
                    #This is the kth item because of zero-indexing
                    num_seen += 1
                    most_recent_dist = cur_lis[i][0]
                else:
                    if cur_lis[i][0] == most_recent_dist:
                        #keep duplicates
                        continue
                    else:
                        #We can exclude everything else in our list
                        cur_lis = cur_lis[0:i]
                        break
            return cur_lis


        cur_neighborhood = [] #Will be lists of lists
        size = 0
        for other in D:
            distance = self.dist(other)
            if size < k or distance <= cur_neighborhood[-1]:
               #Insert item and adjust list to maintain neighborhood constraints
               cur_neighborhood = insert_and_trim(distance, self, cur_neighborhood)

        self.neighborhood = [x[1] for x in cur_neighborhood] #Just Item objects

    def k_dist(self):
        """
        Returns distance from self to its k-nearest neighbor
        """
        if self.neighborhood == []:
            raise Exception("Did you set the k-neighborhood of this point yet?")
        
        return self.neighborhood[-1] #Assumes that neighborhood is in increasing sorted order by distance

    def reach_dist(self, other):
        """
        self and other are Item objects
        """
        k_dist_self = self.k_dist()
        dist = self.dist(other)

        return max(k_dist_self, dist)

    def lrd(self, k):
        """
        Local reachability density for self
        """
        sum = 0
        for x in self.neighborhood:
            sum += self.reach_dist(x)

        if sum == 0:
            raise Exception("Can't take inverse of 0. Does this Item have a neighborhood?")

        return ((1 / k) * sum) ** (-1)
    
    def lof(self, k):
        """
        Local outlier factor for self
        """
        self_lrd = self.lrd(k)
        sum = 0
        for x in self.neighborhood:
            sum += (x.lrd(k) / self_lrd)

        return (1 / k) * sum