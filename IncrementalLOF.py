"""
A quick implementation of Incremental LOF using the methods in Item

This can be extended for other adaptations of incremental LOF
"""

from item import Item

class IncrementalLOF():
    def __init__(self, k):
        self.data = [] #List of Item objects
        self.k = k #Parameter

    def __repr__(self, k):
        return str(self.data)

    def insert(self, point):
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
        for x in self.data:
            #See if x is in p's neighborhood
            if p.insert_into_neighborhood(x, self.k):
                x.reverse_knn.append(p)

            #See if p is in x's neighborhood
            if x.insert_into_neighborhood(p, self.k):
                p.reverse_knn.append(x)

    def batch_insert(self, l):
        for x in l:
            self.insert(x)