"""
File: IncrementalLOF.py
Author: Aidan Clark
Updated: 4/10/2025
Description: An implementation of the Incremental Local Outlier Factor algorithm, as described in
this paper: https://ieeexplore.ieee.org/document/4221341
"""

from item import Item

import matplotlib.pyplot as plt

import io
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

class IncrementalLOF():
    """
    A class for running the Incremental LOF algorithm.
    We follow a streaming model.
    """
    def __init__(self, k, threshold = 1, print_outliers = False):
        """
        k is the number of nearest neighbors to consider in our calculations.
        This can be tuned for the dataset.

        We store a list of Item objects representing all of the items that have been received so far.
        """
        self.data = [] #List of Item objects
        self.k = k
        self.threshold = threshold
        self.outliers = []
        self.print_outliers = print_outliers #Prints outliers upon detection

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

        Returns True if the item is deemed an outlier and False otherwise
        """
        outlier = False

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

        #Make a decision about whether p is an outlier
        if p.lof > self.threshold:
            self.outliers.append(p)
            outlier = True
            if self.print_outliers:
                print("Outlier Detected:", p.tuple, "with a LOF of", p.lof)

        #Insert p
        self.data.append(p)

        return (outlier, p.lof)



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

    def pretty_picture(self):
        """
        Generates a matplotlib scatterplot of the current state of the algo
        It shows "regular" points in black and points deemed as outliers in red
        Only works for 2D datasets (I suppose I could extend this to work for 3D datasets too...)
        """
        if len(self.data[0].tuple) != 2:
            raise Exception("Dimension of data won't work with pretty_picture")
        
        x = [i.tuple[0] for i in self.data]
        y = [i.tuple[1] for i in self.data]

        x_o = [i.tuple[0] for i in self.outliers]
        y_o = [i.tuple[1] for i in self.outliers]

        plt.scatter(x = x, y = y, color = "black")
        plt.scatter(x = x_o, y = y_o, color = "red", label = "Outliers")
        plt.legend()
        

    def batch_insert_with_gif(self, l, gif_name):
        """
        Takes l, a list of tuples to insert, and a gif_name and generates a gif
        called gif_name that shows the state of the algorithm after each insert

        The GIF creation code was taken from: 
        https://safjan.com/how-to-create-animated-gif-from-matplotlib-plot-in-python/

        This method is useful for testing and for illustrating how our algorithms work.
        """
        # Create the animation
        fig = plt.figure(figsize=(10, 6))

        frames = []

        for x in l:
            plt.clf()
            self.insert(x)
            self.pretty_picture()
            buf = io.BytesIO()
            plt.savefig(buf, format = "png")
            buf.seek(0)
            frames.append(Image.open(buf))
            

        # Create and save the animated GIF
        frames[0].save(
            gif_name,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

    def batch_insert_gif_lof_scale(self, l, gif_name):
        """
        Does the same thing as above, but colors each point based on lof value
        These lof's may not be the same as they were when the outlier decision was made!!
        Note that past points, while their lof's may change, do not get rejudged on outlier or not
        """

        # Create the animation
        fig = plt.figure(figsize=(10, 6))

        frames = []

        for x in l:
            plt.clf()
            self.insert(x)

            x_i = [i.tuple[0] for i in self.data if i not in self.outliers]
            y_i = [i.tuple[1] for i in self.data if i not in self.outliers]
            lof_i = [i.lof for i in self.data if i not in self.outliers]

            x_o = [i.tuple[0] for i in self.outliers]
            y_o = [i.tuple[1] for i in self.outliers]
            lof_o = [i.lof for i in self.outliers]

            global_min = 0
            global_max = 4

            plt.scatter(x = x_i, y = y_i, vmin = global_min, vmax = global_max, c = lof_i, cmap = 'viridis')
            plt.scatter(x = x_o, y = y_o, vmin = global_min, vmax = global_max, c = lof_o, cmap = 'viridis', marker = "*", label = "Outlier")
            
            plt.colorbar(label='LOF')
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format = "png")
            buf.seek(0)
            frames.append(Image.open(buf))
        

        # Create and save the animated GIF
        frames[0].save(
            gif_name,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

        