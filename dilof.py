"""
Python implementation of Density-based Incremental Local Outlier Factor (DILOF) Outlier Detection algorithm
"""

from cs543FinalProjectOutliers.item import Item

class DILOF:
    def __init__(self,window_size, threshold, step_size, regularizer, max_iter):
        self.data = [] #List of Item objects
        self.outliers = []

        self.window_size = window_size #Maximum window size
        self.threshold = threshold #Threshold for outlier scores
        self.step_size = step_size
        self.regularizer = regularizer
        self.max_iter = max_iter

    def insert(self, tuple):
        p = Item(tuple)
        #Use LOD to compute LOF of p
         
    
    def lod(self, item):
        
