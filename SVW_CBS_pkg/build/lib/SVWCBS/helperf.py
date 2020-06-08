# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:11:20 2020

@author: Simon
"""

import networkx as nx
import numpy as np
import time
import itertools
import pickle
from scipy import sparse as sp

# custom exceptions
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class GraphError(Error):
    """Exception raised for errors in the input related to graphs.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

		
"""
Functions for complexity measures and node ranking methods used in the thesis of Simon van Wageningen during
the internship at Statistics Netherlands

-----------------------------------------------------------------------------------------------------------------------

For all functions that use the adjacency matrix:
    the adjacency matrix has to be in the format where the rows are the sources and the columns are the destinations
"""  


"""            
---------------------------------------------------------------------------------------------------------------------------------------
Other useful but not necessarily needed functions
"""


"""
run_time(f, G)
Timing Function:
A timing function that returns the amount of time it took to run a particular function, useful
for making comparisons

Input:  f, a function f with its own function inputs
        G, graph object for the input of f
Output: ret_val, the returning values from the input function
        tot_time, the time it took to run function f
"""


def run_time(f, G):
    
    start = time.time()
    ret_val = f(G = G)
    end = time.time()
    
    tot_time = end - start
    
    return ret_val, tot_time


"""
open_pickle(path)
Open Pickle Function:
A function that opens a pickle file

Input:  path, the path to the pickle file including the pickle name, in string format
Output: obj, the objects stored in the pickle
"""


def open_pickle(path):

    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    
    return obj


"""
save_pickle(path, objects):
Save Pickle Function:
A function that opens a pickle file

Input:  path, the path to the pickle file including the pickle name, in string format
        objects, the objects to be stored in the pickle
"""


def save_pickle(path, objects):
    
    f = open(path, 'wb')
    pickle.dump(objects, f)
    f.close()