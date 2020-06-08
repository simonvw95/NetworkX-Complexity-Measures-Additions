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
------------------------------------------------------------------------------------------------
Novel Graph generation functions
"""


"""
random_routing_graph(n, p = None):
Random Routing Graph:
A digraph generator function that generates a random routing digraph (acyclic)

Input:  n, the amount of nodes the routing digraph will have
        p, the probability in which a node is connected to another node
Output: G, the routing digraph
"""


def random_routing_graph(n, p = None):
    
    if p == None:
        p = (2 * np.log(n) / n)
    # generate an empty array of size n x n    
    A = sp.lil_matrix(np.zeros(shape = (n, n), dtype = np.int32))
    # set the sink
    A[0, 1] = 1
    for i in range(0, (n - 1)):
        for j in range((i + 1), n):
            # create an arc based on a certain probability
            if np.random.rand(1) < p:
                A[i, j] = 1
                
    G = nx.DiGraph(A)
    # check which nodes do not connect to the sink
    nodes_list = np.array(G.nodes())[0:-1]

    # loop over the reversed list!
    # this is done to create a trickle effect
    # if node X connects to the sink and if node X-1 is also connected to node X
    # then node X-1 is also connected to the sink
    
    for i in list(reversed(nodes_list)):
        if nx.has_path(G, i, (n - 1)) == False:
            G.add_edge(i, (n - 1))

    return G