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
Node Ranking metrics:
"""


"""
rank_one(G)
Node Ranking metric r1:

Input:  Graph G, a digraph object from networkx
Output: r_dict, a dictionary of length n with the nodes as keys and node rank values as values
"""


def rank_one(G):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")

    # convert to adjacency matrix notation        
    A = nx.to_scipy_sparse_matrix(G, dtype = np.float)
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    
    # get the outdegree matrix D, where diagonal entries are the outdegree
    D = sp.lil_matrix(np.diagflat(A * np.ones((n, 1), dtype = np.float)))
    
    # get the inverse of the matrix D
    #for i in D.nonzero()[0]:
        #D[i, i] = 1 / D[i, i]
        
    # find the markov matrix P
    P = A.T * sp.linalg.inv(D)

    # do eigenvalue decomposition to solve the markov matrix process
    eig, eig_vec = sp.linalg.eigs(P, 3)
    left_vec = eig_vec[:, 0]
    
    # if we get complex numbers then take the absolute value
    if left_vec.dtype == 'complex':
        r = abs(left_vec) / abs(sum(left_vec))
    else:
        r = left_vec / sum(left_vec)

    r_dict = dict(zip(nodes_list, r))
    
    return r_dict


# deprecated, uses non sparse matrices operations
#def rank_one(G):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#
#    # convert to adjacency matrix notation        
#    A = nx.to_numpy_matrix(G, dtype = np.int32)
#    nodes_list = list(G.nodes())
#    n = len(nodes_list)
#    
#    # get the outdegree matrix D, where diagonal entries are the outdegree
#    D = np.diagflat(np.dot(A, np.ones((n, 1), dtype = np.uint32)))
#    
#    # find the markov matrix P where each row i is divided by the outdegree of that row's node
#    P = np.dot(A.T, np.linalg.pinv(D))
#
#    # do eigenvalue decomposition to solve the markov matrix process
#    eig, eig_vec = np.linalg.eig(P)
#    left_vec = eig_vec[:, 0]
#    
#    # if we get complex numbers then take the absolute value
#    if left_vec.dtype == 'complex':
#        r2 = abs(left_vec) / abs(sum(left_vec))
#    else:
#        r2 = left_vec / sum(left_vec)
#
#    r_dict = dict(zip(nodes_list, r))
#    
#    return r_dict


"""
rank_two(G, sigma = 0.85)
Node Ranking metric r2:
    
Input:  Graph G, a digraph object from networkx
        sigma, a control parameter, default is 0.85 same as pagerank default
Output: r_dict, a dictionary of length n with the nodes as keys and node rank values as values
"""


def rank_two(G, sigma = 0.85):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")

    # convert to adjacency matrix notation
    A = nx.to_scipy_sparse_matrix(G, dtype = np.int32)
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    
    # we require the identity matrix for this
    I = sp.csc_matrix(np.diagflat(np.ones(n, dtype = np.int32)))
    
    # now we need the transposed indegree vector, not the matrix
    indegree_vec = (np.ones((n, 1), dtype = np.uint8).T * A).T

    r = sp.linalg.inv(I - sigma * A.T) * indegree_vec
    r_dict = dict(zip(nodes_list, r[:, 0]))
    
    return r_dict


# deprecated, uses numpy matrix multiplication
#def rank_two(G, sigma = 0.85):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#
#    # convert to adjacency matrix notation
#    A = nx.to_numpy_matrix(G, dtype = np.int32)
#    nodes_list = list(G.nodes())
#    n = len(nodes_list)
#    
#    # we require the identity matrix for this
#    I = np.diagflat(np.ones(n, dtype = np.int32))
#    
#    # now we need the transposed indegree vector, not the matrix
#    indegree_vec = np.dot(np.ones((n, 1)).T, A).T
#
#    r2 = np.dot(np.linalg.inv(I - sigma * A.T), indegree_vec)
#    r_dict = dict(zip(nodes_list, r))
#    
#    return r_dict


"""
coreness(G)
Coreness Centrality:
    
Input:  Graph G, a (di)graph object from networkx
Output: r_core, a dictionary of length n with the nodes as keys and node rank values as values
"""


def coreness(G):

    # first do k-shell decomposition
    core_numbers = nx.core_number(G)
    
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    
    # initialize result list
    r_res = np.zeros(shape = (n, 1), dtype = np.uint32)

    # loop over all the nodes
    for i in range(n):
        coreness_sums = 0
        current_node = nodes_list[i]
        
        # get the neigbhors of the current node
        neighbors_node = list(G.neighbors(current_node))
        
        # loop over the neighbors
        for neighbor in neighbors_node:
            # extract the coreness value of each neighbor, this loop is more robust as nodes can have names now
            coreness_sums += core_numbers.get(neighbor)

        r_res[i] = coreness_sums

    r_core = dict(zip(nodes_list, r_res[:, 0]))
    
    return r_core


"""
coreness_plus(G, coreness_dict = None):
Coreness Plus Centrality:
    
Input:  Graph G, a (di)graph object from networkx
        coreness_dict, a dictionary containing nodes of the graph G as keys and coreness values as values, OPTIONAL
Output: r_dict, a dictionary of length n with the nodes as keys and node rank values as values
"""


def coreness_plus(G, coreness_dict = None):

    # if coreness is not supplied then run coreness centrality
    if coreness_dict == None:
        coreness_dict = coreness(G)
    
    nodes_list = list(G.nodes())
    n = len(list(nodes_list))
    
    # initialize result list
    r_res = np.zeros(shape = (n, 1), dtype = np.uint32)

    # loop over all the nodes
    for i in range(n):
        coreness_plus_sums = 0
        current_node = nodes_list[i]
        
        # get the neighbors of the current node
        neighbors_node = list(G.neighbors(current_node))
        
        # loop over the neighbors
        for neighbor in neighbors_node:
            # extract the coreness centrality value of each neighbor
            coreness_plus_sums += coreness_dict.get(neighbor)
            
        r_res[i] = coreness_plus_sums

    r_core_plus = dict(zip(nodes_list, r_res[:, 0]))
    
    return r_core_plus

	
"""
SIR(G, node)
SIR:
Single node influence measuring using SIR
Using the Susceptible Influenced Recovered (SIR) model, calculate the influence that
each node has on the rest of the network. These node influence values can, arguably,
be used as the node rank 'ground truth'.
Based on: https://www.sciencedirect.com/science/article/pii/S0378437113010406?via%3Dihub
Note that the below function does not include multiple simulations yet

Input:  Graph G, a (di)graph object from networkx
        node, a target node for which you want to calculate the influence
Output: influence, an integer representing how many other nodes the original node infected
"""


def SIR(G, node):
    
    # make a copy of the graph G so we can mutate it
    G_copy = G.copy()
    
    # set mutation probability equal to a number slightly larger than degree / degree^2
    degree = int(G_copy.number_of_edges() * 2 / G_copy.number_of_nodes())
    beta = round((degree / (degree ** 2) + 0.03), ndigits = 2)
    
    # set the infected
    dtype_node = type(node)
    infected = np.array([node])
    recovered = np.array([], dtype = dtype_node)
    
    # keep running the simulation until there are no more infections
    while len(infected) != 0:
        new_infected = np.array([], dtype = dtype_node)
        
        # loop over all infected 
        for i in infected:
            # get neighbors
            N = np.array(list(G_copy.neighbors(i)))
            
            # infect neighbors based on probability beta
            
            # create a list of N random numbers between 0 and 1
            rand_numbers = np.random.rand(len(N))
            # add newly infected to a list
            new_infected = np.append(new_infected, N[rand_numbers < beta])
            # remove the initial node / set to recovered
            G_copy.remove_node(i)
            recovered = np.append(recovered, i)
        
        # make sure that the removed nodes aren't in the new infected nodes
        infected = np.setdiff1d(np.unique(new_infected), recovered)
        
    influence = len(recovered)
    
    return influence

	
"""
full_graph_influence_mc(G, MC)
Full graph influence:
Single node influence measuring using SIR
Using the Susceptible Influenced Recovered (SIR) model, calculate the influence that
each node has on the rest of the network. These node influence values can, arguably,
be used as the node rank 'ground truth'.
Based on: https://www.sciencedirect.com/science/article/pii/S0378437113010406?via%3Dihub
Note that the below function does not include multiple simulations yet

Input:  Graph G, a (di)graph object from networkx
        MC, the amount of times you want to simulate the infection events
Output: node_influence_dict, a dictionary with all the nodes and their float influence values
"""    


def full_graph_influence_mc(G, MC):
    
    # get a list of all the nodes in G
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    
    # initialize result list
    influence_list = np.zeros(n, dtype = np.float32)
    
    # loop over all nodes
    for i in range(n):
        if i % 10 == 0:
            print("I'm at node: ", i, ", only ", n - i, " more nodes to go!")
        
        # replicate the simulation for a single node MC times
        result_array = np.zeros(MC, dtype = np.uint32)
        G_main = G.copy()
        
        # MC simulation
        for replication in range(MC):
            result_array[replication] = SIR(G_main, nodes_list[i])
        
        influence_list[i] = np.mean(result_array)
        
    node_influence_dict = dict(zip(nodes_list, influence_list))
    
    return node_influence_dict

    
"""
kendall_tau(R1, R2)
Kendall's Tau:
Note: This is not a centrality measure but a correlation metric, it can be used to compare node ranking metrics
and judge how similar they are in terms of agreements. This is mostly used to compare a node ranking metric with the ground truth
though it is also possible to compare node ranking metrics directly
Based on: https://www.sciencedirect.com/science/article/pii/S0378437113010406?via%3Dihub

Input:  R1, a list or numpy array containing the ranks of all nodes
        R2, a list or numpy array containing the ranks of all nodes, must be of equal size of R1 and order of nodes must be the same
Output: tau, a singular value between -1 and 1 where 1 is perfect agreement, -1 perfect disagreement and values
        around 0 mean that the two metrics are not dependent on each other
"""


def kendall_tau(R1, R2):

    n = len(R1)
    n_t = n * (n - 1) / 2
    
    if n != len(R2):
        raise InputError("Ranking vectors are not of equal size!")

    # calculate number of concordant pairs: number of pairs where both metrics agree on direction
    # e.g.: R1.i > R1.j and R2.i > R2.j; OR R1.i < R1.j and R2.i < R2.j
    # calculate number of discordant pairs: number of pairs where both metrics disagree on direction
    # e.g.: R1.i > R1.j and R2.i < R2.j; OR R1.i < R1.j and R2.i > R2.j
    # calculate number of equal pairs? disregard for now
    # e.g.: R1.i = R2.j or R2.i = R2.j
    n_c = 0
    n_d = 0
    t_1_list = np.zeros(n, dtype = np.float32)
    t_2_list = np.zeros(n, dtype = np.float32)
    
    for i in range(n):
        t_1 = 0
        t_2 = 0
        for j in range(i + 1, n):
            # first concordant pairs
            if (R1[i] > R1[j] and R2[i] > R2[j]) or (R1[i] < R1[j] and R2[i] < R2[j]):
                n_c += 1
            # then discordant pairs
            elif (R1[i] > R1[j] and R2[i] < R2[j]) or (R1[i] < R1[j] and R2[i] > R2[j]):
                n_d += 1
            elif R1[i] == R1[j]:
                t_1 += 1
            elif R2[i] == R2[j]:
                t_2 += 1
        t_1_list[i] = t_1 * (t_1 - 1) / 2
        t_2_list[i] = t_2 * (t_2 - 1) / 2

    n_t_1 = sum(t_1_list)
    n_t_2 = sum(t_2_list)
    tau = (n_c - n_d) / np.sqrt((n_t - n_t_1) * (n_t - n_t_2))
    
    return tau