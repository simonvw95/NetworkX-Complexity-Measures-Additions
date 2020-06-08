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
Functions and helper functions for full graph complexity measures

"""


"""
warshall(G)
Warshall's algorithm:
Used to compute the transitive closure matrix A* from the adjacency matrix A, which
can then be used to check whether the matrix A is strongly connected
http://www.csl.mtu.edu/cs4321/www/Lectures/Lecture%2016%20-%20Warshall%20and%20Floyd%20Algorithms.htm

Input:  Graph G, a (di)graph object from networkx
Output: A*, the transitive closure matrix,
        note that its elements are in uint8 format, beware with matrix operations
        check, a boolean variable, True if the transitive closure matrix is the same as all ones matrix (fully reachable)
"""


def warshall(G):
    
    # convert to adjacency matrix notation
    A = sp.lil_matrix(nx.to_scipy_sparse_matrix(G, dtype = np.uint8))

    n = A.shape[0]
    
    # copy the matrix since we're making changes
    A_copy = A.copy()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                A_copy[j, k] = A_copy[j, k] or (A_copy[j, i] and A_copy[i, k])

    A_star = A_copy
    
    # computing grandsum is faster than comparing 2 large arrays
    check = (np.sum(A_star) == (n * n))

    return A_star, check

# deprecated, uses numpy matrix notation
#def warshall(G):
#    
#    # convert to adjacency matrix notation
#    A = nx.to_numpy_array(G, dtype = np.uint8)
#
#    n = len(A)
#    
#    # copy the matrix since we're making changes
#    A_copy = np.copy(A)
#
#    for i in range(n):
#        for j in range(n):
#            for k in range(n):
#                A_copy[j][k] = A_copy[j][k] or (A_copy[j][i] and A_copy[i][k])
#
#    A_star = A_copy
#    
#    # computing grandsum is faster than comparing 2 large arrays
#    check = (np.sum(A_star) == (n * n))
#
#    return A_star, check


"""
trans_close(G)
Transitive Close:
Used as an alternative to Warshall's algorithm, computes the transitive closure matrix
in a much faster way

Input:  Graph G, a digraph object from networkx
Output: A*, the transitive closure matrix in scipy sparse format
        note that its elements are in uint8 format, beware with matrix operations

"""


def trans_close(G):
    
    #trans_closure_G = nx.transitive_closure(G, False) # for newer versions of networkx
    trans_closure_G = nx.transitive_closure(G)
    A_star = nx.to_scipy_sparse_matrix(trans_closure_G, dtype = np.uint8)
    
    return A_star

	
"""
strongly_connected_check(G)
Strongly connected check:
Checks whether a digraph G or adjacency matrix A is fully reachable/strongly connected

Input:  Graph G, a digraph object from networkx
Output: check, a boolean variable, True if the network is fully reachable/strongly connected
"""


def strongly_connected_check(G):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")
    
    check = nx.is_strongly_connected(G)
    
    return check

	
"""
average_degree(G)
Average degree complexity:
Graph complexity measure, it simply illustrates the average degree over all nodes in the graph

Input:  Graph G, a (di)graph object from networkx
Output: C_deg, a single float value representing the average degree of the graph G
"""


def average_degree(G):
    
    m = len(G.edges())
    n = len(G.nodes())

    C_deg = 2 * m / n
    
    return C_deg


"""
routing(G)
Routing complexity:
Routing complexity measure, it measures the amount of paths from source node to sink node in
ACYCLIC DIRECTED graphs, this is a requirement otherwise the function may not find a solution

Input:  Graph G, an acyclic digraph object from networkx
Output: C_rou, a single float value representing the routing complexity of the digraph G
"""


def routing(G):

    A = nx.to_scipy_sparse_matrix(G, dtype = np.int32)

    # check if digraph is acyclic, the lower triangle of A (if in the correct format) should be all 0s
    if nx.is_directed_acyclic_graph(G) == False:
        raise GraphError("Given graph is not acyclic!")
        
    n = A.shape[0]
    
    # we require the identity matrix for this
    I = sp.csr_matrix(np.diagflat(np.ones(n, dtype = np.int32)))
    
    # round early to account for summation inaccuracies
    C_rou = np.log(round(np.linalg.inv((I - A).todense())[0, n-1], ndigits = 2))

    return round(C_rou, ndigits = 2)

# deprecated, older method without sparse matrices
#def routing(G):
#
#    A = nx.to_numpy_array(G, dtype = np.int32)
#
#    # check if digraph is acyclic, the lower triangle of A (if in the correct format) should be all 0s
#    if np.sum(np.tril(A, k = 0)) != 0:
#        raise GraphError("Given graph is not acyclic!")
#        
#    n = len(A)
#    
#    # we require the identity matrix for this
#    I = np.diagflat(np.ones(n, dtype = np.int32))
#    
#    C_rou = np.log(np.linalg.pinv(I - A)[0, n-1])
#        
#    return C_rou


"""
arc_symmetry(G)
Arc symmetry:
Arc Symmetry complexity measure, it quantifies how many arcs do not have a counter-arc
DIRECTED graphs are required since we're dealing with arcs

Input:  Graph G, a digraph object from networkx
Output: C_sym, a single float value representing the arc symmetry complexity of the digraph G
"""


def arc_symmetry(G):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")
        
    edge_list = G.edges()
    cnt_symm = 0
    for e in edge_list:
        reverse_edge = tuple(reversed(e))

        # check whether the reverse edge exists
        if G.has_edge(*reverse_edge) == True:
            cnt_symm += 1
    
    # subtract the amount of edges that have a reverse edge from m
    # then follow the same matrix calculation        
    C_sym = round(1 - (cnt_symm / G.number_of_edges()), ndigits = 4)
        
    return C_sym

# deprecated, new better calculation
#def arc_symmetry(G):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#        
#    edge_list = G.edges()
#    n = G.number_of_nodes()
#    cnt_symm = 0
#    for e in edge_list:
#        reverse_edge = tuple(reversed(e))
#
#        # check whether the reverse edge exists
#        if G.has_edge(*reverse_edge) == True:
#            cnt_symm += 1
#    
#    # subtract the amount of edges that have a reverse edge from m
#    # then follow the same matrix calculation        
#    C_sym = (G.number_of_edges() - cnt_symm) * 2 / (n * (n - 1))
#        
#    return C_sym

# deprecated, older uses sparse matrix lookups
#def arc_symmetry(G):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#        
#    A = nx.to_scipy_sparse_matrix(G, dtype = np.uint8)
#    nonzeros = A.nonzero()
#    n = A.shape[0]
#    
#    
#    n_nonzeros = len(nonzeros[0])
#    cnt_symm = 0
#    for i in range(n_nonzeros):
#        index = (nonzeros[0][i], nonzeros[1][i])
#        for j in range(i, n_nonzeros):
#            reverse_index = tuple(reversed(index))
#            if nonzeros[0][j] == reverse_index[0] and nonzeros[1][j] == reverse_index[1]:
#                cnt_symm += 1
#                
#    # multiply cnt_symm by 2, now we have the amount of arcs have a counter arc
#    cnt_symm *= 2
#    
#    C_sym = (G.number_of_edges() - cnt_symm) * 2 / (n * (n - 1))

# deprecated, oldest uses matrix multiplication
#def arc_symmetry(G):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#        
#    A = nx.to_numpy_array(G, dtype = np.int32)
#    theta_mat = np.copy(A)
#    n = len(theta_mat)
#    e = np.ones((n, 1), dtype = np.int8)
#    
#    for i in range(n):
#        for j in range(n):
#            theta_mat[i, j] = max(A[i, j], A[j, i]) - min(A[i, j], A[j, i])
#
#    C_sym = (np.dot(e.T, np.dot(theta_mat, e))) / (n * (n - 1))
#    
#    return C_sym[0][0]


"""
entropy(G)
Entropy:
Entropy complexity measure, attempts to measure how far a digraph is removed from full reachability
The lower bound is 0, then the digraph is fully reachable or there are isolated nodes.

Input:  Graph G, a digraph object from networkx
Output: C_e, a single float value representing how far the digraph G is removed from full reachability,
        small values indicate that the graph is closer to full reachability than large values
"""


def entropy(G):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")

    # calculate transitive closure matrix A*
    A_star = trans_close(G = G)
    n = A_star.shape[0]
    e = sp.csr_matrix(np.ones((n, 1), dtype = np.int32)) # column vector
    
    delta = sp.lil_matrix((e.T * A_star) / ((e.T * A_star) * e)[0, 0])
    delta_sums = 0
    
    for i in range(n):
        # if the value is 0 then it will throw a log warning, circumvent this
        if delta[0, i] > 0:
            delta_sums += delta[0, i] * np.log(delta[0, i])
        else:
            delta_sums += delta[0, i] * delta[0, i]

    C_e = -delta_sums
        
    return 1 - round(C_e / (np.log(n)), ndigits = 4)

# deprecated, uses nonsparse matrix multiplications
#def entropy(G):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#
#    # calculate transitive closure matrix A*
#    A_star = trans_close(G = G)
#    n = len(A_star)
#    e = np.ones((n, 1), dtype = np.uint8) # column vector
#    
#    delta = (np.dot(e.T, A_star)) / (np.dot(np.dot(e.T, A_star), e))
#    delta_sums = 0
#    
#    for i in range(n):
#        # if the value is 0 then it will throw a log warning, circumvent this
#        if delta[0][i] > 0:
#            delta_sums += delta[0][i] * np.log(delta[0][i])
#        else:
#            delta_sums += delta[0][i] * delta[0][i]
#
#    C_e = -delta_sums
#        
#    return C_e


"""
Deterministic reachability
Deterministic reachability complexity measure, attempts to measure how far a digraph is removed from full reachability in a deterministic manner

Input:  Graph G, a digraph object from networkx
Output: C_det, a single float value representing how far the digraph G is removed from full reachability,
        small values indicate that the graph is closer to full reachability than large values
"""

# work in progress

def deter_reach(v_min, m):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")
        
    # get all the strongly connected components
    sccs = list(sorted(nx.strongly_connected_components(G), key = len, reverse = False))
    n_sccs = len(sccs)
    sccs_sub = [0] * n_sccs
    
    # create subgraphs for each scc
    for i in range(n_sccs):
        sccs_sub[i] = G.subgraph(sccs[i])
        
    # for each subgraph try to connect it to the biggest scc
    for i in sccs_sub:
        # if the scc only has one node, it always needs a reverse arc
        if i.number_of_nodes() == 1:
            node_i = list(i.nodes())[0]
            pred, succ = list(G.predecessors(node_i)), list(G.neighbors(node_i))
    C_det = v_min / m
    
    return C_det    
    

# outdated probabilistic reachability, found a better method
    
#"""
#Probabilistic reachability
#Probabilistic reachability complexity measure, attempts to measure how far a digraph is removed
#from full reachability in a probabilistic manner
#
#Input:  Graph G, a digraph object from networkx
#        p, starting probability with a starting value of 0.5
#        exclude, a boolean variable that indicates whether nodes with an in or outdegree of 0
#        should be exluded from the simulations (these always need to receive a counter-arc
#        and are thus not interesting to simulate)
#        MC, the amount of simulations performed
#Output: C_prob, a single float value representing the minimum probability for which every digraph
#        attains full reachability 
#        
#"""
#
#def prob_reach(G, p = 0.5, exclude = True, MC = 100):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#        
#    nodes_list = list(G.nodes())
#    type_nodes = type(nodes_list[0])
#    edge_list = list(G.edges())
#    
#    # find nodes that have an in or outdegree of 0 AND only one in or outcoming arc
#    if exclude == True:
#        # find the nodes with indegree 0
#        indegree = np.array(list(dict(G.in_degree()).values()), dtype = np.uint32)
#        outdegree = np.array(list(dict(G.out_degree()).values()), dtype = np.uint32)
#        
#         # find the nodes with indegree 0 and outdegree 1
#        indegree_0_indices = np.argwhere((indegree == 0) & (outdegree == 1))
#        indegree_0_nodes = np.array(nodes_list, dtype = type_nodes)[indegree_0_indices]
#       
#        # find the nodes with outdegree 0 and indegree 1
#        outdegree_0_indices = np.argwhere((outdegree == 0) & (indegree == 1))
#        outdegree_0_nodes = np.array(nodes_list, dtype = type_nodes)[outdegree_0_indices]
#        
#        # combine these node arrays
#        useless_nodes = np.union1d(indegree_0_nodes, outdegree_0_nodes)
#        
#        # create a new graph without the useless nodes
#        G_copy = G.copy()
#        G_copy.remove_nodes_from(useless_nodes)
#        
#        # update edge_list
#        edge_list = list(G_copy.edges())
#        
#    # find edges that do not have a reversed arc
#    
#    # use lists as appending to python lists is faster than appending to numpy arrays
#    unsaturated_pairs = []
#    for e in edge_list:
#        reverse_edge = tuple(reversed(e))
#        
#        # check whether the reverse edge exists
#        if G.has_edge(*reverse_edge) == False:
#            unsaturated_pairs.append(e)
#     
#    # now we can do the experiments
#    # continue running the experiments until we have a probability that (almost) always reaches full probability   
#    full_reach_fraction = 0.0       
#    
#    while full_reach_fraction < 0.95:
#
#        print(round(p, ndigits = 2))
#        # run 100 simulations per probability
#        reach_count = 0
#        
#        for i in range(MC):
#            if exclude == True:
#                G_aug = G_copy.copy()
#            else:
#                G_aug = G.copy()
#                
#            # loop over the unsaturated pairs and connect them with probability p
#            for e in unsaturated_pairs:
#                # add an edge with probability p
#                if p > np.random.rand(1):
#                    G_aug.add_edge(e[1], e[0])
#                    
#            check = strongly_connected_check(G = G_aug)
#            
#            if check == True:
#                reach_count += 1
#        
#        # get the fraction of how many times full reachability was acquired
#        full_reach_fraction = reach_count / MC
#        p += 0.01
#        
#    C_prob = round(p, ndigits = 2)
#    
#    return(C_prob)

# deprecated, faster solutions can be found working with the edge list for large graphs
     
#def prob_reach(G, p = 0.5, exclude = True):
#    
#    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
#        raise GraphError("Given graph is not directed")
#    
#    nodes_list = list(G.nodes())
#    n = len(nodes_list)
#    
#    # convert to adjacency matrix notation
#    A = nx.to_numpy_array(G, dtype = np.uint8)
#    
#    # get the in and outdegree if we want to exclude useless nodes
#    if exclude == True:
#        indegree = list(dict(G.in_degree()).values())
#        outdegree = list(dict(G.out_degree()).values())
#        
#    # find unsaturated pairs, pairs with an arc but no counter-arc
#    unsaturated_pairs = []
#    irrelevant_edges = []
#    
#    for i in range(n):
#        
#        # find the unsaturated pairs that we ALWAYS have to saturate to get full reachability
#        if exclude == True:
#            if indegree[i] == 0 and outdegree[i] == 1:
#                irrelevant_edges.append(([list(G.successors(nodes_list[i]))[0], i]))
#            if outdegree[i] == 0 and indegree[i] == 1:
#                irrelevant_edges.append(([i, list(G.predecessors(nodes_list[i]))[0]]))
#        
#        # find all unsaturated pairs
#        for j in range(i, n):
#            if A[i][j] != A[j][i]:
#                # unsaturated pair found
#                if A[i][j] == 0:
#                    unsaturated_pairs.append(([i, j]))
#                else:
#                    unsaturated_pairs.append(([j, i]))
#         
#    # remove irrelevant unsaturated pairs from unsaturated pairs and from graph
#    if exclude == True:
#        G_copy = G.copy()
#        
#        for i in irrelevant_edges:
#            unsaturated_pairs.remove(i)
#            G_copy.remove_edge(i[1], i[0])
#            
#        G_copy.remove_nodes_from(list(nx.isolates(G_copy)))
#        
#    # continue running the experiments until we have a probability that (almost) always reaches full probability   
#    full_reach_fraction = 0.0       
#    
#    while full_reach_fraction < 0.95:
#
#        print(round(p, ndigits = 2))
#        # run 100 simulations per probability
#        reach_count = 0
#        
#        for i in range(100):
#            if exclude == True:
#                G_aug = G_copy.copy()
#            else:
#                G_aug = G.copy()
#                
#            # loop over the unsaturated pairs and connect them with probability p
#            for j in range(len(unsaturated_pairs)):
#                # add an edge with probability p
#                if p > np.random.rand(1):
#                    G_aug.add_edge(unsaturated_pairs[j][0], unsaturated_pairs[j][1])
#                    
#            check = strongly_connected_check(G = G_aug)
#            
#            if check == True:
#                reach_count += 1
#        
#        # get the fraction of how many times full reachability was acquired
#        full_reach_fraction = reach_count / 100
#        p += 0.01
#        
#    C_prob = p
#    
#    return(C_prob)
#        


"""
prob_reach_two(G, p = 0.3, step_size = 0.01, max_iter = 1000, precision = 0.0001, MC = 100)
Probabilistic reachability 2:
Probabilistic reachability complexity measure, attempts to measure how far a digraph is removed
from full reachability in a probabilistic manner

Input:  Graph G, a digraph object from networkx
        p, starting probability with a starting value of 0.3
        step_size, the size of the steps between probabilities
        max_iter, the maximum amount of iterations the algorithm will perform set to 1000
        precision, the algorithm will terminate if the step taken is smaller than the precision
        MC, the amount of simulations performed
Output: C_prob, a single float value representing the probability where a single step does 	not improve the reachability by more than the precision
"""


def prob_reach_two(G, p = 0.3, step_size = 0.01, max_iter = 1000, precision = 0.0001, MC = 100):
    
    if type(G) != nx.DiGraph and type(G) != nx.MultiDiGraph:
        raise GraphError("Given graph is not directed")
        
    edge_list = list(G.edges())
    n = G.number_of_nodes()
    
    # find edges that do not have a reversed arc
    
    # use lists as appending to python lists is faster than appending to numpy arrays
    unsaturated_pairs = []
    for e in edge_list:
        reverse_edge = tuple(reversed(e))
        
        # check whether the reverse edge exists
        if G.has_edge(*reverse_edge) == False:
            unsaturated_pairs.append(reverse_edge)
     
    # now we can do the experiments
    # continue running the experiments until we have a probability that (almost) always reaches full probability    
    prev_proportion = len(max(nx.strongly_connected_components(G), key = len)) / n
    current_iter = 0
    while current_iter < max_iter:
        
        # run MC simulations per probability
        reach_proportion = np.zeros(MC, dtype = np.float32)
        
        for i in range(MC):
            G_aug = G.copy()
                
            # loop over the unsaturated pairs and connect them with probability p
            for e in unsaturated_pairs:
                # add an edge with probability p
                if p > np.random.rand(1):
                    G_aug.add_edge(*e)
                    
            # check the size of the largest connected component and compare
            size_scc_aug = len(max(nx.strongly_connected_components(G_aug), key = len))
            reach_proportion[i] = size_scc_aug / n
        
        
        # get the average of the reach proportion
        current_proportion = np.mean(reach_proportion)
        
        #print("Probability ", round(p, ndigits = 2), " has an average reachability of ", current_proportion)
        
        step = abs(current_proportion - prev_proportion)
        if step < precision:
            break
        
        if p != 1.00:
            prev_proportion = current_proportion
            p += step_size
            current_iter += 1
        else:
            break
        
    C_prob = round(p, ndigits = 2)
    
    return C_prob


    
    
"""
average_distance(G)
Average Distance:
Average Distance complexity measure, calculates the average distance between all nodes in the connected graph
This metric is already defined by networkx

Input:  Graph G, a (di)graph object from networkx
Output: C_d, a single float value representing the average distance/shortest path length between nodes
"""

def average_distance(G):
              
    C_d = nx.average_shortest_path_length(G = G)
    
    return C_d
        


"""
route_search(G)
Route search:
Route search complexity measure, calculates the ratio between a path that visits every edge
and a tour that visits every edge and returns to the same node.
Closely related to the chinese postman problem
algorithm inspired by: http://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/
Input:  Graph G, a graph object from networkx
Output: C_search
"""

def route_search(G):
    
    m = G.number_of_edges()
    
    # find odd nodes
    degree = np.array(list((dict(G.degree()).values())))
    
    odd_nodes = np.argwhere((degree % 2) == 1)
    odd_nodes = odd_nodes.flatten()
    odd_nodes = np.array(G.nodes())[odd_nodes] # this step is necessary for if nodes have names
    odd_node_pairs_list = list(itertools.combinations(odd_nodes, 2))
    
    # get the shortest paths between all odd node pairs
    # store these paths in a new graph with weight 1/length path
    path_lengths = {}
    G_temp = nx.Graph()
    for i in odd_node_pairs_list:
        path_lengths[i] = nx.dijkstra_path_length(G, i[0], i[1])
        G_temp.add_edge(i[0], i[1], weight = 1 / path_lengths[i])
        
    # do min weight matching of the edges in the temp graph to find the best edges to add
    best_edges = list(nx.max_weight_matching(G_temp))
    
    # loop over all best edges and add their weights 
    best_edges_dict = {}
    for i in best_edges:
        best_edges_dict[i] = G_temp.get_edge_data(*i)['weight']
        
    # sort the best edges based on weight, highest weight is most important
    best_edges_sorted = sorted(best_edges_dict.items(), key = lambda x: x[1], reverse = True)
    
    # for a semi-eulerian graph, one with an euler path and NOT an euler cycle
    # we only add all best edges except the worst one
    # for an eulerian graph that has an euler cycle
    # we add all best edges
    
    tau_G = m
    lambda_G = m
    
    cnt = 0
    # loop over all best edges and get their path lengths
    for i in best_edges_sorted:
        
        current_path_length = path_lengths.get(i[0])
        
        # if the length of the path is none then the node pair has to be in reverse order to get the length
        if current_path_length == None:
            current_path_length = path_lengths.get(i[0][::-1])
        
        # add all path lengths as the amount of edges needed
        
        # add all path lengths except the last (worst) for the euler path
        if cnt < len(best_edges_sorted) - 1:
            lambda_G += current_path_length
            
        tau_G += current_path_length
        
        cnt += 1
    
    C_search = lambda_G / tau_G
    
    return C_search


# deprecated, incorrect calculations

#def route_search(G):
#         
#    if type(G) != nx.Graph:
#        raise AttributeError("Given graph is not undirected")
#
#    # if it contains a euler cycle then the metric is equal to 1
#    # euler cycle is present when every degree is even
#    degree = np.array(list((dict(G.degree()).values())))
#    odd_freq = int(np.sum(degree % 2))
#    
#    if odd_freq == 0:
#        C_search = 1
#    else:
#        # turn the graph into an euler graph that contains an euler cycle
#        # (it adds parallel edges similar to retracing edges)
#        G_eul = nx.eulerize(G)
#        tau_G = len(list(nx.eulerian_circuit(G_eul)))
#        
#        # if there are 2 nodes with an odd frequency then there is an euler path
#        if odd_freq == 2:
#            lambda_G = len(list(nx.eulerian_path(G)))
#        else:
#            # the length of the optimal path is equal to total amount of edges
#            # plus the amount of times that you have to retrace an additional edge
#            # this amount is expressed by max(0, #odd nodes/2 - 1)
#            lambda_G = G.number_of_edges() + max(0, ((odd_freq / 2) - 1))
#
#        C_search = lambda_G / tau_G
#        
#    return C_search
#    
##        odd_nodes = np.argwhere((degree % 2) == 1)
##        odd_nodes = odd_nodes.flatten()
##        odd_nodes = np.array(G.nodes())[odd_nodes] # this step is necessary for if nodes have names
##        odd_node_pairs_list = list(itertools.combinations(odd_nodes, 2))
##        
##        
##        # get the shortest paths between all pairs
##        path_lengths = {}
##        for i in odd_node_pairs_list:
##            path_lengths[i] = nx.dijkstra_path_length(G, i[0], i[1])
##            
##        # add all paths between pairs as new edges to a new graph
##        G_aug = nx.Graph()
##        for key, value in path_lengths.items():
##            length = value
##            G_aug.add_edge(key[0], key[1], weight = length)
##            
##            
##        # find odd degree pairs with lowest combined path length/weight
##        best_odd_pairs = list(nx.algorithms.max_weight_matching(G_aug, True))
##        
##        # add these edges to a new multigraph that allows self loops and parallel edges
##        G_new = nx.MultiGraph(G.copy())
##        weight_sum = 0
##        for i in best_odd_pairs:
##            G_new.add_edge(i[0], i[1], weight = path_lengths.get((i[0], i[1])))
##            weight_sum += path_lengths.get((i[0], i[1])) - 1
##        
##        # find the eulerian cycle length
##        tau2 = len(list(nx.eulerian_circuit(G_new, ))) + weight_sum

