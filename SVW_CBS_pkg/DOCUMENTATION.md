from SVWCBS import complf as cmp # import this module to use graph complexity measures

from SVWCBS import genf as genf # import this module to use custom made graph generation function(s)

from SVWCBS import rankn as rn # import this module to use node ranking measures

from SVWCBS import helperf as hf # import this module to use helper functions (pickle opening functions for example)



"""
DOCUMENTATION REGARDING FUNCTIONS

Functions for complexity measures and node ranking methods used in the thesis of Simon van Wageningen during
the internship at Statistics Netherlands

------------------------------------------------------------------------------------------------

For all functions that use the adjacency matrix:
    the adjacency matrix has to be in the format where the rows are the sources and the columns are the destinations
""" 


"""
------------------------------------------------------------------------------------------------
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


"""
trans_close(G)
Transitive Close:
Used as an alternative to Warshall's algorithm, computes the transitive closure matrix
in a much faster way

Input:  Graph G, a digraph object from networkx
Output: A*, the transitive closure matrix in scipy sparse format
        note that its elements are in uint8 format, beware with matrix operations

"""


"""
strongly_connected_check(G)
Strongly connected check:
Checks whether a digraph G or adjacency matrix A is fully reachable/strongly connected

Input:  Graph G, a digraph object from networkx
Output: check, a boolean variable, True if the network is fully reachable/strongly connected
"""


"""
average_degree(G)
Average degree complexity:
Graph complexity measure, it simply illustrates the average degree over all nodes in the graph

Input:  Graph G, a (di)graph object from networkx
Output: C_deg, a single float value representing the average degree of the graph G
"""


"""
routing(G)
Routing complexity:
Routing complexity measure, it measures the amount of paths from source node to sink node in
ACYCLIC DIRECTED graphs, this is a requirement otherwise the function may not find a solution

Input:  Graph G, an acyclic digraph object from networkx
Output: C_rou, a single float value representing the routing complexity of the digraph G
"""


"""
arc_symmetry(G)
Arc symmetry:
Arc Symmetry complexity measure, it quantifies how many arcs do not have a counter-arc
DIRECTED graphs are required since we're dealing with arcs

Input:  Graph G, a digraph object from networkx
Output: C_sym, a single float value representing the arc symmetry complexity of the digraph G
"""


"""
entropy(G)
Entropy:
Entropy complexity measure, attempts to measure how far a digraph is removed from full reachability
The lower bound is 0, then the digraph is fully reachable or there are isolated nodes.

Input:  Graph G, a digraph object from networkx
Output: C_e, a single float value representing how far the digraph G is removed from full reachability,
        small values indicate that the graph is closer to full reachability than large values
"""


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


"""
average_distance(G)
Average Distance:
Average Distance complexity measure, calculates the average distance between all nodes in the connected graph
This metric is already defined by networkx

Input:  Graph G, a (di)graph object from networkx
Output: C_d, a single float value representing the average distance/shortest path length between nodes
"""


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


"""
---------------------------------------------------------------------------------------------
Node Ranking metrics:
"""


"""
rank_one(G)
Node Ranking metric r1:

Input:  Graph G, a digraph object from networkx
Output: r_dict, a dictionary of length n with the nodes as keys and node rank values as values
"""


"""
rank_two(G, sigma = 0.85)
Node Ranking metric r2:
    
Input:  Graph G, a digraph object from networkx
        sigma, a control parameter, default is 0.85 same as pagerank default
Output: r_dict, a dictionary of length n with the nodes as keys and node rank values as values
"""


"""
coreness(G)
Coreness Centrality:
    
Input:  Graph G, a (di)graph object from networkx
Output: r_core, a dictionary of length n with the nodes as keys and node rank values as values
"""


"""
coreness_plus(G, coreness_dict = None):
Coreness Plus Centrality:
    
Input:  Graph G, a (di)graph object from networkx
        coreness_dict, a dictionary containing nodes of the graph G as keys and coreness values as values, OPTIONAL
Output: r_dict, a dictionary of length n with the nodes as keys and node rank values as values
"""


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


"""            
------------------------------------------------------------------------------------------------
Other useful but not necessarily needed helper functions
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


"""
open_pickle(path)
Open Pickle Function:
A function that opens a pickle file

Input:  path, the path to the pickle file including the pickle name, in string format
Output: obj, the objects stored in the pickle
"""


"""
save_pickle(path, objects):
Save Pickle Function:
A function that opens a pickle file

Input:  path, the path to the pickle file including the pickle name, in string format
        objects, the objects to be stored in the pickle
"""
