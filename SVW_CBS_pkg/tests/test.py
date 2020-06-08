from SVWCBS import complf as cmp # import this module to use graph complexity measures
import networkx as nx

# generate a random graph
G = nx.fast_gnp_random_graph(n = 50, p = 0.5, directed = True)

# use any function from the complexity measure functions
arc_results = cmp.arc_symmetry(G)
print(arc_results)
prob_results = cmp.prob_reach_two(G)
print(prob_results)

from SVWCBS import rankn as rn # import this module to use node ranking measures

r1_results = rn.rank_one(G)

core_results = rn.coreness(G)


from SVWCBS import genf as genf # import this module to use custom made graph generation 

G_rou = genf.random_routing_graph(50, p = 0.5)

arc_results_rou = cmp.arc_symmetry(G_rou)
print(arc_results_rou)


from SVWCBS import helperf as hf # import this module to use helper functions (pickle opening functions for example)
metric_results, time_results = hf.run_time(f = cmp.arc_symmetry, G = G_rou)
print(metric_results, time_results)


from SVWCBS import * # you can import all modules at once but then you'll have to specify each module when calling a function
arc_results = complf.arc_symmetry(G)