import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

def serialize_array(arr, filename):
    with open(filename, 'w') as f:
        for item in arr:
            # write each item with precision of double (default in python)
            f.write("%d " % item)

# Example usage:
    # arr = np.array([1.1234567891234567, 2.2345678912345678, 3.3456789123456789, 4.4567891234567890, 5.5678912345678901], dtype=np.float64)
    # serialize_array(arr, 'array_data.txt')


NUMNODES = 50

G = nx.gnp_random_graph(NUMNODES, 0.05, 42)


# convert the 2d array G.edges, structured as:
    # [[v1, v2],[v1,v3],[v4,v5],...[v45, v48]]
    # to: [v1, v2, v1, v3...v45, v48]
G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
edge_edge_array = np.array(list(G.edges)).flatten()
serialize_array(edge_edge_array, "random_50.txt")
