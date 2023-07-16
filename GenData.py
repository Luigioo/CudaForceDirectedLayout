import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

# Example usage:
    # arr = np.array([1.1234567891234567, 2.2345678912345678, 3.3456789123456789, 4.4567891234567890, 5.5678912345678901], dtype=np.float64)
    # serialize_array(arr, 'array_data.txt')
def serialize_array(arr, filename, aorw):
    with open(filename, aorw) as f:  # Open the file in append mode
        for item in arr:
            f.write("%d " % item)
        f.write("\n")  # Add a newline character at the end



# convert the 2d array G.edges, structured as:
    # [[v1, v2],[v1,v3],[v4,v5],...[v45, v48]]
    # to: [v1, v2, v1, v3...v45, v48]
def GraphToArray(G, fileName):
    G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
    edge_edge_array = np.array(list(G.edges)).flatten()
    serialize_array(edge_edge_array, fileName, 'w')

    # Append the number of nodes
    num_nodes = len(G.nodes())
    serialize_array(np.array([num_nodes]), fileName, 'a')


NUMNODES = 500
G = nx.gnp_random_graph(NUMNODES, 0.05, 42)
GraphToArray(G, "graph_data/random.txt")

# Generate the grid graph
m = 50  # Number of rows
n = 50  # Number of columns
G = nx.grid_2d_graph(m, n)
GraphToArray(G, "graph_data/grid.txt")

# G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
# edge_edge_array = np.array(list(G.edges)).flatten()
# serialize_array(edge_edge_array, "random_50.txt")
