# common libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import subprocess
import math


def generate_grid_graph(rows, columns):
    G = nx.Graph()
    
    # Add nodes
    for r in range(rows):
        for c in range(columns):
            node_id = r*rows+c
            G.add_node(node_id)
    
    # Add edges

    for r in range(rows):
        for c in range(columns):
            node_id = r*rows+c
            
            # Connect horizontally
            if c < columns - 1:
                neighbor_id = r*rows+(c+1)
                G.add_edge(node_id, neighbor_id)
            
            # Connect vertically
            if r < rows - 1:
                neighbor_id = (r+1)*rows+c
                G.add_edge(node_id, neighbor_id)
    
    return G


ENABLE_CROSSING_EDGES = False

NUMNODES = 5
# G = nx.gnp_random_graph(NUMNODES, 0.05, 42)
# GenData.GraphToArray(G, "graph_data/random.txt")

# Generate the grid graph

G = generate_grid_graph(5, 6)
# edges = list(G.edges)
# edge_edge_array = [node for edge in edges for node in edge]

# print(edge_edge_array)

G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
edge_edge_array = np.array(list(G.edges)).flatten()
print(edge_edge_array)