import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import math

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
    # G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
    # edge_edge_array = np.array(list(G.edges)).flatten()
    edges = list(G.edges)
    edge_edge_array = [node for edge in edges for node in edge]
    serialize_array(edge_edge_array, fileName, 'w')

    # Append the number of nodes
    num_nodes = len(G.nodes())
    serialize_array(np.array([num_nodes]), fileName, 'a')

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


def generate_scale_free_graph(nodes, m):
    G = nx.barabasi_albert_graph(nodes, m)
    return G

def scale_free_avg_degree(nodes, m):
    return 2 * m * (nodes - m) / nodes

def generate_small_world_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    return G


def genLargeGrid(sideLength):
    with open("graph_data/large_grid.txt", 'w') as f:  # Open the file in append mode
        for r in range(sideLength-1):
            for c in range(sideLength-1):
                id = r*sideLength+c
                f.write(str(id)+" "+str(id+1)+" "+str(id)+" "+str(id+sideLength)+" ")
            id = r*sideLength+sideLength-1
            f.write(str(id)+" "+str(id+sideLength)+" ")
        rid = (sideLength-1)*sideLength
        for c in range(sideLength-1):
            id = rid+c
            f.write(str(id)+" "+str(id+1)+" ")
            
        f.write("\n")  # Add a newline character at the end
        f.write(str(sideLength*sideLength)+" \n")

if __name__ == "__main__":

    # NUMNODES = 1<<10
    # G = nx.gnp_random_graph(NUMNODES, 0.05, 42)
    # GraphToArray(G, "graph_data/random.txt")

    # Generate the grid graph
    # G = nx.grid_2d_graph(math.isqrt(NUMNODES), math.isqrt(NUMNODES),)
    # GraphToArray(G, "graph_data/grid.txt")
    genLargeGrid(1000)
    # G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
    # edge_edge_array = np.array(list(G.edges)).flatten()
    # serialize_array(edge_edge_array, "random_50.txt")
