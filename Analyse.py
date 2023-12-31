
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from matplotlib.colors import ListedColormap

# Macro to enable or disable calculating and printing crossing edges
ENABLE_CROSSING_EDGES = False

def deserialize_array(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [list(map(int, line.strip().split())) for line in lines]
        return np.array(data, dtype=object)

def ArrayToGraph(fileName):
    edge_edge_array = deserialize_array(fileName)

    # Extract the number of nodes from the last line
    num_nodes = edge_edge_array[-1][0]

    # Remove the last line from the array
    edge_edge_array = edge_edge_array[:-1]

    edge_edge_array = np.array(edge_edge_array[0])

    # Check if the size of edge_edge_array is a multiple of 2
    if edge_edge_array.size % 2 != 0:
        raise ValueError("Size of edge_edge_array must be a multiple of 2")

    # Reshape the array to restore the original structure
    edge_array = np.reshape(edge_edge_array, (-1, 2))

    # Create an empty graph
    G = nx.Graph()

    # Add all nodes to the graph
    G.add_nodes_from(range(1, num_nodes))

    # Add edges to the graph
    G.add_edges_from(edge_array)

    # Print the number of nodes and the edges of the graph
    # print("Number of nodes:", num_nodes)
    # print("Edges:")
    # print(G.edges())
    return G


def group_elements(arr):
    result = {}
    for i in range(0, len(arr), 2):
        # Get the two elements
        element1 = arr[i]
        element2 = arr[i+1] if i+1 < len(arr) else None

        result[i//2] = [element1, element2]

    return result

def read_double_values(file_path):
    arr = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    value = float(line.strip())
                    arr.append(value)
                except ValueError:
                    print(f"Invalid value found: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except IOError:
        print(f"Error reading file: {file_path}")

    if len(arr) == 0:
        print("No valid values found in the file.")

    return arr

        

def calculate_edge_crossings(graph, pos):
    crossings = 0

    # Iterate over each pair of edges
    for u, v in graph.edges():
        for x, y in graph.edges():
            if u != x and u != y and v != x and v != y:
                # Check for edge crossings
                if do_edges_cross(pos[u], pos[v], pos[x], pos[y]):
                    crossings += 1

    return crossings

def do_edges_cross(p1, p2, p3, p4):
    # Check if two edges cross each other
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    return (
        min(x1, x2) < max(x3, x4) and
        min(y3, y4) < max(y1, y2) and
        min(x3, x4) < max(x1, x2) and
        min(y1, y2) < max(y3, y4) and
        ((x1 - x2) * (y3 - y1) - (y1 - y2) * (x3 - x1)) * ((x1 - x2) * (y4 - y1) - (y1 - y2) * (x4 - x1)) < 0 and
        ((x3 - x4) * (y1 - y3) - (y3 - y4) * (x1 - x3)) * ((x3 - x4) * (y2 - y3) - (y3 - y4) * (x2 - x3)) < 0
    )

def displayResults(G, cuda_pos, cuda_runtime, cpu_pos=None, cpu_runtime=None, settings_string=""):
    if cpu_pos is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        ax1.set_title(f'FR CPU\nTime: {cuda_runtime:.4f} seconds')
        nx.draw(G, pos=cuda_pos, ax=ax1, node_size=20)

        ax2.set_title(f'FR CPU\nTime: {cpu_runtime:.4f} seconds')
        nx.draw(G, pos=cpu_pos, ax=ax2, node_size=20)

        plt.tight_layout()
    else:
        fig, ax1 = plt.subplots(figsize=(7, 7))

        ax1.set_title(f'Graph 1 (fr_cuda)\nTime: {cuda_runtime:.4f} seconds')
        nx.draw(G, pos=cuda_pos, ax=ax1, node_size=20)

    plt.show()

def saveResults(G, cuda_pos, cuda_runtime, cpu_pos=None, cpu_runtime=None, settings_string=""):
    if cpu_pos is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        ax1.set_title(f'GPU Enabled\nTime: {cuda_runtime:.4f} seconds')
        nx.draw(G, pos=cuda_pos, ax=ax1, node_size=20)

        ax2.set_title(f'CPU Only\nTime: {cpu_runtime:.4f} seconds')
        nx.draw(G, pos=cpu_pos, ax=ax2, node_size=20)

        plt.tight_layout()
        graph_name = f"./diagram_data/{settings_string}_gpu_cpu_{cuda_runtime}_{cpu_runtime}.png"

    else:
        fig, ax1 = plt.subplots(figsize=(7, 7))

        ax1.set_title(f'GPU Enabled\nTime: {cuda_runtime:.4f} seconds')
        nx.draw(G, pos=cuda_pos, ax=ax1, node_size=20)
        
        graph_name = f"./diagram_data/{settings_string}gpu_{cuda_runtime}.png"

    if not os.path.exists(os.path.dirname(graph_name)):
        try:
            os.makedirs(os.path.dirname(graph_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != OSError.errno.EEXIST:
                raise

    plt.savefig(graph_name)

def fsaveResults(G, cuda_pos, cuda_runtime, cpu_pos=None, cpu_runtime=None, settings_string=""):
    # Create a colormap for nodes and edges
    nodes_colormap = plt.cm.plasma
    edges_colormap = plt.cm.viridis

    node_color = [G.degree(v) for v in G]
    edge_color = range(G.number_of_edges())

    if cpu_pos is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        ax1.set_title(f'GPU Enabled\nTime: {cuda_runtime:.4f} seconds', fontsize=15)
        nx.draw(G, pos=cuda_pos, ax=ax1, node_size=50, node_color=node_color, edge_color=edge_color,
                cmap=nodes_colormap, edge_cmap=edges_colormap, width=2)

        ax2.set_title(f'CPU Only\nTime: {cpu_runtime:.4f} seconds', fontsize=15)
        nx.draw(G, pos=cpu_pos, ax=ax2, node_size=50, node_color=node_color, edge_color=edge_color,
                cmap=nodes_colormap, edge_cmap=edges_colormap, width=2)

        plt.tight_layout()
        graph_name = f"./diagram_data/{settings_string}_gpu_cpu_{cuda_runtime}_{cpu_runtime}.png"

    else:
        fig, ax1 = plt.subplots(figsize=(7, 7))

        ax1.set_title(f'GPU Enabled\nTime: {cuda_runtime:.4f} seconds', fontsize=15)
        nx.draw(G, pos=cuda_pos, ax=ax1, node_size=50, node_color=node_color, edge_color=edge_color,
                cmap=nodes_colormap, edge_cmap=edges_colormap, width=2)
        
        graph_name = f"./diagram_data/{settings_string}gpu_{cuda_runtime}.png"

    if not os.path.exists(os.path.dirname(graph_name)):
        try:
            os.makedirs(os.path.dirname(graph_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != OSError.errno.EEXIST:
                raise

    plt.savefig(graph_name)
# Example usage
# G = nx.karate_club_graph()
# numNodes = 50

# G = nx.gnp_random_graph(numNodes, 0.05, 42)

# G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}

# # convert the 2d array G.edges, which has a structure:
# # [[v1, v2],[v1,v3],[v4,v5],...[v45, v48]]
# # to: [v1, v2, v1, v3...v45, v48]
# edge_edge_array = np.array(list(G.edges)).flatten()

# start_time_cuda = time.time()
# result = algo.fr_cuda(edge_edge_array, numNodes, 1)  # execute cuda program
# end_time_cuda = time.time()
# cuda_time = end_time_cuda - start_time_cuda
# cuda_pos = group_elements(result)


# edge_edge_array = np.array(list(G.edges)).flatten()
# start_time_fr = time.time()
# normal_pos = algo.fr(edge_edge_array, numNodes, 1)  # execute cpu program
# end_time_fr = time.time()
# fr_time = end_time_fr - start_time_fr
# normal_pos = group_elements(normal_pos)



if __name__ == "__main__":

    arr = read_double_values("position_data/outputCuda.txt")
    cuda_pos = group_elements(arr)

    # arr = read_double_values("position_data/fr_CPU.txt")
    # normal_pos = group_elements(arr)

    G = ArrayToGraph('./graph_data/grid_nodes_100_input.txt')


    if ENABLE_CROSSING_EDGES:
        print("edge crossings (fr_cuda): " + str(calculate_edge_crossings(G, cuda_pos)))

    # if ENABLE_CROSSING_EDGES:
    #     print("edge crossings (fr): " + str(calculate_edge_crossings(G, normal_pos)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title('Graph 1 (fr_cuda)\nTime: {:.4f} seconds'.format(1.1))
    nx.draw(G, pos=cuda_pos, ax=ax1)

    # ax2.set_title('Graph 2 (fr)\nTime: {:.4f} seconds'.format(1.1))
    # nx.draw(G, pos=normal_pos, ax=ax2)

    plt.tight_layout()
    plt.show()
