
# common libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import subprocess
import math

# local
import GenData
import Analyze
import RunCPP


# G = nx.gnp_random_graph(NUMNODES, 0.05, 42)
# GenData.GraphToArray(G, "graph_data/random.txt")

# Generate the grid graph


if __name__ == "__main__":
    ENABLE_CROSSING_EDGES = False

    NUMNODES = 2<<16

    G = GenData.generate_grid_graph(math.isqrt(NUMNODES), math.isqrt(NUMNODES))
    # self_edges = [(u, v) for u, v in G.edges() if u == v]
    # G.remove_edges_from(self_edges)
    GenData.GraphToArray(G, "graph_data/grid.txt")

    RunCPP.runProgram("./x64/Debug/CudaRuntime2.exe")

    arr = Analyze.read_double_values("position_data/fr_CUDA.txt")
    cuda_pos = Analyze.group_elements(arr)

    arr = Analyze.read_double_values("position_data/fr_CPU.txt")
    normal_pos = Analyze.group_elements(arr)

    # G = Analyze.ArrayToGraph('graph_data/random.txt')


    if ENABLE_CROSSING_EDGES:
        print("edge crossings (fr_cuda): " + str(Analyze.calculate_edge_crossings(G, cuda_pos)))

    if ENABLE_CROSSING_EDGES:
        print("edge crossings (fr): " + str(Analyze.calculate_edge_crossings(G, normal_pos)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title('Graph 1 (fr_cuda)\nTime: {:.4f} seconds'.format(1.1))
    nx.draw(G, pos=cuda_pos, ax=ax1)

    ax2.set_title('Graph 2 (fr)\nTime: {:.4f} seconds'.format(1.1))
    nx.draw(G, pos=normal_pos, ax=ax2)

    plt.tight_layout()
    plt.show()


