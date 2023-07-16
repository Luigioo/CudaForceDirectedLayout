
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


if __name__ == "__main__":
    ENABLE_CROSSING_EDGES = False

    NUMNODES = 1<<30

    INPUT_FOLDER = "./graph_data/"
    OUTPUT_FOLDER = "./position_data/"


    input_path = INPUT_FOLDER+"grid.txt"
    output_path = OUTPUT_FOLDER+"gridCuda.txt"    
    G = GenData.generate_grid_graph(math.isqrt(NUMNODES), math.isqrt(NUMNODES))
    GenData.GraphToArray(G, input_path)
    RunCPP.runProgram(["./x64/Debug/CudaRuntime2.exe", "-i", input_path, "-o", output_path, "-p", "0", "-t", "50"])
    arr = Analyze.read_double_values(output_path)
    cuda_pos = Analyze.group_elements(arr)

    # input_path = INPUT_FOLDER+"grid.txt"
    # output_path = OUTPUT_FOLDER+"gridCpu.txt"    
    # G = GenData.generate_grid_graph(math.isqrt(NUMNODES), math.isqrt(NUMNODES))
    # GenData.GraphToArray(G, input_path)
    # RunCPP.runProgram(["./x64/Debug/CudaRuntime2.exe", "-i", input_path, "-o", output_path, "-p", "1", "-t", "50"])
    # arr = Analyze.read_double_values(output_path)
    # normal_pos = Analyze.group_elements(arr)


    if ENABLE_CROSSING_EDGES:
        print("edge crossings (fr_cuda): " + str(Analyze.calculate_edge_crossings(G, cuda_pos)))

    if ENABLE_CROSSING_EDGES:
        print("edge crossings (fr): " + str(Analyze.calculate_edge_crossings(G, normal_pos)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title('Graph 1 (fr_cuda)\nTime: {:.4f} seconds'.format(1.1))
    nx.draw(G, pos=cuda_pos, ax=ax1)

    # ax2.set_title('Graph 2 (fr)\nTime: {:.4f} seconds'.format(1.1))
    # nx.draw(G, pos=normal_pos, ax=ax2)

    plt.tight_layout()
    plt.show()


