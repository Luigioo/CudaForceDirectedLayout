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
import test

# Constants

# Dont change
INPUT_FOLDER = "./graph_data/"
OUTPUT_FOLDER = "./position_data/"
# Analysis settings
ENABLE_CROSSING_EDGES = False
RUN_GPU_ONLY = False
RUN_NETWORKX_ALGO = False
# input settings
NUMNODES = 100
INTERATIONS = 500
GRAPH_TYPE = "grid"

settings_string = str(GRAPH_TYPE) + "_" + "nodes_"+str(NUMNODES)
input_path = INPUT_FOLDER + settings_string + "_input.txt"
cuda_output_path = OUTPUT_FOLDER + settings_string + "_cuda_output.txt"
cpu_output_path = OUTPUT_FOLDER + settings_string +"_cpu_output.txt"

def generate_data():
    start_time = time.time()
    G = GenData.generate_grid_graph(math.isqrt(NUMNODES), math.isqrt(NUMNODES))
    GenData.GraphToArray(G, input_path)
    end_time = time.time()
    print(f"Data generation time: {end_time - start_time} seconds")
    return G

def run_cpp_program(G):
    # CUDA
    start_time = time.time()
    process = subprocess.Popen(["./x64/Debug/CudaRuntime2.exe", "-i", input_path, "-o", cuda_output_path, "-p", "0", "-t", str(INTERATIONS)], stdout=subprocess.PIPE, universal_newlines=True)
    while True:
        output = process.stdout.readline()
        print(output.strip())
        # Break the loop once the process is done
        return_code = process.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output 
            for output in process.stdout.readlines():
                print(output.strip())
            break
    end_time = time.time()
    cuda_runtime = end_time - start_time
    print(f"Running the CUDA algorithm time: {cuda_runtime} seconds")
    arr = Analyze.read_double_values(cuda_output_path)
    cuda_pos = Analyze.group_elements(arr)

    # CPU
    cpu_pos = None
    cpu_runtime = None
    if not RUN_GPU_ONLY:
        start_time = time.time()
        process = subprocess.Popen(["./x64/Debug/CudaRuntime2.exe", "-i", input_path, "-o", cpu_output_path, "-p", "1", "-t", str(INTERATIONS)], stdout=subprocess.PIPE, universal_newlines=True)
        while True:
            output = process.stdout.readline()
            print(output.strip())
            # Break the loop once the process is done
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output 
                for output in process.stdout.readlines():
                    print(output.strip())
                break
        end_time = time.time()
        cpu_runtime = end_time - start_time
        print(f"Running the CPU algorithm time: {cpu_runtime} seconds")
        arr = Analyze.read_double_values(cpu_output_path)
        cpu_pos = Analyze.group_elements(arr)

    return cuda_pos, cpu_pos, cuda_runtime, cpu_runtime

def analyze_results(G, cuda_pos, cuda_runtime, cpu_pos, cpu_runtime):
    if ENABLE_CROSSING_EDGES and not RUN_GPU_ONLY:
        print("edge crossings (fr_cuda): " + str(Analyze.calculate_edge_crossings(G, cuda_pos)))
        print("edge crossings (fr_cpu): " + str(Analyze.calculate_edge_crossings(G, cpu_pos)))
    elif ENABLE_CROSSING_EDGES:
        print("edge crossings (fr_cuda): " + str(Analyze.calculate_edge_crossings(G, cuda_pos)))
    Analyze.displayResults(G, cuda_pos, cuda_runtime, cpu_pos, cpu_runtime)

def run_networkx_fr(G):
    pos = nx.spring_layout(G)  # by default nx.spring_layout uses the Fruchterman-Reingold algorithm
    plt.figure(figsize=(7, 7))  # set the figure size
    nx.draw(G, pos, node_size=20)  # draw the graph
    plt.show()  # show the plot


if __name__ == "__main__":
    if(not RUN_NETWORKX_ALGO):
        G = generate_data()
        cuda_pos, cpu_pos, cuda_runtime, cpu_runtime = run_cpp_program(G)
        analyze_results(G, cuda_pos, cuda_runtime, cpu_pos, cpu_runtime)
    else:
        G = generate_data()
        run_networkx_fr(G)



