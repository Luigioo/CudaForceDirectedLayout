import numpy as np
import networkx as nx

import numpy as np
import networkx as nx
import random

def fruchterman_reingold_layout_cpu(G, k=None, iterations=50, temp=0.1, cooling_factor=0.9, seed=None):
    num_nodes = len(G)
    num_edges = len(G.edges)
    edges = np.array(G.edges())
    
    # Initialize random seed
    random.seed(seed)

    # Compute default spring constant if not provided
    if k is None:
        A = 1.0
        k = np.sqrt(A / num_nodes)

    # Initialize positions randomly
    pos = np.random.rand(num_nodes, 2)

    # Initialize forces
    repulsive_forces = np.zeros((num_nodes, 2))
    attractive_forces = np.zeros((num_nodes, 2))

    for _ in range(iterations):
        # Calculate repulsive forces
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                delta = pos[i] - pos[j]
                distance = np.linalg.norm(delta)
                repulsive_force = k * k / distance
                repulsive_forces[i] += delta * repulsive_force / distance
                repulsive_forces[j] -= delta * repulsive_force / distance

        # Calculate attractive forces
        for i in range(num_edges):
            node1, node2 = edges[i]
            delta = pos[node1] - pos[node2]
            distance = np.linalg.norm(delta)
            # attractive_force = distance * distance / k
            attractive_forces[node1] -= delta * distance / k
            attractive_forces[node2] += delta * distance / k

        # Apply forces
        for i in range(num_nodes):
            net_force = attractive_forces[i] + repulsive_forces[i]
            distance = np.linalg.norm(net_force)
            displacement = min(distance, 0.1) * net_force / distance
            pos[i] += displacement
            pos[i] = np.clip(pos[i], 0.01, 1.0)  # Ensure nodes stay within bounding box

        # Reset attractive and repulsive forces
        attractive_forces.fill(0)
        repulsive_forces.fill(0)

        # Cool down
        temp *= cooling_factor

    return {i: pos[i] for i in range(num_nodes)}


def simplified_spring_layout(G, iterations=50):
    # Initialize the positions with a uniform random distribution
    pos = np.random.rand(len(G.nodes), 2)

    # Normalized distance for repulsion (to avoid calculating the sqrt every time)
    k = np.sqrt(1.0 / len(G.nodes))

    # Start the iterations
    for _ in range(iterations):
        # Calculate the repulsive forces
        displacement = np.zeros((len(G.nodes), 2))
        for i in range(len(G.nodes)):
            for j in range(i + 1, len(G.nodes)):
                delta = pos[i] - pos[j]
                magnitude = np.sqrt(delta @ delta)
                distance = np.linalg.norm(delta)
                # print(str(magnitude-distance))
                displacement[i] += delta * k * k / magnitude
                displacement[j] -= delta * k * k / magnitude

        # Calculate the attractive forces
        for i, j in G.edges:
            delta = pos[i] - pos[j]
            magnitude = np.sqrt(delta @ delta)
            displacement[i] -= delta * magnitude / k
            displacement[j] += delta * magnitude / k

        # Update the positions
        for i in range(len(G.nodes)):
            magnitude = np.sqrt(displacement[i] @ displacement[i])
            pos[i] += displacement[i] * min(magnitude, 0.1) / magnitude
            # pos[i] = np.clip(pos[i], 0.01, 1.0)

    # Return as a dictionary of node : position
    return {n: pos[i] for i, n in enumerate(G.nodes)}
