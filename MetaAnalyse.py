import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

ADDLABELS = False

def extract_information(folder_path='diagram_data'):
    results = {}
    pattern = re.compile(r'Graph_(?P<graph_type>smallworld|grid|other_graph_types)_'
                         r'(?:Degree_(?P<degree>\d+)_)?'
                         r'Nodes_(?P<nodes>\d+)_'
                         r'Iterations_(?P<iterations>\d+)_'
                         r'gpu_cpu_(?P<gpu_runtime>\d+\.\d+)_'
                         r'(?P<cpu_runtime>\d+\.\d+)')

    for file_name in os.listdir(folder_path):
        match = pattern.match(file_name)
        if match:
            info = match.groupdict()
            if info['graph_type'] == 'grid':
                info['degree'] = '4'

            key = (info['graph_type'], info['degree'], info['nodes'], info['iterations'])
            if key not in results:
                results[key] = {'gpu_runtime': [], 'cpu_runtime': [], 'creation_times': []}

            # Get the creation time
            file_path = os.path.join(folder_path, file_name)
            creation_time = os.path.getctime(file_path)

            results[key]['gpu_runtime'].append(float(info['gpu_runtime']))
            results[key]['cpu_runtime'].append(float(info['cpu_runtime']))
            results[key]['creation_times'].append(creation_time)

    # Remove the entire record for the earliest generated file, and calculate the average
    averaged_results = []
    for key, value in results.items():
        graph_type, degree, nodes, iterations = key

        # If there are at least 2 values, find the index of the earliest creation time and remove both GPU and CPU runtimes at that index
        if len(value['creation_times']) >= 2:
            idx_earliest = value['creation_times'].index(min(value['creation_times']))
            value['gpu_runtime'].pop(idx_earliest)
            value['cpu_runtime'].pop(idx_earliest)

        gpu_avg = sum(value['gpu_runtime']) / len(value['gpu_runtime'])
        cpu_avg = sum(value['cpu_runtime']) / len(value['cpu_runtime'])
        averaged_results.append({
            'graph_type': graph_type,
            'degree': degree,
            'nodes': nodes,
            'iterations': iterations,
            'gpu_runtime': gpu_avg,
            'cpu_runtime': cpu_avg
        })

    return averaged_results




def draw_effect_of_nodes(data, graph_type, iterations, degree=None):
    nodes = []
    gpu_runtimes = []
    cpu_runtimes = []

    for item in data:
        if item['graph_type'] == graph_type and item['iterations'] == str(iterations) and (degree is None or item['degree'] == str(degree)):
            nodes.append(int(item['nodes']))
            gpu_runtimes.append(float(item['gpu_runtime']))
            cpu_runtimes.append(float(item['cpu_runtime']))

    # Sort data by the number of nodes
    sorted_data = sorted(zip(nodes, gpu_runtimes, cpu_runtimes))
    nodes, gpu_runtimes, cpu_runtimes = zip(*sorted_data)

    plt.plot(nodes, gpu_runtimes, 'o-', label='GPU Runtime')
    plt.plot(nodes, cpu_runtimes, 's-', label='CPU Runtime')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime (seconds)')
    plt.legend()

    # Prepare the file name using the title text and save the plot
    file_name = f'Effect of Number of Nodes on GPU and CPU Runtime for {graph_type.capitalize()} Graphs ({iterations} iterations, {degree if degree else "all degrees"})'
    file_path = os.path.join('meta_data', file_name.replace(' ', '_') + '.pdf')

    # Create the meta_data folder if it doesn't exist
    os.makedirs('meta_data', exist_ok=True)

    if ADDLABELS:
        # Add labels for GPU runtimes
        for i, txt in enumerate(gpu_runtimes):
            plt.annotate(f'{txt:.2f}', (nodes[i], gpu_runtimes[i]), color='blue', xytext=(0,5), textcoords='offset points')

        # Add labels for CPU runtimes
        for i, txt in enumerate(cpu_runtimes):
            plt.annotate(f'{txt:.2f}', (nodes[i], cpu_runtimes[i]), color='red', xytext=(0,-15), textcoords='offset points')

    # Save the plot
    plt.savefig(file_path, format='pdf')
    plt.show()
    print(f"Diagram saved to {file_path}")

def draw_effect_of_degrees(data, graph_type, nodes, iterations):
    degrees = []
    gpu_runtimes = []
    cpu_runtimes = []

    for item in data:
        if item['graph_type'] == graph_type and item['nodes'] == str(nodes) and item['iterations'] == str(iterations):
            degrees.append(int(item['degree']))
            gpu_runtimes.append(float(item['gpu_runtime']))
            cpu_runtimes.append(float(item['cpu_runtime']))

    sorted_data = sorted(zip(degrees, gpu_runtimes, cpu_runtimes))
    print("asdf"+str(len(sorted_data)))
    degrees, gpu_runtimes, cpu_runtimes = zip(*sorted_data)

    plt.plot(degrees, gpu_runtimes, 'o-', label='GPU Runtime')
    plt.plot(degrees, cpu_runtimes, 's-', label='CPU Runtime')
    plt.xlabel('Degree')
    plt.ylabel('Runtime (seconds)')
    plt.legend()

    file_name = f'Effect of Degree on GPU and CPU Runtime for {graph_type.capitalize()} Graphs ({nodes} nodes, {iterations} iterations)'
    file_path = os.path.join('meta_data', file_name.replace(' ', '_') + '.pdf')
    os.makedirs('meta_data', exist_ok=True)

    for i, txt in enumerate(gpu_runtimes):
        plt.annotate(f'{txt:.2f}', (degrees[i], gpu_runtimes[i]), color='blue', xytext=(0,5), textcoords='offset points')

    for i, txt in enumerate(cpu_runtimes):
        plt.annotate(f'{txt:.2f}', (degrees[i], cpu_runtimes[i]), color='red', xytext=(0,-15), textcoords='offset points')

    plt.savefig(file_path, format='pdf')
    plt.show()
    print(f"Diagram saved to {file_path}")


def draw_effect_of_iterations(data, graph_type, nodes, degree):
    iterations = []
    gpu_runtimes = []
    cpu_runtimes = []

    for item in data:
        if item['graph_type'] == graph_type and item['nodes'] == str(nodes) and (degree is None or item['degree'] == str(degree)):
            iterations.append(int(item['iterations']))
            gpu_runtimes.append(float(item['gpu_runtime']))
            cpu_runtimes.append(float(item['cpu_runtime']))

    sorted_data = sorted(zip(iterations, gpu_runtimes, cpu_runtimes))
    iterations, gpu_runtimes, cpu_runtimes = zip(*sorted_data)

    plt.plot(iterations, gpu_runtimes, 'o-', label='GPU Runtime')
    plt.plot(iterations, cpu_runtimes, 's-', label='CPU Runtime')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Runtime (seconds)')
    plt.legend()

    file_name = f'Effect of Number of Iterations on GPU and CPU Runtime for {graph_type.capitalize()} Graphs ({nodes} nodes, {degree if degree else "all degrees"} degrees)'
    file_path = os.path.join('meta_data', file_name.replace(' ', '_') + '.pdf')
    os.makedirs('meta_data', exist_ok=True)

    for i, txt in enumerate(gpu_runtimes):
        plt.annotate(f'{txt:.2f}', (iterations[i], gpu_runtimes[i]), color='blue', xytext=(0,5), textcoords='offset points')

    for i, txt in enumerate(cpu_runtimes):
        plt.annotate(f'{txt:.2f}', (iterations[i], cpu_runtimes[i]), color='red', xytext=(0,-15), textcoords='offset points')

    plt.savefig(file_path, format='pdf')
    plt.show()
    print(f"Diagram saved to {file_path}")


def investigate_sequence_effect(graph_type=None, nodes=None, iterations=None, degree=None, folder_path='diagram_data'):
    results = []
    pattern = re.compile(r'Graph_(?P<graph_type>\w+)_'
                         r'(?:Degree_(?P<degree>\d+)_)?'
                         r'Nodes_(?P<nodes>\d+)_'
                         r'Iterations_(?P<iterations>\d+)_'
                         r'gpu_cpu_(?P<gpu_runtime>\d+\.\d+)_'
                         r'(?P<cpu_runtime>\d+\.\d+)')

    for file_name in os.listdir(folder_path):
        match = pattern.match(file_name)
        if match:
            info = match.groupdict()
            if info['graph_type'] == 'grid':
                info['degree'] = '4'

            # Check if the graph matches the specified criteria
            if (graph_type and info['graph_type'] != graph_type) or \
               (nodes and info['nodes'] != nodes) or \
               (iterations and info['iterations'] != iterations) or \
               (degree and info['degree'] != degree):
                continue

            # Get the creation time and convert it to a timestamp
            file_path = os.path.join(folder_path, file_name)
            creation_time = os.path.getctime(file_path)
            info['creation_time'] = datetime.fromtimestamp(creation_time)

            results.append(info)

    # Sort results by creation time to get the sequence of execution
    results.sort(key=lambda x: x['creation_time'])

    gpu_runtimes = [float(r['gpu_runtime']) for r in results]
    cpu_runtimes = [float(r['cpu_runtime']) for r in results]
    sequence = list(range(1, len(gpu_runtimes) + 1))

    plt.plot(sequence, gpu_runtimes, 'o-', label='GPU Runtime')
    plt.plot(sequence, cpu_runtimes, 's-', label='CPU Runtime')
    plt.xlabel('Execution Sequence')
    plt.ylabel('Runtime (seconds)')
    plt.legend()

    if ADDLABELS:
        # Add labels for GPU and CPU runtimes
        for i, (gpu, cpu) in enumerate(zip(gpu_runtimes, cpu_runtimes)):
            plt.annotate(f'{gpu:.2f}', (sequence[i], gpu_runtimes[i]), color='blue', xytext=(0, 5), textcoords='offset points')
            plt.annotate(f'{cpu:.2f}', (sequence[i], cpu_runtimes[i]), color='red', xytext=(0, -15), textcoords='offset points')

    # Prepare the file name and save the plot
    file_name = f'Effect_of_Execution_Sequence_on_GPU_and_CPU_Runtime_{graph_type}_Nodes_{nodes}_Iterations_{iterations}_Degree_{degree}.pdf'
    file_path = os.path.join('meta_data', file_name)

    # Create the meta_data folder if it doesn't exist
    os.makedirs('meta_data', exist_ok=True)

    # Save the plot
    plt.savefig(file_path, format='pdf')
    plt.show()
    print(f"Diagram saved to {file_path}")


def main():
    data = extract_information()
    # draw_effect_of_nodes(data, 'grid', 50, 4)
    # draw_effect_of_iterations(data, 'grid', 2000, 4)
    draw_effect_of_degrees(data, 'smallworld', 1000, 50)

if __name__ == '__main__':
    main()
    # investigate_sequence_effect(graph_type='grid', nodes='1000', iterations='50', degree='4')