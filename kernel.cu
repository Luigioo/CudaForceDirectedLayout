
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <iomanip>


#include "CudaFr.cuh"
#include "LoadData.h"
#include "CpuFr.h"
#include "Config.h"


using namespace std;

template <typename T>
bool areArraysEqual(const T arr1[], const T arr2[], int size) {

	for (int i = 0; i < size; i++)
		if (arr1[i] != arr2[i]) return false;

	return true;
}

std::string recordTime() {
	static std::chrono::time_point<std::chrono::system_clock> lastTime = std::chrono::system_clock::now();
	std::chrono::time_point<std::chrono::system_clock> currentTime = std::chrono::system_clock::now();
	std::chrono::duration<double, std::milli> elapsed_milliseconds = currentTime - lastTime;
	lastTime = currentTime;

	std::ostringstream stream;
	stream << std::fixed << std::setprecision(3) << elapsed_milliseconds.count();

	std::string elapsedStr = stream.str();
	size_t decimalPos = elapsedStr.find('.');
	if (decimalPos != std::string::npos) {
		elapsedStr.resize(decimalPos + 4); // Keep up to 3 decimal places
	}

	return elapsedStr + " ms";
}

int main(int argc, char* argv[]) {
	recordTime();
	Config cfg;
	cfg.ParseArgs(argc, argv);
	cout<<"Input Path: "<<cfg.input_path<<endl;

	int arrsize;
	int numEdges;
	int numNodes;
	auto currentTime = std::chrono::system_clock::now();
	int* arr = LoadData::readFileAndConvertInt(cfg.input_path, arrsize, numNodes);
	numEdges = arrsize / 2;
	cout << "Reading file took " << recordTime() << endl;

	if (arr != nullptr) {
		cout << "Number of Nodes: " << numNodes << endl;
		cout << "Number of Edges: " << numEdges << endl;
	}

	if (cfg.processor == 0) {
		cout << "Running CUDA algorithm" << endl;
		recordTime();
		double* positions = fruchterman_reingold_layout_cuda(arr, numEdges, numNodes, cfg.iteration);
		cout<<"CUDA took "<<recordTime()<<endl;
		LoadData::writeFileWithPrecision(cfg.output_path, positions, numNodes * 2);
		delete[] positions;
	}
	else if (cfg.processor == 1) {
		cout << "Running CPU algorithm" << endl;
		recordTime();
		double* cpuPos = CpuFr::fruchterman_reingold_layout_cpu(arr, numEdges, numNodes, cfg.iteration);
		cout << "CPU took " << recordTime() << endl;
		LoadData::writeFileWithPrecision(cfg.output_path, cpuPos, numNodes * 2);
		delete[] cpuPos;
	}

	delete[] arr;


}


