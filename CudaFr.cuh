#ifndef CUDA_FR_H
#define CUDA_FR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/complex.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void calculateRepulsiveForces(const double* positions,
    double* repulsiveForces, const double k,
    const int numNodes);

__global__ void calculateAttractiveForces(int* edges, int numEdges, const double* positions,
    double* attractiveForces, const double k,
    const int numNodes);

__global__ void applyForces(double* positions,
    double* attractiveForces, double* repulsiveForces, int numNodes, double temp);

double* fruchterman_reingold_layout_cuda(
    int* edges, int numEdges, int numNodes, int iterations,
    double k, double temp, double cooling_factor, int seed);

void gpuAssert(cudaError_t code, const char* file, int line, bool abort);

#endif /* CUDA_FR_H */