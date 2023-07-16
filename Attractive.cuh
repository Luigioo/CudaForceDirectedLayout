#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void calculateAttractiveForcesReduce(int* edges, int numEdges, const double* positions,
    double* attractiveForces, const double k,
    const int numNodes) {

    __shared__ double cacheForces[2 * 256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    cacheForces[2 * cacheIndex] = 0;
    cacheForces[2 * cacheIndex + 1] = 0;

    if (i < numEdges) {
        int node1 = edges[i * 2];
        int node2 = edges[i * 2 + 1];

        double attractiveForceX = 0.0;
        double attractiveForceY = 0.0;

        double deltaX = positions[node1 * 2] - positions[node2 * 2];
        double deltaY = positions[node1 * 2 + 1] - positions[node2 * 2 + 1];
        double distance = max(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
        double attractiveForce = distance * distance / k;

        attractiveForceX = attractiveForce * (deltaX / distance);
        attractiveForceY = attractiveForce * (deltaY / distance);

        cacheForces[2 * cacheIndex] -= attractiveForceX;
        cacheForces[2 * cacheIndex + 1] -= attractiveForceY;
    }

    __syncthreads();

    // Reduce the cache array to a single value using atomicAdd.
    if (cacheIndex == 0) {
        for (int j = 0; j < blockDim.x; j++) {
            atomicAdd(&(attractiveForces[2 * blockIdx.x]), cacheForces[2 * j]);
            atomicAdd(&(attractiveForces[2 * blockIdx.x + 1]), cacheForces[2 * j + 1]);
        }
    }
}