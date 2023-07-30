
//system
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <assert.h>
//cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//thrust
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
//local
#include "CudaFr.cuh"
#include "Attractive.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct AbsoluteValue {
    __device__ double operator()(const double& x) const {
        return fabs(x);
    }
};

// CUDA kernel for calculating repulsive forces
__global__ void calculateRepulsiveForces(const double* positions,
    double* repulsiveForces, const double k,
    const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNodes) {
       
        double repulsiveForceX = 0.0;
        double repulsiveForceY = 0.0;

        for (int j = 0; j < numNodes; ++j) {
            if (i == j)
                continue;

            double deltaX = positions[i * 2] - positions[j * 2];
            double deltaY = positions[i * 2 + 1] - positions[j * 2 + 1];
            double distance = fmax(0.01, sqrt(deltaX * deltaX + deltaY * deltaY));
            double repulsiveForce = k * k / distance;

            repulsiveForceX += repulsiveForce * (deltaX / distance);
            repulsiveForceY += repulsiveForce * (deltaY / distance);

        }

        repulsiveForces[i * 2] = repulsiveForceX;
        repulsiveForces[i * 2 + 1] = repulsiveForceY;

    }
}

__global__ void calculateAttractiveForcesSingleThread(int* edges, int numEdges, const double* positions,
    double* attractiveForces, const double k,
    const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i==1)
        for (int i = 0; i < numEdges; ++i) {
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

            attractiveForces[node1 * 2] -= attractiveForceX;
            attractiveForces[node1 * 2 + 1] -= attractiveForceY;
            attractiveForces[node2 * 2] += attractiveForceX;
            attractiveForces[node2 * 2 + 1] += attractiveForceY;
        }
}

// CUDA kernel for calculating attractive forces
__global__ void calculateAttractiveForces(int* edges, int numEdges, const double* positions,
    double* attractiveForces, const double k,
    const int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
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

        atomicAdd(&(attractiveForces[node1 * 2]), -attractiveForceX);
        atomicAdd(&(attractiveForces[node1 * 2 + 1]), -attractiveForceY);
        atomicAdd(&(attractiveForces[node2 * 2]), attractiveForceX);
        atomicAdd(&(attractiveForces[node2 * 2 + 1]), attractiveForceY);

    }

}

// CUDA kernel for applying forces to position array
__global__ void applyForces(double* positions,
    double* attractiveForces, double* repulsiveForces, int numNodes, double temp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 1) {

        for (int i = 0; i < numNodes; ++i) {
            double netForceX = attractiveForces[i * 2] + repulsiveForces[i * 2];
            double netForceY = attractiveForces[i * 2 + 1] + repulsiveForces[i * 2 + 1];

            double distance = max(0.01, sqrt(netForceX * netForceX + netForceY * netForceY));
            double displacementX = min(distance, temp) * netForceX / distance;
            double displacementY = min(distance, temp) * netForceY / distance;

            positions[i * 2] += displacementX;
            positions[i * 2 + 1] += displacementY;

            //// Ensure node stays within bounding box
            //positions[i * 2] = max(0.01, min(positions[i * 2], 1.0));
            //positions[i * 2 + 1] = max(0.01, min(positions[i * 2 + 1], 1.0));
        }


    }
}

//Takes in edges data in the form:
    // [v1, v2, v1, v3...v45, v48]
    // where v1 is connected to v2, v1 is conneted to v3, and v45 is connected to v48
    // Returns a double array of calculated positions of the vetices
    // [x1, y1, x2, y2, .... , xn, yn]
    // where x1 is the x position for vertex 1
double* fruchterman_reingold_layout_cuda(
    int* edges, int numEdges, int numNodes, int iterations,
    double k, double temp, double cooling_factor, int seed) {

    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (k == 0.0) {
        // Compute default spring constant
        double A = 1.0;
        k = sqrt(A / numNodes);
    }

    // Allocate host memory and Initialize positions randomly
    double* pos = new double[numNodes * 2];
    for (int i = 0;i < numNodes;i++) {
        pos[i * 2] = dis(gen);
        pos[i * 2 + 1] = dis(gen);

    }

    // Allocate device memory
    int* d_edges;
    double* d_positions;
    double* d_repulsiveForces;
    double* d_attractiveForces;
    gpuErrchk(cudaMalloc((void**)&d_edges, numEdges * 2 * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_edges, edges, numEdges * 2 * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_positions, numNodes * 2 * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_positions, pos, numNodes * 2 * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_repulsiveForces, numNodes * 2 * sizeof(double)));
    gpuErrchk(cudaMemset(d_repulsiveForces, 0, numNodes * 2 * sizeof(double)));

    gpuErrchk(cudaMalloc((void**)&d_attractiveForces, numNodes * 2 * sizeof(double)));
    gpuErrchk(cudaMemset(d_attractiveForces, 0, numNodes * 2 * sizeof(double)));

    // CUDA grid and block dimensions
    int threadNeeded = max(numEdges, numNodes);
    int blockSize = 256;
    int gridSize = (threadNeeded + blockSize - 1) / blockSize;

    for (int iter = 0; iter < iterations; ++iter) {
        // Compute repulsive forces
        calculateRepulsiveForces<<<gridSize,blockSize>>>(d_positions, d_repulsiveForces, k, numNodes);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        // Compute attractive forces
        //calculateAttractiveForces<<<gridSize,blockSize>>>(d_edges, numEdges, d_positions, d_attractiveForces, k, numNodes);
        calculateAttractiveForces<<<gridSize, blockSize>>>(d_edges, numEdges, d_positions, d_attractiveForces, k, numNodes);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        applyForces<<<gridSize,blockSize>>>(d_positions, d_attractiveForces, d_repulsiveForces, numNodes, temp);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        temp *= cooling_factor;


        //thrust::device_ptr<double> devicePtr = thrust::device_pointer_cast(d_attractiveForces);
        //thrust::device_vector<double> deviceVec(devicePtr, devicePtr + numNodes*2);
        //thrust::transform(deviceVec.begin(), deviceVec.end(), deviceVec.begin(), AbsoluteValue());
        //double sumAttractiveForces = thrust::reduce(deviceVec.begin(), deviceVec.end());
        //cout<<"sum of attrac: "<<sumAttractiveForces << endl;

        // Accumulate sum of absolute forces
        //double sumRepulsiveForces = 0.0;
        //double sumAttractiveForces = 0.0;

        //auto sumRepulsive = thrust::device_pointer_cast(d_repulsiveForces);
        //std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1); // Set precision to maximum
        //cout << "repulsive: " << endl;
        //for (int i = 0; i < numNodes*2; i++) {
        //    std::cout << sumRepulsive[i] << " ";
        //    if (i != 0 && i % 9 == 0) cout << endl;
        //}
        //cout << "attracgtive: " << endl;
        //for (int i = 0; i < numNodes * 2; i++) {
        //    std::cout << sumAttractive[i] << " ";
        //    if (i != 0 && i % 9 == 0) cout << endl;
        //}

        ////std::cout << "repo2x: " << sumRepulsive[4] << std::endl;

        //std::cout << "repulsive forces: " << sumRepulsiveForces << std::endl;
        //std::cout << "Sum of absolute values of attractive forces: " << sumAttractiveForces << std::endl;

        //if (iter == 0) {
        //    for (int i = 0; i < 10; i++) {
        //        std::cout << sumRepulsive[i] << " ";
        //    }
        //    std::cout << std::endl;
        //}

        //reset attractive and repulsive forces
        gpuErrchk(cudaMemset(d_attractiveForces, 0, numNodes * 2 * sizeof(double)));
        gpuErrchk(cudaMemset(d_repulsiveForces, 0, numNodes * 2 * sizeof(double)));

        cout << "iteration: " << iter << endl;

    }

    cudaMemcpy(pos, d_positions, numNodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_repulsiveForces);
    cudaFree(d_attractiveForces);

    return pos;
}
