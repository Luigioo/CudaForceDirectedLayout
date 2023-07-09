#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/complex.h>

double* fruchterman_reingold_layout_cuda(
    int* edges, int numEdges, int numNodes, int iterations = 50,
    double k = 0.0, double temp = 1.0, double cooling_factor = 0.95, int seed = 42);
