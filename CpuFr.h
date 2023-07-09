#pragma once
class CpuFr {
public:
    static double* fruchterman_reingold_layout_cpu(int* edges, int numEdges, int numNodes,
        int iterations = 50, double k = 0.0, double temp = 1.0,
        double cooling_factor = 0.95, int seed = 42);
private:
    static void calculateRepulsiveForces(const double* positions, double* repulsiveForces, const double k, const int numNodes);
    static void calculateAttractiveForces(int* edges, int numEdges, const double* positions, double* attractiveForces,
    static     const double k, const int numNodes);
    static void applyForces(double* positions, double* attractiveForces, double* repulsiveForces,
        int numNodes, double temp);

};

