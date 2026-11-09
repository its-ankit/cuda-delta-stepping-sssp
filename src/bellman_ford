#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Set every distance to INF, and the source to 0
__global__ void initDist(int N, int *dist, int source)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N)
        dist[v] = (v == source) ? 0 : INF;
}

// One round of Bellman-Ford: thread `node` relaxes all its outgoing edges
__global__ void relaxKernel(int N, const int *offsets, const int *neighs,
                            const int *weights, int *dist, int *changed)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N)
        return;

    int d = dist[node];
    if (d == INF)
        return;

    for (int i = offsets[node]; i < offsets[node + 1]; i++)
    {
        int neigh = neighs[i];
        int newDist = d + weights[i];
        if (newDist < dist[neigh])
        {
            atomicMin(&dist[neigh], newDist);
            *changed = 1;
        }
    }
}

// ============================================================================
// SINGLE SOURCE DRIVER (Bellman-Ford, for practice)
// ============================================================================

void run_bellman_ford_single_source(int N, int source,
                                    const int *d_offsets, const int *d_neighs,
                                    const int *d_weights, int *d_dist, int *d_changed)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // reset distances for this source
    initDist<<<grid_size, BLOCK_SIZE>>>(N, d_dist, source);

    int h_changed = 1;
    while (h_changed)
    {
        cudaMemset(d_changed, 0, sizeof(int));               // reset flag on the host side
        relaxKernel<<<grid_size, BLOCK_SIZE>>>(N, d_offsets, d_neighs,
                                               d_weights, d_dist, d_changed);
        cudaMemcpy(&h_changed, d_changed, sizeof(int),       // kernel end = barrier
                   cudaMemcpyDeviceToHost);
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Unable to open output file " << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;
    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
        infile >> sources[i];

    int *offsets = new int[N + 1]{0};
    for (int i = 0; i <= N; ++i)
        infile >> offsets[i];

    int *neighs = new int[E]{0};
    for (int i = 0; i < E; ++i)
        infile >> neighs[i];

    int *weights = new int[E]{0};
    for (int i = 0; i < E; ++i)
        infile >> weights[i];

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    // distances: CPU copy (for printing) and GPU copy (the algorithm updates this)
    int *h_dist = new int[N];
    int *d_dist = nullptr;
    cudaMalloc(&d_dist, N * sizeof(int));

    // "something changed" flag, allocated once
    int *d_changed = nullptr;
    cudaMalloc(&d_changed, sizeof(int));

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];
        outfile << source << "\n";

        run_bellman_ford_single_source(N, source, d_offsets, d_neighs,
                                       d_weights, d_dist, d_changed);

        cudaMemcpy(h_dist, d_dist, N * sizeof(int), cudaMemcpyDeviceToHost);
        for (int v = 0; v < N; ++v)
            outfile << h_dist[v] << "\n";
    }

    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_changed);

    delete[] offsets;
    delete[] neighs;
    delete[] weights;
    delete[] h_dist;

    outfile.close();
    return 0;
}
