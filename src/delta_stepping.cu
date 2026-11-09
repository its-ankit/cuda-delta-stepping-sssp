#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.

// Queue membership of each vertex (prevents duplicate insertions)
#define STATE_NONE 0
#define STATE_NEAR 1
#define STATE_FAR  2

// Device-side counters, stored together in one small array
#define CNT_NEXT   0   // size of the next Near round
#define CNT_FAR    1   // size of the Far queue
#define CNT_BUCKET 2   // vertices processed in the current bucket
#define NUM_COUNTERS 3

static inline int gridFor(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }

// ============================================================================
// CUDA KERNELS
// ============================================================================

// dist = INF everywhere except the source; all queue flags cleared
__global__ void initKernel(int N, int source, int *dist, int *state, int *inBucket)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    dist[v] = (v == source) ? 0 : INF;
    state[v] = (v == source) ? STATE_NEAR : STATE_NONE;
    inBucket[v] = 0;
}

// Mark vertices just promoted from Far as members of Near
__global__ void markNearKernel(const int *nearQ, int nearCount, int *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nearCount) state[nearQ[i]] = STATE_NEAR;
}

// Start of a light round: take each Near vertex out of the queue (so it can be
// re-inserted if it improves again) and record it once in the bucket list,
// which the heavy phase uses later.
__global__ void beginRoundKernel(const int *nearQ, int nearCount, int *state,
                                 int *inBucket, int *bucket, int *counters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nearCount) return;
    int u = nearQ[i];
    state[u] = STATE_NONE;
    if (atomicExch(&inBucket[u], 1) == 0)
        bucket[atomicAdd(&counters[CNT_BUCKET], 1)] = u;
}

// Relax one improved vertex v: inside the current window -> next Near round,
// beyond it -> Far queue (each at most once, thanks to the state array)
__device__ void pushImproved(int v, int newDist, int cutoff, int *state,
                             int *nextQ, int *farQ, int *counters)
{
    if (newDist <= cutoff)
    {
        if (atomicExch(&state[v], STATE_NEAR) != STATE_NEAR)
            nextQ[atomicAdd(&counters[CNT_NEXT], 1)] = v;
    }
    else
    {
        if (atomicCAS(&state[v], STATE_NONE, STATE_FAR) == STATE_NONE)
            farQ[atomicAdd(&counters[CNT_FAR], 1)] = v;
    }
}

// Light phase: one thread per Near vertex relaxes its light edges (w <= delta)
__global__ void lightRelaxKernel(const int *nearQ, int nearCount,
                                 const int *offsets, const int *neighs, const int *weights,
                                 int *dist, int *state, int activeDelta, int cutoff,
                                 int *nextQ, int *farQ, int *counters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nearCount) return;
    int u = nearQ[i];
    int d = dist[u];

    for (int e = offsets[u]; e < offsets[u + 1]; e++)
    {
        int w = weights[e];
        if (w > activeDelta) continue;                 // heavy edge: later
        int v = neighs[e];
        int newDist = d + w;
        if (newDist >= dist[v]) continue;              // cheap pre-check
        if (newDist < atomicMin(&dist[v], newDist))    // really improved
            pushImproved(v, newDist, cutoff, state, nextQ, farQ, counters);
    }
}

// Heavy phase: once per bucket, every vertex settled in it relaxes its heavy
// edges (w > delta). Targets always land beyond the window, i.e. in Far.
__global__ void heavyRelaxKernel(const int *bucket, int bucketCount,
                                 const int *offsets, const int *neighs, const int *weights,
                                 int *dist, int *state, int activeDelta, int cutoff,
                                 int *nextQ, int *farQ, int *counters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bucketCount) return;
    int u = bucket[i];
    int d = dist[u];

    for (int e = offsets[u]; e < offsets[u + 1]; e++)
    {
        int w = weights[e];
        if (w <= activeDelta) continue;                // light edge: done already
        int v = neighs[e];
        int newDist = d + w;
        if (newDist >= dist[v]) continue;
        if (newDist < atomicMin(&dist[v], newDist))
            pushImproved(v, newDist, cutoff, state, nextQ, farQ, counters);
    }
}

// Clear the "already in this bucket" flags for the next bucket
__global__ void endBucketKernel(const int *bucket, int bucketCount, int *inBucket)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bucketCount) inBucket[bucket[i]] = 0;
}

// ============================================================================
// THRUST FUNCTORS
// ============================================================================

struct DistOf      // key of a Far entry = its current tentative distance
{
    const int *dist;
    __host__ __device__ int operator()(int v) const { return dist[v]; }
};

struct AtMost      // dist[v] <= limit  (ghost test / window test)
{
    const int *dist;
    int limit;
    __host__ __device__ bool operator()(int v) const { return dist[v] <= limit; }
};

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

struct Workspace   // all device buffers, allocated once in main()
{
    int *state, *inBucket;          // per-vertex flags
    int *nearQ, *nextQ;             // current and next Near round
    int *farQ, *farTmp, *keys;      // Far queue, scratch copy, sort keys
    int *bucket;                    // vertices settled in the current bucket
    int *counters;                  // CNT_NEXT, CNT_FAR, CNT_BUCKET
};

static int readCounter(const int *counters, int which)
{
    int value;
    cudaMemcpy(&value, counters + which, sizeof(int), cudaMemcpyDeviceToHost);
    return value;
}

static void writeCounter(int *counters, int which, int value)
{
    cudaMemcpy(counters + which, &value, sizeof(int), cudaMemcpyHostToDevice);
}

static int readInt(const int *ptr)
{
    int value;
    cudaMemcpy(&value, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    return value;
}

static int clampedAdd(int a, int b)   // a + b without overflowing past INF - 1
{
    long long s = (long long)a + b;
    return s > INF - 1 ? INF - 1 : (int)s;
}

void run_delta_stepping_single_source(
    int N, int source, int delta_mode, int K,
    const int *d_offsets, const int *d_neighs, const int *d_weights,
    int *d_dist, Workspace &ws, std::ofstream &outfile)
{
    const bool adaptive = (delta_mode == -1);

    initKernel<<<gridFor(N), BLOCK_SIZE>>>(N, source, d_dist, ws.state, ws.inBucket);
    cudaMemset(ws.counters, 0, NUM_COUNTERS * sizeof(int));

    // The source starts alone in Near, in the window [0, cutoff]
    cudaMemcpy(ws.nearQ, &source, sizeof(int), cudaMemcpyHostToDevice);
    int nearCount   = 1;
    int cutoff      = adaptive ? 0 : clampedAdd(0, delta_mode);
    int activeDelta = adaptive ? 1 : delta_mode;

    while (true)
    {
        // -------- Phase 3: Near is empty -> refill it from Far --------
        if (nearCount == 0)
        {
            int farCount = readCounter(ws.counters, CNT_FAR);

            // drop ghosts: entries already settled in an earlier bucket
            farCount = thrust::remove_if(thrust::device, ws.farQ, ws.farQ + farCount,
                                         AtMost{d_dist, cutoff}) - ws.farQ;
            if (farCount == 0) break;

            int promoted;
            if (adaptive)
            {
                // sort Far by distance, pick the window so that ~K vertices fit
                thrust::transform(thrust::device, ws.farQ, ws.farQ + farCount,
                                  ws.keys, DistOf{d_dist});
                thrust::sort_by_key(thrust::device, ws.keys, ws.keys + farCount, ws.farQ);

                int Dmin   = readInt(ws.keys);
                int target = std::min(K - 1, farCount - 1);
                int Dcut   = readInt(ws.keys + target);
                int rawDelta = Dcut - Dmin;

                outfile << rawDelta << "\n";                 // adaptive delta log
                activeDelta = (rawDelta == 0) ? 1 : rawDelta;
                cutoff = Dcut;

                // right-inclusive: take every vertex with dist <= Dcut
                promoted = thrust::upper_bound(thrust::device, ws.keys,
                                               ws.keys + farCount, Dcut) - ws.keys;
            }
            else
            {
                // static: window is [Dmin, Dmin + delta]
                int Dmin = thrust::transform_reduce(thrust::device, ws.farQ,
                                                    ws.farQ + farCount, DistOf{d_dist},
                                                    INF, thrust::minimum<int>());
                cutoff = clampedAdd(Dmin, delta_mode);
                promoted = thrust::partition(thrust::device, ws.farQ, ws.farQ + farCount,
                                             AtMost{d_dist, cutoff}) - ws.farQ;
            }

            // first `promoted` entries -> Near, the rest stay in Far
            cudaMemcpy(ws.nearQ, ws.farQ, promoted * sizeof(int), cudaMemcpyDeviceToDevice);
            int remaining = farCount - promoted;
            cudaMemcpy(ws.farTmp, ws.farQ + promoted, remaining * sizeof(int),
                       cudaMemcpyDeviceToDevice);
            std::swap(ws.farQ, ws.farTmp);
            writeCounter(ws.counters, CNT_FAR, remaining);

            nearCount = promoted;
            markNearKernel<<<gridFor(nearCount), BLOCK_SIZE>>>(ws.nearQ, nearCount, ws.state);
        }

        // -------- Phase 1: light edges, in rounds until Near stops growing --------
        writeCounter(ws.counters, CNT_BUCKET, 0);
        while (nearCount > 0)
        {
            beginRoundKernel<<<gridFor(nearCount), BLOCK_SIZE>>>(
                ws.nearQ, nearCount, ws.state, ws.inBucket, ws.bucket, ws.counters);

            writeCounter(ws.counters, CNT_NEXT, 0);
            lightRelaxKernel<<<gridFor(nearCount), BLOCK_SIZE>>>(
                ws.nearQ, nearCount, d_offsets, d_neighs, d_weights, d_dist, ws.state,
                activeDelta, cutoff, ws.nextQ, ws.farQ, ws.counters);

            nearCount = readCounter(ws.counters, CNT_NEXT);
            std::swap(ws.nearQ, ws.nextQ);
        }

        // -------- Phase 2: heavy edges, once per bucket --------
        int bucketCount = readCounter(ws.counters, CNT_BUCKET);
        if (bucketCount > 0)
        {
            heavyRelaxKernel<<<gridFor(bucketCount), BLOCK_SIZE>>>(
                ws.bucket, bucketCount, d_offsets, d_neighs, d_weights, d_dist, ws.state,
                activeDelta, cutoff, ws.nextQ, ws.farQ, ws.counters);
            endBucketKernel<<<gridFor(bucketCount), BLOCK_SIZE>>>(
                ws.bucket, bucketCount, ws.inBucket);
        }
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
    {
        infile >> sources[i];
    }

    int *offsets = new int[ N + 1 ] { 0 };
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> weights[i];
    }

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[ N ] { 0 }; // sssp distance arry. tent means tentative distance
    int *d_tent = nullptr;
    cudaMalloc(&d_tent, N * sizeof(int));

    // All working buffers are allocated once and reused for every source
    Workspace ws;
    cudaMalloc(&ws.state,    N * sizeof(int));
    cudaMalloc(&ws.inBucket, N * sizeof(int));
    cudaMalloc(&ws.nearQ,    N * sizeof(int));
    cudaMalloc(&ws.nextQ,    N * sizeof(int));
    cudaMalloc(&ws.farQ,     N * sizeof(int));
    cudaMalloc(&ws.farTmp,   N * sizeof(int));
    cudaMalloc(&ws.keys,     N * sizeof(int));
    cudaMalloc(&ws.bucket,   N * sizeof(int));
    cudaMalloc(&ws.counters, NUM_COUNTERS * sizeof(int));

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];

        outfile << source << "\n";

        run_delta_stepping_single_source(N, source, delta_mode, K,
                                         d_offsets, d_neighs, d_weights,
                                         d_tent, ws, outfile);

        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }
    }

    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_tent);
    cudaFree(ws.state);
    cudaFree(ws.inBucket);
    cudaFree(ws.nearQ);
    cudaFree(ws.nextQ);
    cudaFree(ws.farQ);
    cudaFree(ws.farTmp);
    cudaFree(ws.keys);
    cudaFree(ws.bucket);
    cudaFree(ws.counters);

    delete[] offsets;
    delete[] neighs;
    delete[] weights;
    delete[] h_tent;

    outfile.close();
    return 0;
}
