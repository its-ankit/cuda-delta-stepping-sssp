# GPU Single-Source Shortest Paths with Static and Adaptive ∆-Stepping

A CUDA implementation of **∆-stepping** for single-source shortest paths (SSSP) on large directed graphs, with two modes:

- **Static ∆**: a fixed bucket width given in the input.
- **Adaptive ∆**: the bucket width is recomputed at every step so that about **K** vertices are processed together, keeping the GPU busy without wasted work.

Built for CS6023 (GPU Programming), IIT Madras, Assignment 3.

---

## 1. The big picture

**Goal:** given a weighted directed graph and a source vertex, find the shortest distance from the source to every vertex.

The two classic algorithms sit at opposite extremes:

| | Dijkstra | Bellman-Ford |
|---|---|---|
| Order | one vertex at a time, smallest distance first | all edges, over and over |
| Wasted work | none | a lot (vertices updated many times) |
| Parallelism | almost none | huge |
| Good on a GPU? | no | fast, but wasteful |

**∆-stepping is the middle ground.** Vertices are grouped into **buckets** by distance, each ∆ wide:

```
bucket 0: distances [0, ∆]    bucket 1: (∆, 2∆]    bucket 2: (2∆, 3∆]   ...
```

Buckets are processed **in order** (like Dijkstra), but all the vertices **inside one bucket** are processed **in parallel** (like Bellman-Ford).

- ∆ = 1 → tiny buckets → behaves like Dijkstra
- ∆ very large → one huge bucket → behaves like Bellman-Ford

---

## 2. Graph format: CSR

The graph arrives in **Compressed Sparse Row** form, three flat arrays:

```
offsets[N+1]   neighs[E]   weights[E]
```

The outgoing edges of vertex `u` are at positions `offsets[u]` to `offsets[u+1] - 1`:

```
offsets = [0, 2, 4, 5, 7, 7, 7]
neighs  = [1, 2, 2, 3, 4, 4, 5]
weights = [2, 5, 1, 4, 3, 1, 2]

vertex 0: 0→1 (2), 0→2 (5)
vertex 1: 1→2 (1), 1→3 (4)
vertex 2: 2→4 (3)
vertex 3: 3→4 (1), 3→5 (2)
vertices 4, 5: no edges
```

CSR is a flattened adjacency list. Compared to an edge list `(src, dst, w)`, it needs `N + 1` numbers instead of `E` for the sources, and it answers "give me all edges of vertex `u`" instantly, which ∆-stepping needs for every vertex in a bucket. Three plain arrays also copy to the GPU with three `cudaMemcpy` calls.

---

## 3. The algorithm

### Light and heavy edges

An edge's type depends **only on its weight**, never on distances:

- **Light**: `w ≤ ∆`. Its target may land in the **same** bucket.
- **Heavy**: `w > ∆`. Its target always lands in a **later** bucket.

Each bucket is processed in two phases:

1. **Light phase (in rounds):** every vertex in the bucket relaxes its light edges. Targets that land in the same bucket join it, so we repeat until the bucket stops growing.
2. **Heavy phase (once):** every vertex settled in the bucket relaxes its heavy edges, sending targets to later buckets.

**Why delay heavy edges?** During the light phase a vertex's distance can still improve. Relaxing its heavy edges early would have to be redone. After the light phase its distance is final, so each heavy edge is relaxed once.

Example (∆ = 10): edges S→A (8), S→B (1), B→A (2), A→X (50).
Without the split, A relaxes A→X with distance 8 (X = 58), then B improves A to 3, and A→X must be relaxed again (X = 53). With the split, A→X is relaxed once, with the final distance 3.

### Near and Far queues

Distances can reach about 2 billion, so allocating one array per bucket is impossible. Since only the lowest bucket is ever active, two queues are enough:

- **Near**: the vertices of the current bucket (the window `[Dmin, cutoff]`).
- **Far**: every vertex beyond the window, all future buckets mixed together.

Far stores **only vertex IDs**. A vertex's bucket is always computed from its **current** distance, because the distance can improve after it was queued.

When Near is empty, the bucket is finished: find `Dmin` in Far, set the next window, and move the vertices inside it into Near.

A `queue_state[v]` array (`NONE`, `NEAR`, `FAR`) makes sure a vertex is never inserted twice into the same queue when many threads improve it at once.

### Static vs adaptive window

- **Static:** `cutoff = Dmin + ∆`.
- **Adaptive:** sort Far by distance and choose the window so that about K vertices fit:

```
Dmin     = keys[0]
target   = min(K - 1, |Far| - 1)
Dcutoff  = keys[target]
∆raw     = Dcutoff - Dmin          → printed to the output log
promote every vertex with dist ≤ Dcutoff   (thrust::upper_bound)
```

### The four adaptive "catches"

The printed ∆ sequence must match the reference exactly, which depends on four details:

1. **Re-anchor `Dmin`** to the smallest key in Far every time, never to the previous cutoff. Otherwise empty distance gaps inflate ∆.
2. **∆ = 0 plateaus:** print the raw ∆ (even 0), but use `activeDelta = 1` internally so zero-width windows still make progress.
3. **Right-inclusive extraction:** promote **all** vertices with `dist ≤ Dcutoff`, not just K of them, so tied distances are cleared in one pass.
4. **Ghost pruning:** a Far entry whose vertex was later pulled into Near and settled is a *ghost*. Remove every entry with `dist ≤ last cutoff` before computing `Dmin`.

---

## 4. GPU design

### Kernels

| Kernel | One thread per | Job |
|---|---|---|
| `initKernel` | vertex | `dist = INF` (source = 0), clear flags |
| `beginRoundKernel` | Near vertex | take it out of Near, record it once in the bucket list |
| `lightRelaxKernel` | Near vertex | relax light edges, push improved targets to next Near or Far |
| `heavyRelaxKernel` | settled vertex | relax heavy edges, push targets to Far |
| `endBucketKernel` | settled vertex | reset bucket flags |
| `markNearKernel` | promoted vertex | mark vertices moved from Far as `NEAR` |

The Far refill (prune, sort, search, partition) uses **Thrust**: `remove_if`, `transform`, `sort_by_key`, `upper_bound`, `transform_reduce`, `partition`.

### Concurrency

- **`atomicMin(&dist[v], newDist)`**: many threads may improve the same vertex at once. `atomicMin` keeps the smallest value, and its return value tells the thread whether it actually improved `v`.
- **Queue insertion:** `atomicExch(&state[v], NEAR)` and `atomicCAS(&state[v], NONE, FAR)` let exactly one thread insert a vertex; `atomicAdd` on a counter gives it a unique slot.
- **Global barrier = kernel boundary.** `__syncthreads()` only synchronises one block, so each light round is a separate kernel launch; the host reads the next-round size and relaunches.
- **Deterministic ∆ log.** The printed ∆ depends only on the set of Far vertices and their final distances at each refill, not on the order in which threads ran, so the trace is identical on every run.

### Memory

All buffers (`dist`, `state`, Near, next Near, Far, Far scratch, sort keys, bucket list, counters) are allocated **once** in `main()` and reused for every source, as the assignment requires. Each is at most N ints, since every vertex can sit in each queue at most once.

---

## 5. Repository layout

```
src/delta_stepping.cu   final solution: static + adaptive ∆-stepping
src/bellman_ford.cu     practice version: parallel Bellman-Ford with atomicMin
```

`bellman_ford.cu` was the warm-up: it introduces CSR on the GPU, one thread per vertex, `atomicMin`, and a host loop that relaunches until nothing changes. It produces correct distances but no ∆ log, and does far more redundant work.

## 6. How to run

Requires an NVIDIA GPU and the CUDA Toolkit (Thrust ships with it). On Google Colab, choose a T4 GPU runtime.

```bash
nvcc -O3 -arch=sm_75 src/delta_stepping.cu -o main
./main input.txt output.txt
```

### Input

```
<delta> <K>            static: delta >= 0, K = -1     adaptive: delta = -1, K > 0
<N> <E> <S>
<src_0> ... <src_S-1>
<offsets[0..N]>
<neighs[0..E-1]>
<weights[0..E-1]>
```

### Output

For each source: the source on one line, then (adaptive mode only) one line per printed ∆, then N lines of distances. Unreachable vertices print `2147483647`.

### Constraints

N ≤ 2,000,000, E ≤ 300,000,000, edge weights 0 to 1000, S ≤ 20. The longest possible path is about 2 × 10⁹, so 32-bit distances never overflow and `INT_MAX` is safe as the "unreachable" value.
