// ---------------- apsp_opt.cu ----------------
#include <cuda.h>
#include <cstdio>

#define INF 1000000000
#define B   32                 // tile size (compile‑time constant)
#define PAD (B + 1)            // row padding: B*PAD = 1056 ints = 4224 B

//----------------------------------------------------------------------------//
// 内部工具：全局坐标 / 越界判断
//----------------------------------------------------------------------------//
__device__ __forceinline__
int idx(int i, int j, int n) { return i * n + j; }

__device__ __forceinline__
bool in(int i, int n) { return i < n; }

//----------------------------------------------------------------------------//
// kernel‑0 : 处理 pivot 块 G(p,p)
//----------------------------------------------------------------------------//
__global__ void k_pivot(int *g, int n, int p)
{
    __shared__ int tile[B * PAD];      // 32×32，行 padding 1，避免列冲突
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gi = p * B + ty;
    const int gj = p * B + tx;

    // 1. 载入到 shared
    tile[ty * PAD + tx] = (in(gi, n) && in(gj, n)) ?
                          g[idx(gi, gj, n)] : INF;
    __syncthreads();                   // 确保全 tile 可用

#pragma unroll
    for (int k = 0; k < B; ++k) {
        int tmp = tile[ty * PAD + k] + tile[k * PAD + tx];
        if (tmp < tile[ty * PAD + tx])
            tile[ty * PAD + tx] = tmp;
        // 行/列 k 本轮不会被真正改变 → 可以安全省去同步
    }

    if (in(gi, n) && in(gj, n))
        g[idx(gi, gj, n)] = tile[ty * PAD + tx];
}

//----------------------------------------------------------------------------//
// kernel‑1 : 处理 pivot 所在行块和列块
// 一个 blockIdx.y==0 表示“行块”，blockIdx.y==1 表示“列块”
//----------------------------------------------------------------------------//
__global__ void k_row_col(int *g, int n, int p)
{
    __shared__ int piv[B * PAD];           // G(p,p)
    __shared__ int tgt[B * PAD];           // G(p,q) 或 G(q,p)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int q  = blockIdx.x;             // 目标块序号
    const bool is_row = (blockIdx.y == 0);

    if (q == p) return;                    // 跳过对角

    // ---------------- ① 载入 pivot ----------------
    const int pi = p * B + ty;
    const int pj = p * B + tx;
    piv[ty * PAD + tx] = (in(pi, n) && in(pj, n)) ?
                         g[idx(pi, pj, n)] : INF;

    // ---------------- ② 载入目标块 ----------------
    int gi = is_row ? p * B + ty : q * B + ty;
    int gj = is_row ? q * B + tx : p * B + tx;
    tgt[ty * PAD + tx] = (in(gi, n) && in(gj, n)) ?
                         g[idx(gi, gj, n)] : INF;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < B; ++k) {
        int val = is_row ?
                  piv[ty * PAD + k] + tgt[k * PAD + tx] :
                  tgt[ty * PAD + k] + piv[k * PAD + tx];

        if (val < tgt[ty * PAD + tx])
            tgt[ty * PAD + tx] = val;
    }

    if (in(gi, n) && in(gj, n))
        g[idx(gi, gj, n)] = tgt[ty * PAD + tx];
}

//----------------------------------------------------------------------------//
// kernel‑2 : 处理其余块 G(i,j), i!=p, j!=p
//----------------------------------------------------------------------------//
__global__ void k_outer(int *g, int n, int p)
{
    __shared__ int A[B * PAD];    // G(i,p)
    __shared__ int Bk[B * PAD];   // G(p,j)
    __shared__ int C[B * PAD];    // G(i,j)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int bi = blockIdx.y;
    const int bj = blockIdx.x;

    if (bi == p || bj == p) return;

    //------------------- ① load tiles -------------------
    int gi = bi * B + ty;
    int gp = p  * B + tx;
    A[ty * PAD + tx] = (in(gi, n) && in(gp, n)) ?
                       g[idx(gi, gp, n)] : INF;

    int gp2 = p  * B + ty;
    int gj  = bj * B + tx;
    Bk[ty * PAD + tx] = (in(gp2, n) && in(gj, n)) ?
                        g[idx(gp2, gj, n)] : INF;

    int gj2 = bj * B + tx;
    C[ty * PAD + tx]  = (in(gi, n) && in(gj2, n)) ?
                        g[idx(gi, gj2, n)] : INF;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < B; ++k) {
        int val = A[ty * PAD + k] + Bk[k * PAD + tx];
        if (val < C[ty * PAD + tx])
            C[ty * PAD + tx] = val;
    }

    if (in(gi, n) && in(gj2, n))
        g[idx(gi, gj2, n)] = C[ty * PAD + tx];
}

//----------------------------------------------------------------------------//
// Host 端接口
//----------------------------------------------------------------------------//
void apsp(int n, /*device*/ int *g, cudaStream_t stream = 0)
{
    const dim3 threads(B, B);

    // grid0 只有一个 block
    const dim3 grid_rowcol((n + B - 1) / B, 2);            // x = num_blocks, y= {row,col}
    const dim3 grid_outer((n + B - 1) / B, (n + B - 1) / B);

    for (int p = 0, nb = (n + B - 1) / B; p < nb; ++p) {
        k_pivot   <<< 1,         threads, B * PAD * sizeof(int), stream >>>(g, n, p);
        k_row_col <<< grid_rowcol, threads, 2 * B * PAD * sizeof(int), stream >>>(g, n, p);
        k_outer   <<< grid_outer,  threads, 3 * B * PAD * sizeof(int), stream >>>(g, n, p);
    }
    cudaDeviceSynchronize();   // 可按需去掉
}
// ------------------------------------------------------
