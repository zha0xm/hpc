#define WARP_SIZE 32
#define TILE_NNZ  256       /* 1 tile 搬多少条非零，可调 64–1024 */

#include <cuda_runtime.h>
#include "spmm_opt.h"

__global__ void spmm_shared_tile_kernel(const int   *__restrict__ ptr,
                                        const int   *__restrict__ idx,
                                        const float *__restrict__ val,
                                        const float *__restrict__ vin,
                                        float       *__restrict__ vout,
                                        int   num_rows,
                                        int   in_feat)
{
    __shared__ int   s_idx[TILE_NNZ];
    __shared__ float s_val[TILE_NNZ];

    int row  = blockIdx.x;              // 一行⇔一 block
    if (row >= num_rows) return;

    int lane = threadIdx.x;             // 0-31

    int row_beg = ptr[row];
    int row_end = ptr[row + 1];
    int nnz     = row_end - row_beg;

    /* ---------- 初始化输出寄存器 ---------- */
#pragma unroll
    for (int f = lane; f < in_feat; f += WARP_SIZE) {
        vout[row * in_feat + f] = 0.0f;
    }

    /* ---------- tile 级循环处理本行所有非零 ---------- */
    for (int tile_base = 0; tile_base < nnz; tile_base += TILE_NNZ) {

        /* 本 tile 实际包含多少条边 */
        int tile_e = (nnz - tile_base) < TILE_NNZ ? (nnz - tile_base) : TILE_NNZ;

        /* ① Stride-copy 把 (idx,val) 搬进 shared memory */
        for (int t = lane; t < tile_e; t += WARP_SIZE) {
            int g_e  = row_beg + tile_base + t;
            s_idx[t] = idx[g_e];
            s_val[t] = val[g_e];
        }
        __syncthreads();   /* 共享内存准备就绪 */

        /* ② 计算：复用刚缓存的 tile */
#pragma unroll
        for (int f = lane; f < in_feat; f += WARP_SIZE) {

            float acc = vout[row * in_feat + f];   // 取回寄存器累加器

#pragma unroll
            for (int t = 0; t < tile_e; ++t) {
                acc = fmaf(__ldg(&vin[s_idx[t] * in_feat + f]),   // vin
                           s_val[t],                              // w
                           acc);                                  // FMA
            }

            vout[row * in_feat + f] = acc;        // 写回结果
        }
        __syncthreads();   /* 同步后进入下一 tile */
    }
}

/* =========================================================
 *  SpMMOpt::preprocess  &  run
 * =======================================================*/
void SpMMOpt::preprocess(float*, float*)
{
    block = dim3(WARP_SIZE, 1, 1);      // 32 线程 = 1 warp
    grid  = dim3(num_v, 1, 1);          // 一行一个 block
    /* 若 num_v > 65535，可改 2D grid：grid.x=65535, grid.y=(num_v+65534)/65535 */
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_shared_tile_kernel<<<grid, block>>>(
        d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}

