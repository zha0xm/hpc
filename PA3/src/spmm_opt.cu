// =========  常量定义一定要放在最前面  =========
#ifndef WARP_SIZE
#define WARP_SIZE 32          // 一个 warp 的大小
#endif

#ifndef TILE_NNZ
#define TILE_NNZ 256          // 每次搬进 shared memory 的非零数
#endif
// =============================================

#include <cuda_runtime.h>     // 保证 __ldg() 有声明
#include "spmm_opt.h"

/*----------------------------------------------------------
 *  每个 block (=1 warp=32 线程) 处理 1 行
 *  把 (idx,val) 分批缓存进 shared memory
 *---------------------------------------------------------*/
__global__ void spmm_shared_tile_kernel(const int   *__restrict__ ptr,
                                        const int   *__restrict__ idx,
                                        const float *__restrict__ val,
                                        const float *__restrict__ vin,
                                        float       *__restrict__ vout,
                                        int   num_rows,
                                        int   in_feat)
{
    /* -------- static shared memory -------- */
    __shared__ int   s_idx[TILE_NNZ];
    __shared__ float s_val[TILE_NNZ];

    const int row  = blockIdx.x;      // 一行一个 block
    if (row >= num_rows) return;

    const int lane = threadIdx.x;     // 0–31

    const int row_start = ptr[row];
    const int row_end   = ptr[row + 1];
    const int nnz       = row_end - row_start;

    /* ---------- 初始化本行输出 ---------- */
#pragma unroll 4
    for (int f = lane; f < in_feat; f += WARP_SIZE)
        vout[row * in_feat + f] = 0.0f;

    /* ---------- 分批缓存 (idx,val) 并计算 ---------- */
    for (int tile_base = 0; tile_base < nnz; tile_base += TILE_NNZ)
    {
        int local_e = tile_base + lane;          // warp 每线程搬 1 edge
        if (local_e < nnz)
        {
            int g_e = row_start + local_e;
            s_idx[local_e - tile_base] = idx[g_e];
            s_val[local_e - tile_base] = val[g_e];
        }
        __syncthreads();                         // 等待共享内存写完

        int tile_e_end = min(TILE_NNZ, nnz - tile_base);

#pragma unroll 4                                // 展开 feature 循环
        for (int f = lane; f < in_feat; f += WARP_SIZE)
        {
            float acc = vout[row * in_feat + f]; // 取寄存器累加器

#pragma unroll 4                                // 展开 tile 内 edge 循环
            for (int t = 0; t < tile_e_end; ++t)
            {
                int   col = s_idx[t];
                float w   = s_val[t];
                acc += __ldg(&vin[col * in_feat + f]) * w;
            }
            vout[row * in_feat + f] = acc;       // 写回寄存器
        }
        __syncthreads();                         // 下一 tile 前同步
    }
}

/*------------------- SpMMOpt 成员函数 -------------------*/
void SpMMOpt::preprocess(float *vin, float *vout)
{
    block = dim3(WARP_SIZE, 1, 1);   // 32 threads = 1 warp
    grid  = dim3(num_v, 1, 1);       // 一行一个 block
    // 若 num_v > 65535，可用 2D grid：grid.x = 65535, grid.y = ...
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_shared_tile_kernel<<<grid, block>>>(
        d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
