/**********************************************************************
 *  spmm_opt.cu  —— one-kernel version (row × feature-tile 2-D grid)
 *********************************************************************/
#include <cuda_runtime.h>
#include "spmm_opt.h"

/* ---------- 可调常量 ---------- */
#define WARP_SIZE 32
#define TILE_NNZ  256          /* shared-mem tile size for (idx,val) */

/* ------------------------------------------------------------------ */
/*  Kernel：blockDim.x = 32 (1 warp) ;  grid = (ceil(F/32), num_v)    */
/* ------------------------------------------------------------------ */
__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    /* -------- shared memory 缓存本行的 (idx,val) -------- */
    __shared__ int   s_idx[TILE_NNZ];
    __shared__ float s_val[TILE_NNZ];

    /* -------- 行号、特征号计算 -------- */
    const int row  = blockIdx.y;
    if (row >= num_v) return;

    const int feat = blockIdx.x * WARP_SIZE + threadIdx.x;
    if (feat >= INFEATURE) return;              /* 网格已确保越界线程极少 */

    /* -------- 本行 CSR 范围 -------- */
    const int row_beg = ptr[row];
    const int nnz     = ptr[row + 1] - row_beg;

    float acc = 0.f;

    /* -------- 分批把 (idx,val) 搬进 shared memory -------- */
    for (int base = 0; base < nnz; base += TILE_NNZ) {
        int tile_e = (nnz - base < TILE_NNZ) ? (nnz - base) : TILE_NNZ;

        /* stride-copy：warp 32 线程协作加载 */
        for (int t = threadIdx.x; t < tile_e; t += WARP_SIZE) {
            int e = row_beg + base + t;
            s_idx[t] = idx[e];
            s_val[t] = val[e];
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < tile_e; ++t)
            acc = fmaf(__ldg(&vin[s_idx[t] * INFEATURE + feat]),  // 输入
                       s_val[t],                                // 权重
                       acc);                                    // FMA

        __syncthreads();
    }

    vout[row * INFEATURE + feat] = acc;
}

/* ------------------------------------------------------------------ */
/*  SpMMOpt::preprocess  &  run                                       */
/* ------------------------------------------------------------------ */
void SpMMOpt::preprocess(float*, float*)
{
    /* 1 warp / CTA                                                      */
    block = dim3(WARP_SIZE, 1, 1);

    /* 2-D grid：x = feature tiles，y = 行                               */
    grid  = dim3((feat_in + WARP_SIZE - 1) / WARP_SIZE,  /* tile 数   */
                 num_v,                                   /* 行数     */
                 1);

    /* 若 num_v 或 grid.x 超过 65 535，可改用 3-D grid，或把行切块。 */
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel<<<grid, block>>>(
        d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
