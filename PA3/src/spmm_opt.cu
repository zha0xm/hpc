/**********************************************************************
 *  spmm_opt.cu — single kernel 2-D grid  (row × 32-feature tile)
 *********************************************************************/
#include <cuda_runtime.h>
#include "spmm_opt.h"

/* -------- 常量 -------- */
#define WARP_SIZE 32
#define TILE_NNZ  256          /* shared-mem tile size for (idx,val) */

/* -------- kernel 前向声明 -------- */
extern "C"
__global__ void spmm_kernel(int *ptr, int *idx, float *val,
                            float *vin, float *vout,
                            int num_rows, int in_feat);

/* -------- kernel 定义 -------- */
__global__ void spmm_kernel(int *ptr, int *idx, float *val,
                            float *vin, float *vout,
                            int num_rows, int in_feat)
{
    __shared__ int   s_idx[TILE_NNZ];
    __shared__ float s_val[TILE_NNZ];

    const int row  = blockIdx.y;
    if (row >= num_rows) return;           /* 整个 block 都无效，安全退出 */

    const int lane = threadIdx.x;          /* 0–31 */
    const int feat = blockIdx.x * WARP_SIZE + lane;
    const bool valid = (feat < in_feat);   /* 该线程负责的特征是否合法 */

    /* CSR 行范围 */
    const int row_beg = ptr[row];
    const int row_end = ptr[row + 1];
    const int nnz     = row_end - row_beg;

    float acc = 0.f;

    /* -------- tile 循环 -------- */
    for (int base = 0; base < nnz; base += TILE_NNZ) {

        int tile_e = (nnz - base < TILE_NNZ) ? (nnz - base) : TILE_NNZ;

        /* stride-copy (idx,val) → shared */
        for (int t = lane; t < tile_e; t += WARP_SIZE) {
            int e      = row_beg + base + t;
            s_idx[t]   = idx[e];
            s_val[t]   = val[e];
        }
        __syncthreads();

        if (valid) {
#pragma unroll 4
            for (int t = 0; t < tile_e; ++t)
                acc = fmaf(__ldg(&vin[s_idx[t] * in_feat + feat]),
                           s_val[t], acc);
        }
        __syncthreads();
    }

    if (valid)
        vout[row * in_feat + feat] = acc;
}

/* -------- SpMMOpt::preprocess & run -------- */
void SpMMOpt::preprocess(float*, float*)
{
    block = dim3(WARP_SIZE, 1, 1);                      /* 1 warp / CTA  */
    grid  = dim3((feat_in + WARP_SIZE - 1) / WARP_SIZE, /* feature tiles */
                 num_v, 1);                             /* rows          */
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel<<<grid, block>>>(
        d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
