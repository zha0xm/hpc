#include <cuda_runtime.h>
#include "spmm_opt.h"

#define WARP_SIZE 32
#define TILE_NNZ  256

__global__ void spmm_shared_tile_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    __shared__ int   s_idx[TILE_NNZ];
    __shared__ float s_val[TILE_NNZ];

    int row  = blockIdx.x;
    if (row >= num_v) return;

    int lane = threadIdx.x;

    int row_beg = ptr[row];
    int row_end = ptr[row + 1];
    int nnz     = row_end - row_beg;

    #pragma unroll
    for (int f = lane; f < INFEATURE; f += WARP_SIZE) {
        vout[row * INFEATURE + f] = 0.0f;
    }

    for (int tile_base = 0; tile_base < nnz; tile_base += TILE_NNZ) {

        int tile_e = (nnz - tile_base) < TILE_NNZ ? (nnz - tile_base) : TILE_NNZ;

        for (int t = lane; t < tile_e; t += WARP_SIZE) {
            int g_e  = row_beg + tile_base + t;
            s_idx[t] = idx[g_e];
            s_val[t] = val[g_e];
        }
        __syncthreads();

        #pragma unroll
        for (int f = lane; f < INFEATURE; f += WARP_SIZE) {

            float acc = vout[row * INFEATURE + f];

            #pragma unroll
            for (int t = 0; t < tile_e; ++t) {
                acc = fmaf(__ldg(&vin[s_idx[t] * INFEATURE + f]), s_val[t], acc);
            }

            vout[row * INFEATURE + f] = acc;
        }
        __syncthreads();
    }
}


void SpMMOpt::preprocess(float*, float*)
{
    dim3 block(32, 1, 1);            // 一个 warp
    dim3 grid( (INFEATURE+31)/32, num_v );
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_shared_tile_kernel<<<grid, block>>>(
        d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}

