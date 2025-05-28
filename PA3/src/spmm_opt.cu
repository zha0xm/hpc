#include "spmm_opt.h"

__global__ void spmm_warp_per_row(const int   *__restrict__ ptr,
                                  const int   *__restrict__ idx,
                                  const float *__restrict__ val,
                                  const float *__restrict__ vin,
                                  float       *__restrict__ vout,
                                  int num_rows, int in_feat)
{
    const int row   = blockIdx.x;           // 1 block 对应 1 行
    if (row >= num_rows) return;

    const int lane  = threadIdx.x;          // 0-31
    const int row_start = ptr[row];
    const int row_end   = ptr[row + 1];

    // 寄存器累加器：每个线程负责 in_feat 中第 lane, lane+32, lane+64... 的列
    for (int f = lane; f < in_feat; f += 32) {
        float acc = 0.f;

        // 遍历该行所有非零
        for (int e = row_start; e < row_end; ++e) {
            int   col = idx[e];
            float w   = val[e];
            acc += __ldg(&vin[col * in_feat + f]) * w;   // coalesced
        }
        vout[row * in_feat + f] = acc;
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    dim3 block(32);                     // 固定 32 threads = 1 warp
    dim3 grid(num_v);                   // 一个 block 对应一行
    this->block = block;
    this->grid  = grid;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_warp_per_row<<<grid, block>>>(d_ptr, d_idx, d_val,
                                       vin, vout, num_v, feat_in);
}
