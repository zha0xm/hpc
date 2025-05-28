#include "spmm_opt.h"
#include <vector>

// 设备端 eid -> row 映射表
__device__ __managed__ int* eid_to_row = nullptr;

__global__ void spmm_kernel_atomic(
    int* idx, float* val,
    float* vin, float* vout,
    int* eid_to_row,
    int num_e, int feat_in)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_e) return;

    int row = eid_to_row[eid];
    int col = idx[eid];
    float a_val = val[eid];

    for (int j = 0; j < feat_in; ++j) {
        float contrib = a_val * vin[col * feat_in + j];
        atomicAdd(&vout[row * feat_in + j], contrib);
    }
}

void SpMMOpt::preprocess(float* vin, float* vout)
{
    // 设置 grid / block 大小
    int BLOCK_SIZE = 256;
    grid.x = (num_e + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;

    // 从 device 端取 ptr 数组（注意此处假设 ptr 已经在 host 有副本）
    std::vector<int> h_ptr(num_v + 1);
    checkCudaErrors(cudaMemcpy(h_ptr.data(), d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost));

    // 构建 eid -> row 映射
    std::vector<int> h_eid_to_row(num_e);
    for (int row = 0; row < num_v; ++row) {
        for (int i = h_ptr[row]; i < h_ptr[row + 1]; ++i) {
            h_eid_to_row[i] = row;
        }
    }

    // 拷贝到 device
    checkCudaErrors(cudaMalloc((void**)&eid_to_row, sizeof(int) * num_e));
    checkCudaErrors(cudaMemcpy(eid_to_row, h_eid_to_row.data(), sizeof(int) * num_e, cudaMemcpyHostToDevice));
}

void SpMMOpt::run(float* vin, float* vout)
{
    spmm_kernel_atomic<<<grid, block>>>(
        d_idx, d_val,
        vin, vout,
        eid_to_row,
        num_e, feat_in);
}
