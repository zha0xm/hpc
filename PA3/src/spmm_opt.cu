/******************************************************************************
 *  可调常量 —— 如需改动直接改 #define 即可                               *
 ******************************************************************************/
#define WARP_SIZE   32          // 固定一个 warp
#define TILE_NNZ    256         // 每次搬进 shared memory 的非零数

/******************************************************************************
 *  每个 block / warp 对应 1 行；把 (idx,val) 分批拷到 shared memory          *
 ******************************************************************************/
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

    const int row  = blockIdx.x;      // 一行一个 block
    if (row >= num_rows) return;

    const int lane = threadIdx.x;     // 0–31  (一个 warp)

    const int row_start = ptr[row];
    const int row_end   = ptr[row + 1];
    const int nnz       = row_end - row_start;

    /* ---------- 初始化本行输出 ---------- */
    for (int f = lane; f < in_feat; f += WARP_SIZE)
        vout[row * in_feat + f] = 0.0f;

    /* ---------- 分批把 (idx,val) 搬进 shared memory ---------- */
    for (int tile_base = 0; tile_base < nnz; tile_base += TILE_NNZ)
    {
        int local_e = tile_base + lane;          // warp 中每线程拷 1 edge
        if (local_e < nnz)
        {
            int g_e = row_start + local_e;
            s_idx[local_e - tile_base] = idx[g_e];
            s_val[local_e - tile_base] = val[g_e];
        }
        __syncthreads();                         // 共享内存就绪

        int tile_e_end = min(TILE_NNZ, nnz - tile_base);

        /* ---------- 计算：重用已缓存的 edges ---------- */
#pragma unroll                                // 展开 feature 循环步长 4
        for (int f = lane; f < in_feat; f += WARP_SIZE)
        {
            float acc = vout[row * in_feat + f]; // 取寄存器中间结果

#pragma unroll                              // 展开 tile 内 edge 循环
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

void SpMMOpt::preprocess(float *vin, float *vout)
{
    /* 一个 block = 一个 warp = 32 线程 */
    block = dim3(WARP_SIZE, 1, 1);

    /* 一行一个 block；如行数 > 65535，改成 2D grid */
    grid  = dim3(num_v, 1, 1);
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_shared_tile_kernel<<<grid, block>>>(
        d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
