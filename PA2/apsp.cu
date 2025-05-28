// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

#define B 32

__global__ void step_1(int n, int *graph, int p) {
    extern __shared__ int block[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = p * B + ty;
    int j = p * B + tx;

    int idx = ty * B + tx;
    block[idx] = (i < n && j < n) ? graph[i * n + j] : 100001;
    __syncthreads();

    int reg_row[B];  // 当前线程处理的 row（ty）
    int reg_col[B];  // 当前线程处理的 col（tx）

#pragma unroll
    for (int k = 0; k < B; ++k) {
        reg_row[k] = block[ty * B + k];  // 第 ty 行
        reg_col[k] = block[k * B + tx];  // 第 tx 列
    }

    int self = block[idx];

#pragma unroll
    for (int k = 0; k < B; ++k) {
        int temp = reg_row[k] + reg_col[k];
        if (temp < self) self = temp;
    }

    if (i < n && j < n)
        graph[i * n + j] = self;
}



__global__ void step_2(int n, int *graph, int p) {
    extern __shared__ int shared[];

    int* pivot = shared;              // G(p, p)
    int* target = &shared[B * B];     // G(p, q) 或 G(q, p)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int q = blockIdx.x;
    const bool is_row = (blockIdx.y == 0);

    if (q == p) return;

    int i, j;
    if (is_row) {
        i = p * B + ty;
        j = q * B + tx;
    } else {
        i = q * B + ty;
        j = p * B + tx;
    }

    int pi = p * B + ty;
    int pj = p * B + tx;
    pivot[ty * B + tx] = (pi < n && pj < n) ? graph[pi * n + pj] : 100001;
    target[ty * B + tx] = (i < n && j < n) ? graph[i * n + j] : 100001;
    __syncthreads();

    int result = target[ty * B + tx];

    if (is_row) {
        int row_k[B];
        int pivot_row[B];
#pragma unroll
        for (int k = 0; k < B; ++k) {
            pivot_row[k] = pivot[ty * B + k];    // 第 ty 行
            row_k[k] = target[k * B + tx];       // 同列元素
        }

#pragma unroll
        for (int k = 0; k < B; ++k) {
            int temp = pivot_row[k] + row_k[k];
            if (temp < result) result = temp;
        }
    } else {
        int col_k[B];
        int pivot_col[B];
#pragma unroll
        for (int k = 0; k < B; ++k) {
            pivot_col[k] = pivot[k * B + tx];    // 第 tx 列
            col_k[k] = target[ty * B + k];       // 同行元素
        }

#pragma unroll
        for (int k = 0; k < B; ++k) {
            int temp = col_k[k] + pivot_col[k];
            if (temp < result) result = temp;
        }
    }

    if (i < n && j < n)
        graph[i * n + j] = result;
}



__global__ void step_3(int n, int* graph, int p) {
    extern __shared__ int shared[];

    int* blockA = shared;                // G(i, p)
    int* blockB = &shared[B * B];        // G(p, j)
    int* blockC = &shared[2 * B * B];    // G(i, j)

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.y;
    int j = blockIdx.x;

    if (i == p || j == p) return;

    int row = i * B + ty;
    int col = j * B + tx;

    int idxA = i * B + ty;
    int idxP = p * B + tx;
    blockA[ty * B + tx] = (idxA < n && idxP < n) ? graph[idxA * n + idxP] : 100001;

    int idxP2 = p * B + ty;
    int idxB = j * B + tx;
    blockB[ty * B + tx] = (idxP2 < n && idxB < n) ? graph[idxP2 * n + idxB] : 100001;

    blockC[ty * B + tx] = (row < n && col < n) ? graph[row * n + col] : 100001;

    __syncthreads();

    int result = blockC[ty * B + tx];

    int regA[B];
    int regB[B];

#pragma unroll
    for (int k = 0; k < B; ++k) {
        regA[k] = blockA[ty * B + k];  // 同一行
        regB[k] = blockB[k * B + tx];  // 同一列
    }

#pragma unroll
    for (int k = 0; k < B; ++k) {
        int temp = regA[k] + regB[k];
        if (temp < result) result = temp;
    }

    if (row < n && col < n) {
        graph[row * n + col] = result;
    }
}


void apsp(int n, /* device */ int* graph) {
    // const int B = 32;
    const int num_blocks = (n + B - 1) / B;
    
    for (int p = 0; p < num_blocks; ++p) {
        step_1<<<1, dim3(B, B), B * B * sizeof(int)>>>(n, graph, p);
        
        step_2<<<dim3(num_blocks, 2), dim3(B, B), 2 * B * B * sizeof(int)>>>(n, graph, p);

        dim3 grid(num_blocks, num_blocks);
        step_3<<<grid, dim3(B, B), 3 * B * B * sizeof(int)>>>(n, graph, p);
    }
}
