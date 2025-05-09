// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

#define B 32

__global__ void step_1(int n, int *graph, int p, int B) {
    extern __shared__ int block[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = p * B + ty;
    int j = p * B + tx;

    // 处理边界块
    block[ty * B + tx] = (i < n && j < n) ? graph[i * n + j] : 100001;

    __syncthreads();

    // Floyd-Warshall
    for (int k = 0; k < B; ++k) {
        int temp = block[ty * B + k] + block[k * B + tx];
        if (temp < block[ty * B + tx]) block[ty * B + tx] = temp;
        // __syncthreads();
    }

    if (i < n && j < n) graph[i * n + j] = block[ty * B + tx];
}


__global__ void step_2(int n, int *graph, int p, int B) {
    extern __shared__ int shared[];

    int* pivot = shared;              // G(p, p)
    int* target = &shared[B * B];     // 待处理的块

   const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int q  = blockIdx.x;             // 目标块序号
    const bool is_row = (blockIdx.y == 0);

    // 跳过对角块
    if (q == p) return;

    int i, j;
    if (is_row) {
        i = p * B + ty;
        j = q * B + tx;
    }
    else {
        i = q * B + ty;
        j = p * B + tx;
    }

    // 读 G(p,p)
    int pi = p * B + ty;
    int pj = p * B + tx;
    pivot[ty * B + tx] = (pi < n && pj < n) ? graph[pi * n + pj] : 100001;

    // 读取 target block
    target[ty * B + tx] = (i < n && j < n) ? graph[i * n + j] : 100001;

    __syncthreads();

    for (int k = 0; k < B; ++k) {
        int temp;
        if (is_row) temp = pivot[ty * B + k] + target[k * B + tx];
        else    temp = target[ty * B + k] + pivot[k * B + tx];

        if (temp < target[ty * B + tx]) target[ty * B + tx] = temp;

        // __syncthreads();
    }

    if (i < n && j < n) graph[i * n + j] = target[ty * B + tx];
}


__global__ void step_3(int n, int* graph, int p, int B) {
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

    for (int k = 0; k < B; ++k) {
        int temp = blockA[ty * B + k] + blockB[k * B + tx];
        if (temp < blockC[ty * B + tx]) {
            blockC[ty * B + tx] = temp;
        }
        // __syncthreads();
    }

    if (row < n && col < n) {
        graph[row * n + col] = blockC[ty * B + tx];
    }
}



void apsp(int n, /* device */ int* graph) {
    const int B = 32;
    const int num_blocks = (n + B - 1) / B;
    
    for (int p = 0; p < num_blocks; ++p) {
        step_1<<<1, dim3(B, B), B * B * sizeof(int)>>>(n, graph, p, B);
        
        step_2<<<dim(num_blocks, 2), dim3(B, B), 2 * B * B * sizeof(int)>>>(n, graph, p, B);

        dim3 grid(num_blocks, num_blocks);
        step_3<<<grid, dim3(B, B), 3 * B * B * sizeof(int)>>>(n, graph, p, B);
    }
}

