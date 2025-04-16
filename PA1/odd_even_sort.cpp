#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"
using namespace std;

bool edge(int rank, bool odd_even, bool last_rank) {
  if (odd_even) { //奇数阶段
    return rank == 0 || (last_rank && (rank & 1)); //0号进程或最后一个进程且进程号为奇数
  }
  else {  //偶数阶段
    return last_rank && !(rank & 1);  //最后一个进程且进程号为偶数
  }
}

void merge_left(float *A, int m, float *B, int n, float *C) {
    int i = 0, j = 0, k = 0;

    while (k < m && i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else C[k++] = B[j++];
    }

    // 如果 A 还有剩余元素且 C 未填满 m 个元素，则继续从 A 中填充
    while (k < m && i < m) C[k++] = A[i++];

    // 如果 B 还有剩余元素且 C 未填满 m 个元素，则继续从 B 中填充
    while (k < m && j < n) C[k++] = B[j++];
}

void merge_right(float *A, int m, float *B, int n, float *C) {
    int i = m - 1;
    int j = n - 1;
    int k = n - 1;

    while (k >= 0 && i >= 0 && j >= 0) {
        if (A[i] >= B[j]) C[k--] = A[i--];
        else C[k--] = B[j--];
    }

    // 如果 A 还有剩余较大的元素，且 C 未填满 m 个，就继续从 A 中取
    while (k >= 0 && i >= 0) C[k--] = A[i--];

    // 如果 B 还有剩余较大的元素，且 C 未填满 m 个，就继续从 B 中取
    while (k >= 0 && j >= 0) C[k--] = B[j--];

}

void Worker::sort() {
    if (out_of_range) return ;
    std::sort(data, data + block_len);  //先对本地数据排序
    if (nprocs == 1) return ;

    bool odd_even = 0;  //奇偶阶段标志符, 0: even, 1: odd
    float* result_buffer = new float [block_len];
    float* recv_buffer = new float [ceiling(n, nprocs)];

    int maxRounds = nprocs; //可以证明至多需要迭代 nproc 轮
    while(maxRounds--){
        if(edge(rank, odd_even, last_rank)) {
          //边界情况不处理
        }
        else if ((rank & 1) == odd_even) {  //一组进程之中的左侧进程
          //向右侧进程发送最后一个数据并接受右侧进程的第一个数据
          MPI_Sendrecv(data + block_len - 1, 1, MPI_FLOAT, rank + 1, 0,
                      recv_buffer, 1, MPI_FLOAT, rank + 1, 1,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          if (recv_buffer[0] < data[block_len - 1]) {
            //计算右侧进程数据长度
            int rblock_len = min(block_len, n - (rank + 1) * block_len);
            //交换数据时可以少接收和少发送一个数据
            MPI_Sendrecv(data, block_len - 1, MPI_FLOAT, rank + 1, 0, 
                        recv_buffer + 1, rblock_len - 1, MPI_FLOAT, rank + 1, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            
            merge_left(data, block_len, recv_buffer, rblock_len, recv_buffer);
            memcpy(data, recv_buffer, sizeof(float) * block_len);
          }
        }
        else {  //右侧进程
          //左侧长度
          int lblock_len = ceiling(n, nprocs);
          //发送第一个数据并接收左侧进程的最后一个数据
          MPI_Sendrecv(data, 1, MPI_FLOAT, rank - 1, 1,
                      recv_buffer + lblock_len - 1, 1, MPI_FLOAT, rank - 1, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          if (data[0] < recv_buffer[lblock_len - 1]) {
            //交换数据时可以少接收和少发送一个数据
            MPI_Sendrecv(data + 1, block_len - 1, MPI_FLOAT, rank - 1, 1, 
                        recv_buffer, lblock_len - 1, MPI_FLOAT, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
               
            merge_right(recv_buffer, lblock_len, data, block_len, result_buffer);
            memcpy(data, result_buffer, sizeof(float) * block_len);
          }
        }
        odd_even ^= 1;
    }
    delete[] result_buffer;
    delete[] recv_buffer;
}
