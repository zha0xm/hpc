#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    // Your code here
    for (int i = 0; i < n; i += 8) {
        // 从内存加载 8 个 float 到 AVX 寄存器
        __m256 vec_a = _mm256_load_ps(a + i);
        __m256 vec_b = _mm256_load_ps(b + i);
        
        // 执行 SIMD 加法
        __m256 vec_c = _mm256_add_ps(vec_a, vec_b);
        
        // 将结果存回内存
        _mm256_store_ps(c + i, vec_c);
    }
}