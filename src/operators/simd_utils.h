// SIMD工具函数
// 参考NCNN的SIMD实现

#pragma once

#include <cstdint>
#include <cstring>

#ifdef __SSE__
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace inferunity {
namespace simd {

// SIMD特性检测
inline bool HasSSE() {
#ifdef __SSE__
    return true;
#else
    return false;
#endif
}

inline bool HasAVX() {
#ifdef __AVX__
    return true;
#else
    return false;
#endif
}

inline bool HasNEON() {
#ifdef __ARM_NEON
    return true;
#else
    return false;
#endif
}

// SIMD向量化加法（参考NCNN实现）
void AddSIMD(const float* a, const float* b, float* c, size_t count);

// SIMD向量化乘法
void MulSIMD(const float* a, const float* b, float* c, size_t count);

// SIMD向量化ReLU
void ReluSIMD(const float* input, float* output, size_t count);

// SIMD矩阵乘法（简化版）
void MatMulSIMD(const float* A, const float* B, float* C, 
                int64_t M, int64_t K, int64_t N);

} // namespace simd
} // namespace inferunity

