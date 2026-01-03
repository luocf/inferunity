// SIMD工具函数实现
// 参考NCNN的SIMD实现

#include "simd_utils.h"
#include <algorithm>
#include <cmath>

namespace inferunity {
namespace simd {

void AddSIMD(const float* a, const float* b, float* c, size_t count) {
    size_t i = 0;
    
#ifdef __AVX__
    if (HasAVX() && count >= 8) {
        size_t simd_count = count & ~7;  // 8的倍数
        for (; i < simd_count; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
    }
#elif defined(__SSE__)
    if (HasSSE() && count >= 4) {
        size_t simd_count = count & ~3;  // 4的倍数
        for (; i < simd_count; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
    }
#elif defined(__ARM_NEON)
    if (HasNEON() && count >= 4) {
        size_t simd_count = count & ~3;  // 4的倍数
        for (; i < simd_count; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(c + i, vc);
        }
    }
#endif
    
    // 处理剩余元素
    for (; i < count; ++i) {
        c[i] = a[i] + b[i];
    }
}

void MulSIMD(const float* a, const float* b, float* c, size_t count) {
    size_t i = 0;
    
#ifdef __AVX__
    if (HasAVX() && count >= 8) {
        size_t simd_count = count & ~7;
        for (; i < simd_count; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
    }
#elif defined(__SSE__)
    if (HasSSE() && count >= 4) {
        size_t simd_count = count & ~3;
        for (; i < simd_count; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_mul_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
    }
#elif defined(__ARM_NEON)
    if (HasNEON() && count >= 4) {
        size_t simd_count = count & ~3;
        for (; i < simd_count; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vmulq_f32(va, vb);
            vst1q_f32(c + i, vc);
        }
    }
#endif
    
    // 处理剩余元素
    for (; i < count; ++i) {
        c[i] = a[i] * b[i];
    }
}

void ReluSIMD(const float* input, float* output, size_t count) {
    size_t i = 0;
    
#ifdef __AVX__
    if (HasAVX() && count >= 8) {
        __m256 zero = _mm256_setzero_ps();
        size_t simd_count = count & ~7;
        for (; i < simd_count; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            __m256 result = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(output + i, result);
        }
    }
#elif defined(__SSE__)
    if (HasSSE() && count >= 4) {
        __m128 zero = _mm_setzero_ps();
        size_t simd_count = count & ~3;
        for (; i < simd_count; i += 4) {
            __m128 v = _mm_loadu_ps(input + i);
            __m128 result = _mm_max_ps(v, zero);
            _mm_storeu_ps(output + i, result);
        }
    }
#elif defined(__ARM_NEON)
    if (HasNEON() && count >= 4) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        size_t simd_count = count & ~3;
        for (; i < simd_count; i += 4) {
            float32x4_t v = vld1q_f32(input + i);
            float32x4_t result = vmaxq_f32(v, zero);
            vst1q_f32(output + i, result);
        }
    }
#endif
    
    // 处理剩余元素
    for (; i < count; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void MatMulSIMD(const float* A, const float* B, float* C, 
                int64_t M, int64_t K, int64_t N) {
    // 简化的SIMD矩阵乘法
    // 实际应该使用更优化的实现（如分块、转置等）
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            
            // 使用SIMD加速内积计算
            int64_t k = 0;
#ifdef __AVX__
            if (HasAVX() && K >= 8) {
                __m256 sum_vec = _mm256_setzero_ps();
                int64_t simd_count = K & ~7;
                for (; k < simd_count; k += 8) {
                    __m256 va = _mm256_loadu_ps(A + i * K + k);
                    __m256 vb = _mm256_loadu_ps(B + k * N + j);
                    // 注意：这里需要转置B或使用gather指令
                    // 简化实现：逐元素相乘后求和
                    for (int kk = 0; kk < 8; ++kk) {
                        sum += A[i * K + k + kk] * B[(k + kk) * N + j];
                    }
                }
            }
#elif defined(__SSE__)
            if (HasSSE() && K >= 4) {
                int64_t simd_count = K & ~3;
                for (; k < simd_count; k += 4) {
                    for (int kk = 0; kk < 4; ++kk) {
                        sum += A[i * K + k + kk] * B[(k + kk) * N + j];
                    }
                }
            }
#endif
            
            // 处理剩余元素
            for (; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            
            C[i * N + j] = sum;
        }
    }
}

} // namespace simd
} // namespace inferunity

