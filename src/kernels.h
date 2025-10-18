#ifndef KERNELS_H
#define KERNELS_H

#include "aligned_allocator.h"
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <immintrin.h>
#include <Eigen/Dense>

// Define uint128_t for GCC.
using uint128_t = unsigned __int128;
constexpr size_t b_cols = 2;

// ========================== simple read ==========================
static inline uint32_t simple_read_32(const uint32_t *const __restrict a, const size_t size) {
  uint32_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}

static inline uint64_t simple_read_64(const uint64_t *const __restrict a, const size_t size) {
  uint64_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}

// ========================== vector dot product ==========================
static inline uint32_t vec_dot_32(const uint32_t *const __restrict a, const uint32_t *const __restrict b,
                   const size_t size) {
  uint32_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static inline uint64_t vec_dot_32_64(const uint32_t *const __restrict a, const uint32_t *const __restrict b,
                   const size_t size) {
  uint64_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += (uint64_t)a[i] *(uint64_t) b[i];
  }
  return sum;
}

static inline uint64_t vec_dot_64(const uint64_t *const __restrict a, const uint64_t *const __restrict b,
                   const size_t size){
  uint64_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static inline uint128_t vec_dot_64_128(const uint64_t *const __restrict a, const uint64_t *const __restrict b,
                   const size_t size) {
  uint128_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += (uint128_t)a[i] * b[i];
  }
  return sum;
}

// AVX2 kernel: dot product of two u32 vectors, accumulate into uint64_t
static inline uint64_t dot_u32_u64_avx2(const uint32_t* __restrict a,
  const uint32_t* __restrict b,
  size_t n)
{
#if defined(__AVX2__)
  __m256i acc_even = _mm256_setzero_si256();
  __m256i acc_odd  = _mm256_setzero_si256();
  size_t i = 0;

  for (; i + 8 <= n; i += 8) {
    __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

    __m256i prod_even = _mm256_mul_epu32(va, vb);

    __m256i va_hi = _mm256_srli_epi64(va, 32);
    __m256i vb_hi = _mm256_srli_epi64(vb, 32);
    __m256i prod_odd = _mm256_mul_epu32(va_hi, vb_hi);

    acc_even = _mm256_add_epi64(acc_even, prod_even);
    acc_odd  = _mm256_add_epi64(acc_odd,  prod_odd);
  }

  alignas(32) uint64_t buf[4];
  uint64_t sum = 0;
  _mm256_store_si256((__m256i*)buf, acc_even);
  sum += buf[0] + buf[1] + buf[2] + buf[3];
  _mm256_store_si256((__m256i*)buf, acc_odd);
  sum += buf[0] + buf[1] + buf[2] + buf[3];

  for (; i < n; ++i) sum += (uint64_t)a[i] * (uint64_t)b[i];
  return sum;
#else
  uint64_t acc0=0, acc1=0, acc2=0, acc3=0;
  size_t i=0;
  for (; i+4<=n; i+=4) {
    acc0 += (uint64_t)a[i+0]*b[i+0];
    acc1 += (uint64_t)a[i+1]*b[i+1];
    acc2 += (uint64_t)a[i+2]*b[i+2];
    acc3 += (uint64_t)a[i+3]*b[i+3];
  }
  for (; i<n; ++i) acc0 += (uint64_t)a[i]*b[i];
  return (acc0+acc1)+(acc2+acc3);
#endif
}

// ========================== matrix vector multiplication ==========================
static inline void mat_vec_32(const uint32_t *const __restrict A,
                const uint32_t *const __restrict b, uint32_t *__restrict out,
                const size_t rows, const size_t cols) {
  uint32_t tmp = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
    #pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}

static inline void mat_vec_32_64(const uint32_t *const __restrict A,
                   const uint32_t *const __restrict b, uint64_t *__restrict out,
                   const size_t rows, const size_t cols) {
  uint64_t tmp = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
    #pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * (uint64_t)b[k];
    }
    out[i] = tmp;
  }
}


/**

Divide matrix A into vertical blocks of 4 rows each.
a0 | a0 | a0 | a0
a1 | a1 | a1 | a1
a2 | a2 | a2 | a2
a3 | a3 | a3 | a3

a0    *     b0 | b1   -->     c0 | c4
a1    *     b0 | b1   -->     c1 | c5
a2    *     b0 | b1   -->     c2 | c6
a3    *     b0 | b1   -->     c3 | c7

*/
// -----------------------------
// Kernel: corrected version
// A_pack: (rows x cols) but packed in vertical tiles of 4 rows: for a 4xK tile,
//          the memory order is [a(i+0,k), a(i+1,k), a(i+2,k), a(i+3,k)] for k=0..K-1
// B_row2: (cols x 2) laid row-major with 2 columns (flat: [b(k,0), b(k,1)] per k)
// out    : (rows x 2) of uint64_t, row-major with stride 2
// rows must be a multiple of 4.
static inline void mat_mat_vert_32_64(const uint32_t *__restrict A_pack,
                                      const uint32_t *__restrict B_row2,
                                      uint64_t *__restrict out,
                                      const size_t rows, const size_t cols) {
  assert(rows % 4 == 0);
  for (size_t i = 0; i + 4 <= rows; i += 4) {
    const uint32_t *__restrict a_ptr = A_pack + i * cols;

    uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0; // column 0 accumulators
    uint64_t c4 = 0, c5 = 0, c6 = 0, c7 = 0; // column 1 accumulators

    for (size_t j = 0; j < cols; ++j) {
      // A: 4 vertically packed elements for column j
      const uint32_t a0 = a_ptr[j * 4 + 0];
      const uint32_t a1 = a_ptr[j * 4 + 1];
      const uint32_t a2 = a_ptr[j * 4 + 2];
      const uint32_t a3 = a_ptr[j * 4 + 3];

      // B: two columns for row j
      const uint32_t b0 = B_row2[j * 2 + 0];
      const uint32_t b1 = B_row2[j * 2 + 1];

      c0 += (uint64_t)a0 * b0;
      c1 += (uint64_t)a1 * b0;
      c2 += (uint64_t)a2 * b0;
      c3 += (uint64_t)a3 * b0;

      c4 += (uint64_t)a0 * b1;
      c5 += (uint64_t)a1 * b1;
      c6 += (uint64_t)a2 * b1;
      c7 += (uint64_t)a3 * b1;
    }

    // Store to (rows x 2) row-major
    out[(i + 0) * 2 + 0] = c0;
    out[(i + 1) * 2 + 0] = c1;
    out[(i + 2) * 2 + 0] = c2;
    out[(i + 3) * 2 + 0] = c3;

    out[(i + 0) * 2 + 1] = c4;
    out[(i + 1) * 2 + 1] = c5;
    out[(i + 2) * 2 + 1] = c6;
    out[(i + 3) * 2 + 1] = c7;
  }
}

static inline void mat_mat_vert8_scalar(const uint32_t *__restrict A_pk,
                                        const uint32_t *__restrict B_rm,
                                        uint64_t *__restrict C_rm, size_t rows,
                                        size_t cols) {
  assert(rows % 8 == 0);
  for (size_t i = 0; i + 8 <= rows; i += 8) {
    const uint32_t *a_ptr = A_pk + i * cols;

    uint64_t c0[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // for B col 0
    uint64_t c1[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // for B col 1

    for (size_t j = 0; j < cols; ++j) {
      const uint32_t a0 = a_ptr[j * 8 + 0];
      const uint32_t a1 = a_ptr[j * 8 + 1];
      const uint32_t a2 = a_ptr[j * 8 + 2];
      const uint32_t a3 = a_ptr[j * 8 + 3];
      const uint32_t a4 = a_ptr[j * 8 + 4];
      const uint32_t a5 = a_ptr[j * 8 + 5];
      const uint32_t a6 = a_ptr[j * 8 + 6];
      const uint32_t a7 = a_ptr[j * 8 + 7];

      const uint64_t b0 = B_rm[j * 2 + 0];
      const uint64_t b1 = B_rm[j * 2 + 1];

      c0[0] += (uint64_t)a0 * b0;
      c1[0] += (uint64_t)a0 * b1;
      c0[1] += (uint64_t)a1 * b0;
      c1[1] += (uint64_t)a1 * b1;
      c0[2] += (uint64_t)a2 * b0;
      c1[2] += (uint64_t)a2 * b1;
      c0[3] += (uint64_t)a3 * b0;
      c1[3] += (uint64_t)a3 * b1;
      c0[4] += (uint64_t)a4 * b0;
      c1[4] += (uint64_t)a4 * b1;
      c0[5] += (uint64_t)a5 * b0;
      c1[5] += (uint64_t)a5 * b1;
      c0[6] += (uint64_t)a6 * b0;
      c1[6] += (uint64_t)a6 * b1;
      c0[7] += (uint64_t)a7 * b0;
      c1[7] += (uint64_t)a7 * b1;
    }

    for (int lane = 0; lane < 8; ++lane) {
      C_rm[(i + lane) * 2 + 0] = c0[lane];
      C_rm[(i + lane) * 2 + 1] = c1[lane];
    }
  }
}

static inline void mat_mat_vert8_avx512(const uint32_t *__restrict A_pk,
                                        const uint32_t *__restrict B_rm,
                                        uint64_t *__restrict C_rm, size_t rows,
                                        size_t cols) {
  assert(rows % 8 == 0);

  for (size_t i = 0; i + 8 <= rows; i += 8) {
    const uint32_t *a_ptr = A_pk + i * cols;

    __m512i acc0 = _mm512_setzero_si512(); // column 0 accumulators (8x u64)
    __m512i acc1 = _mm512_setzero_si512(); // column 1 accumulators

    for (size_t j = 0; j < cols; ++j) {
      // Load 8 x u32 for column j of this 8-row tile
      const __m256i a32 = _mm256_loadu_si256((const __m256i *)(a_ptr + j * 8));
      // Widen to 8 x u64
      const __m512i a64 = _mm512_cvtepu32_epi64(a32);

      // Broadcast B[j,0] and B[j,1] to u64 vectors
      const __m512i b0v =
          _mm512_set1_epi64((long long)(uint64_t)B_rm[j * 2 + 0]);
      const __m512i b1v =
          _mm512_set1_epi64((long long)(uint64_t)B_rm[j * 2 + 1]);

      // Multiply and accumulate: 8 lanes of u64
      const __m512i p0 = _mm512_mullo_epi64(a64, b0v);
      const __m512i p1 = _mm512_mullo_epi64(a64, b1v);
      acc0 = _mm512_add_epi64(acc0, p0);
      acc1 = _mm512_add_epi64(acc1, p1);
    }

    // Store out: we need to interleave into C_rm with stride 2.
    alignas(64) uint64_t tmp0[8], tmp1[8];
    _mm512_store_si512((__m512i *)tmp0, acc0);
    _mm512_store_si512((__m512i *)tmp1, acc1);

    for (int lane = 0; lane < 8; ++lane) {
      C_rm[(i + lane) * 2 + 0] = tmp0[lane];
      C_rm[(i + lane) * 2 + 1] = tmp1[lane];
    }
  }
}

static inline void mat_vec_32_64_Eigen(const uint32_t *A, const uint32_t *b, uint64_t *out, size_t rows, size_t cols) {
  Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
  Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>> matB(b, cols);
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>> matOut(out, rows);
  matOut.noalias() = matA.cast<uint64_t>() * matB.cast<uint64_t>();
}

static inline void mat_vec_32_64_avx2(const uint32_t* __restrict A,
  const uint32_t* __restrict b,
  uint64_t* __restrict out,
  size_t rows, size_t cols) {
  #pragma GCC ivdep
  for (size_t i = 0; i < rows; ++i) {
    const uint32_t* __restrict arow = A + i * cols;
    out[i] = dot_u32_u64_avx2(arow, b, cols);
  }
}

static inline void mat_vec_64(const uint64_t *const __restrict A,
                const uint64_t *const __restrict b, uint64_t *__restrict out,
                const size_t rows, const size_t cols) {
  uint64_t tmp = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
    #pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}

static inline void mat_vec_64_128(const uint64_t *const __restrict A,
                 const uint64_t *const __restrict b, uint128_t *__restrict out,
                 const size_t rows, const size_t cols) {
  uint128_t tmp = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      tmp += (uint128_t)A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}

// ========================== matrix matrix (2-col) ==========================
static inline void mat_mat_64(const uint64_t *__restrict A, const uint64_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += A[offset + k] * B[b_cols * k];
      t1 += A[offset + k] * B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

static inline void mat_mat_64_128(const uint64_t *__restrict A, const uint64_t *__restrict B,
                 uint128_t *__restrict out, const size_t rows,
                 const size_t cols) {
  uint128_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += A[offset + k] * (uint128_t)B[b_cols * k];
      t1 += A[offset + k] * (uint128_t)B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

static inline void mat_mat_32_64(const uint32_t *__restrict A, const uint32_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
    #pragma GCC unroll 128
    for (size_t k = 0; k < cols; k++) {
      t0 += (uint64_t)A[offset + k] * B[b_cols * k];
      t1 += (uint64_t)A[offset + k] * B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

static inline void mat_mat_32_64_Eigen(const uint32_t *__restrict A, const uint32_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
  Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, cols, b_cols);
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matOut(out, rows, b_cols);
  matOut.noalias() = matA.cast<uint64_t>() * matB.cast<uint64_t>();
}

static inline void mat_mat_32_64_avx2(const uint32_t *A, const uint32_t *B, uint64_t *out,
                         size_t rows, size_t cols) {
#if defined(__AVX2__)
  for (size_t i = 0; i < rows; ++i) {
    const uint32_t *arow = A + i * cols;

    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();

    size_t k = 0;
    for (; k + 4 <= cols; k += 4) {
      __m128i a32 = _mm_loadu_si128((const __m128i *)(arow + k));
      __m128i a_lo_dup = _mm_unpacklo_epi32(a32, a32);
      __m128i a_hi_dup = _mm_unpackhi_epi32(a32, a32);
      __m256i a_rep = _mm256_set_m128i(a_hi_dup, a_lo_dup);

      const uint32_t *bptr = B + (k * b_cols);
      __m256i b_pairs = _mm256_loadu_si256((const __m256i *)bptr);

      acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(a_rep, b_pairs));
      __m256i b_hi = _mm256_srli_epi64(b_pairs, 32);
      acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(a_rep, b_hi));
    }

    alignas(32) uint64_t tmp0[4];
    alignas(32) uint64_t tmp1[4];
    _mm256_store_si256((__m256i *)tmp0, acc0);
    _mm256_store_si256((__m256i *)tmp1, acc1);
    uint64_t t0 = tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3];
    uint64_t t1 = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];

    for (; k < cols; ++k) {
      uint64_t a = arow[k];
      t0 += a * (uint64_t)B[k * b_cols + 0];
      t1 += a * (uint64_t)B[k * b_cols + 1];
    }

    out[b_cols * i + 0] = t0;
    out[b_cols * i + 1] = t1;
  }
#else
  mat_mat_32_64(A, B, out, rows, cols);
#endif
}

static inline void mat_mat_32_64_avx512(const uint32_t *A, const uint32_t *B, uint64_t *out,
                          size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    const uint32_t *arow = A + i * cols;
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();

    size_t k = 0;
    const size_t step = 8;
    for (; k + step <= cols; k += step) {
      __m256i a32 = _mm256_loadu_si256((const __m256i*)(arow + k));
      __m512i a64 = _mm512_cvtepu32_epi64(a32);
      const uint32_t* bptr = B + k*2;
      __m512i b_pairs = _mm512_loadu_si512((const void*)bptr);
      acc0 = _mm512_add_epi64(acc0, _mm512_mul_epu32(a64, b_pairs));
      __m512i b1 = _mm512_srli_epi64(b_pairs, 32);
      acc1 = _mm512_add_epi64(acc1, _mm512_mul_epu32(a64, b1));
    }

    alignas(64) uint64_t tmp0[8];
    alignas(64) uint64_t tmp1[8];
    _mm512_store_si512((__m512i *)tmp0, acc0);
    _mm512_store_si512((__m512i *)tmp1, acc1);
    uint64_t t0 = 0, t1 = 0;
    for (int x = 0; x < 8; ++x) {
      t0 += tmp0[x];
      t1 += tmp1[x];
    }

    for (; k < cols; ++k) {
      uint64_t a = arow[k];
      t0 += a * (uint64_t)B[k * b_cols + 0];
      t1 += a * (uint64_t)B[k * b_cols + 1];
    }

    out[b_cols * i + 0] = t0;
    out[b_cols * i + 1] = t1;
  }
}

#endif // KERNELS_H


