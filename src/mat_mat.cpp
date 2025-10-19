#include "TimeLogger.h"
#include "aligned_allocator.h"
#include "common.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void mat_mat_32_64(const u32 *const __restrict A, const u32 *const __restrict B,
                   u64 *__restrict out, const size_t rows, const size_t cols) {
  u64 t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0;
    t1 = 0;
    const size_t offset = i * cols;
#pragma GCC unroll 128
    for (size_t k = 0; k < cols; k++) {
      t0 += (u64)A[offset + k] * B[b_cols * k];
      t1 += (u64)A[offset + k] * B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

void mat_mat_64(const u64 *__restrict A, const u64 *__restrict B,
                u64 *__restrict out, const size_t rows, const size_t cols) {
  u64 t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0;
    t1 = 0;
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

void mat_mat_64_128(const u64 *const __restrict A,
                    const u64 *const __restrict B, u128 *__restrict out,
                    const size_t rows, const size_t cols) {
  u128 t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0;
    t1 = 0;
    const size_t offset = i * cols;
#pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += A[offset + k] * (u128)B[b_cols * k];
      t1 += A[offset + k] * (u128)B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

void mat_mat_vert4_32_64(const u32 *__restrict A, const u32 *__restrict B,
                        u64 *__restrict out, const size_t rows,
                        const size_t cols) {
  // assert(rows % 4 == 0);  // ! let's just assume this for now.
  for (size_t i = 0; i + 4 <= rows; i += 4) {
    const u32 *__restrict a_ptr = A + i * cols;

    // u32 a0, a1, a2, a3;
    // u32 b0, b1;
    u64 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // column 0 accumulators
    u64 c4 = 0, c5 = 0, c6 = 0, c7 = 0; // column 1 accumulators

// #pragma GCC unroll 16
    for (size_t j = 0; j < cols; ++j) {
      // A: 4 vertically packed elements for column j
      const u32 a0 = a_ptr[j * 4 + 0];
      const u32 a1 = a_ptr[j * 4 + 1];
      const u32 a2 = a_ptr[j * 4 + 2];
      const u32 a3 = a_ptr[j * 4 + 3];

      // B: two columns for row j
      const u32 b0 = B[j * 2 + 0];
      const u32 b1 = B[j * 2 + 1];

      c0 += (u64)a0 * b0;
      c1 += (u64)a1 * b0;
      c2 += (u64)a2 * b0;
      c3 += (u64)a3 * b0;

      c4 += (u64)a0 * b1;
      c5 += (u64)a1 * b1;
      c6 += (u64)a2 * b1;
      c7 += (u64)a3 * b1;
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

void mat_mat_vert8_32_64(const uint32_t *__restrict A_pk,
                                        const uint32_t *__restrict B_rm,
                                        uint64_t *__restrict C_rm, size_t rows,
                                        size_t cols) {
  // assert(rows % 8 == 0);
  for (size_t i = 0; i + 8 <= rows; i += 8) {
    const uint32_t *a_ptr = A_pk + i * cols;

    uint64_t c0[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // for B col 0
    uint64_t c1[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // for B col 1

#pragma GCC unroll 8
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

void mat_mat_vert_avx512_32_64(const u32 *__restrict A, const u32 *__restrict B,
                               u64 *__restrict C, size_t rows, size_t cols) {
  for (size_t i = 0; i + 8 <= rows; i += 8) {
    const u32 *a_ptr = A + i * cols;

    __m512i acc0 = _mm512_setzero_si512(); // column 0 accumulators (8x u64)
    __m512i acc1 = _mm512_setzero_si512(); // column 1 accumulators

    for (size_t j = 0; j < cols; ++j) {
      // Load 8 x u32 for column j of this 8-row tile
      const __m256i a32 = _mm256_loadu_si256((const __m256i *)(a_ptr + j * 8));
      // Widen to 8 x u64
      const __m512i a64 = _mm512_cvtepu32_epi64(a32); // TODO: this instruction is weird. See if there is a better way.

      // Broadcast B[j,0] and B[j,1] to u64 vectors
      const __m512i b0v = _mm512_set1_epi64((long long)(u64)B[j * 2 + 0]);
      const __m512i b1v = _mm512_set1_epi64((long long)(u64)B[j * 2 + 1]);

      // Multiply and accumulate: 8 lanes of u64
      const __m512i p0 = _mm512_mullo_epi64(a64, b0v);
      const __m512i p1 = _mm512_mullo_epi64(a64, b1v);
      acc0 = _mm512_add_epi64(acc0, p0);
      acc1 = _mm512_add_epi64(acc1, p1);
    }

    // Store out: we need to interleave into C with stride 2.
    alignas(64) u64 tmp0[8], tmp1[8];
    _mm512_store_si512((__m512i *)tmp0, acc0);
    _mm512_store_si512((__m512i *)tmp1, acc1);

    for (int lane = 0; lane < 8; ++lane) {
      C[(i + lane) * 2 + 0] = tmp0[lane];
      C[(i + lane) * 2 + 1] = tmp1[lane];
    }
  }
}

void mat_mat_32_64_avx2(const u32 *A, const u32 *B, u64 *out, size_t rows,
                        size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    const u32 *arow = A + i * cols;

    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();

    size_t k = 0;
    for (; k + 4 <= cols; k += 4) {
      __m128i a32 = _mm_loadu_si128((const __m128i *)(arow + k));
      __m128i a_lo_dup = _mm_unpacklo_epi32(a32, a32);
      __m128i a_hi_dup = _mm_unpackhi_epi32(a32, a32);
      __m256i a_rep = _mm256_set_m128i(a_hi_dup, a_lo_dup);

      const u32 *bptr = B + (k * b_cols);
      __m256i b_pairs = _mm256_loadu_si256((const __m256i *)bptr);

      acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(a_rep, b_pairs));
      __m256i b_hi = _mm256_srli_epi64(b_pairs, 32);
      acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(a_rep, b_hi));
    }

    alignas(32) u64 tmp0[4];
    alignas(32) u64 tmp1[4];
    _mm256_store_si256((__m256i *)tmp0, acc0);
    _mm256_store_si256((__m256i *)tmp1, acc1);
    u64 t0 = tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3];
    u64 t1 = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];

    for (; k < cols; ++k) {
      u64 a = arow[k];
      t0 += a * (u64)B[k * b_cols + 0];
      t1 += a * (u64)B[k * b_cols + 1];
    }

    out[b_cols * i + 0] = t0;
    out[b_cols * i + 1] = t1;
  }
}

void mat_mat_32_64_avx512(const u32 *A, const u32 *B, u64 *out, size_t rows,
                          size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    const u32 *arow = A + i * cols;
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();

    size_t k = 0;
    const size_t step = 8;
    for (; k + step <= cols; k += step) {
      __m256i a32 = _mm256_loadu_si256((const __m256i *)(arow + k));
      __m512i a64 = _mm512_cvtepu32_epi64(a32);
      const u32 *bptr = B + k * 2;
      __m512i b_pairs = _mm512_loadu_si512((const void *)bptr);
      acc0 = _mm512_add_epi64(acc0, _mm512_mul_epu32(a64, b_pairs));
      __m512i b1 = _mm512_srli_epi64(b_pairs, 32);
      acc1 = _mm512_add_epi64(acc1, _mm512_mul_epu32(a64, b1));
    }

    alignas(64) u64 tmp0[8];
    alignas(64) u64 tmp1[8];
    _mm512_store_si512((__m512i *)tmp0, acc0);
    _mm512_store_si512((__m512i *)tmp1, acc1);
    u64 t0 = 0, t1 = 0;
    for (int x = 0; x < 8; ++x) {
      t0 += tmp0[x];
      t1 += tmp1[x];
    }

    for (; k < cols; ++k) {
      u64 a = arow[k];
      t0 += a * (u64)B[k * b_cols + 0];
      t1 += a * (u64)B[k * b_cols + 1];
    }

    out[b_cols * i + 0] = t0;
    out[b_cols * i + 1] = t1;
  }
}

int main() {

  std::vector<u64, AlignedAllocator<u64, 64>> A_64(rows_64 * cols);
  std::vector<u64, AlignedAllocator<u64, 64>> B_64(cols * b_cols);
  std::vector<u64, AlignedAllocator<u64, 64>> out_64(rows_32 * b_cols);

  std::vector<u32, AlignedAllocator<u32, 64>> A_32(rows_32 * cols);
  std::vector<u32, AlignedAllocator<u32, 64>> B_32(cols * b_cols);
  std::vector<u32, AlignedAllocator<u32, 64>> out_32(rows_32 * b_cols);
  std::vector<u128, AlignedAllocator<u128, 64>> out_128(rows_64 * b_cols);

  std::generate(A_32.begin(), A_32.end(), rand);
  std::generate(B_32.begin(), B_32.end(), rand);
  std::generate(A_64.begin(), A_64.end(), rand);
  std::generate(B_64.begin(), B_64.end(), rand);

  const double mat_32_MB = experiments * (A_32.size() * sizeof(u32)) / (1024.0 * 1024.0);
  const double mat_64_MB = experiments * (A_64.size() * sizeof(u64)) / (1024.0 * 1024.0);

  size_t tot_sum = 0;

  TIME_START("mat_mat_64");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_64(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  tot_sum += out_64[rand() % rows_64];
  TIME_END("mat_mat_64");
  PRINT_THROUGHPUT("mat_mat_64", mat_64_MB);


  TIME_START("mat_mat_64_128");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  tot_sum += out_128[rand() % out_128.size()];
  TIME_END("mat_mat_64_128");
  PRINT_THROUGHPUT("mat_mat_64_128", mat_64_MB);


  TIME_START("mat_mat_32_64");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64");
  PRINT_THROUGHPUT("mat_mat_32_64", mat_32_MB);

  TIME_START("mat_mat_vert4_32_64");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_vert4_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_vert4_32_64");
  PRINT_THROUGHPUT("mat_mat_vert4_32_64", mat_32_MB);

  TIME_START("mat_mat_vert8_32_64");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_vert8_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_vert8_32_64");
  PRINT_THROUGHPUT("mat_mat_vert8_32_64", mat_32_MB);

  TIME_START("mat_mat_vert_avx512_32_64");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_vert_avx512_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32,
                         cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_vert_avx512_32_64");
  PRINT_THROUGHPUT("mat_mat_vert_avx512_32_64", mat_32_MB);

  TIME_START("mat_mat_32_64_avx2");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_32_64_avx2(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64_avx2");
  PRINT_THROUGHPUT("mat_mat_32_64_avx2", mat_32_MB);

  TIME_START("mat_mat_32_64_avx512");
  for (size_t i = 0; i < experiments; i++)
    mat_mat_32_64_avx512(A_32.data(), B_32.data(), out_64.data(), rows_32,
                         cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64_avx512");
  PRINT_THROUGHPUT("mat_mat_32_64_avx512", mat_32_MB);

  std::cout << "tot_sum: " << tot_sum << std::endl;
  return 0;
}
