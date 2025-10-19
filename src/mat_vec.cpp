#include "TimeLogger.h"
#include "aligned_allocator.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <algorithm>

void mat_vec_32(const u32 *const __restrict A, const u32 *const __restrict b,
                u32 *__restrict out, const size_t rows, const size_t cols) {
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

void mat_vec_32_64(const u32 *const __restrict A, const u32 *const __restrict b,
                   u64 *__restrict out, const size_t rows, const size_t cols) {
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

void mat_vec_64(const u64 *const __restrict A, const u64 *const __restrict b,
                u64 *__restrict out, const size_t rows, const size_t cols) {
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

void mat_vec_32_64_avx2(const u32* __restrict A,
  const u32* __restrict b,
  u64* __restrict out,
  size_t rows, size_t cols) {}

void mat_vec_64_128(const u64 *const __restrict A,
                 const u64 *const __restrict b, u128 *__restrict out,
                 const size_t rows, const size_t cols) {
  u128 tmp = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      tmp += (u128)A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}


int main() {
  std::vector<u32, AlignedAllocator<u32, 64>> A_32(rows_32 * cols);
  std::vector<u32, AlignedAllocator<u32, 64>> B_32(cols * b_cols);
  
  std::vector<u64, AlignedAllocator<u64, 64>> A_64(rows_64 * cols);
  std::vector<u64, AlignedAllocator<u64, 64>> B_64(cols * b_cols);
  
  std::vector<u32, AlignedAllocator<u32, 64>> out_32(rows_32 * b_cols);
  std::vector<u64, AlignedAllocator<u64, 64>> out_64(rows_32 * b_cols);
  std::vector<u128, AlignedAllocator<u128, 64>> out_128(rows_64 * b_cols);

  std::generate(A_32.begin(), A_32.end(), rand);
  std::generate(B_32.begin(), B_32.end(), rand);
  std::generate(A_64.begin(), A_64.end(), rand);
  std::generate(B_64.begin(), B_64.end(), rand);
  const double mat_32_MB = experiments * (A_32.size() * sizeof(u32)) / (1024.0 * 1024.0);
  const double mat_64_MB = experiments * (A_64.size() * sizeof(u64)) / (1024.0 * 1024.0);
  size_t tot_sum = 0;

  // ---------------------------------------------------------------------------------------

  TIME_START("mat_vec_32");
  for (size_t i = 0; i < experiments; i++)
    mat_vec_32(A_32.data(), B_32.data(), out_32.data(), rows_32, cols);
  tot_sum += out_32[rand() % out_32.size()];
  TIME_END("mat_vec_32");
  PRINT_THROUGHPUT("mat_vec_32", mat_32_MB);


  TIME_START("mat_vec_32_64");
  for (size_t i = 0; i < experiments; i++)
    mat_vec_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_vec_32_64");
  PRINT_THROUGHPUT("mat_vec_32_64", mat_32_MB);


  TIME_START("mat_vec_32_64_avx2");
  for (size_t i = 0; i < experiments; i++)
    mat_vec_32_64_avx2(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_vec_32_64_avx2");
  PRINT_THROUGHPUT("mat_vec_32_64_avx2", mat_32_MB);


  TIME_START("mat_vec_64");
  for (size_t i = 0; i < experiments; i++)
    mat_vec_64(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  tot_sum += out_64[rand() % rows_64];
  TIME_END("mat_vec_64");
  PRINT_THROUGHPUT("mat_vec_64", mat_64_MB);


  TIME_START("mat_vec_64_128");
  for (size_t i = 0; i < experiments; i++)
    mat_vec_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  tot_sum += out_128[rand() % rows_64];
  TIME_END("mat_vec_64_128");
  PRINT_THROUGHPUT("mat_vec_64_128", mat_64_MB);


  std::cout << "ignore this: " << tot_sum << std::endl;
  return 0;
}
