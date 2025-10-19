#include "aligned_allocator.h"
#include "TimeLogger.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <algorithm>


u32 vec_dot_32(const u32 *const __restrict a, const u32 *const __restrict b, const size_t size) {
  uint32_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;

}
u64 vec_dot_32_64(const u32 *const __restrict a, const u32 *const __restrict b,
                  const size_t size) {
  u64 sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += (u64)a[i] * (u64)b[i];
  }
  return sum;
}

u64 vec_dot_64(const u64 *const __restrict a, const u64 *const __restrict b,
               const size_t size) {
  u64 sum = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

u128 vec_dot_64_128(const u64 *const __restrict a,
                    const u64 *const __restrict b, const size_t size) {
  u128 sum = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += (u128)a[i] * b[i];
  }
  return sum;
}

int main() {
  constexpr size_t experiments = 10;
  constexpr size_t cols = 473;
  constexpr size_t rows_64 = 1 << 17;
  constexpr size_t rows_32 = rows_64 * 2;

  std::vector<u32, AlignedAllocator<u32, 64>> a_32(rows_32 * cols);
  std::vector<u32, AlignedAllocator<u32, 64>> b_32(rows_32 * cols);
  std::vector<u64, AlignedAllocator<u64, 64>> a_64(rows_64 * cols);
  std::vector<u64, AlignedAllocator<u64, 64>> b_64(rows_64 * cols);
  std::generate(a_32.begin(), a_32.end(), std::rand);
  std::generate(b_32.begin(), b_32.end(), std::rand);
  std::generate(a_64.begin(), a_64.end(), std::rand);
  std::generate(b_64.begin(), b_64.end(), std::rand);
  const double vec_32_MB = experiments * (a_32.size() * sizeof(u32)) / (1024.0 * 1024.0);
  const double vec_64_MB = experiments * (a_64.size() * sizeof(u64)) / (1024.0 * 1024.0);

  size_t tot_sum = 0;

  TIME_START("vec_dot_32");
  for (size_t i = 0; i < experiments; i++)
    tot_sum += vec_dot_32(a_32.data(), b_32.data(), a_32.size());
  TIME_END("vec_dot_32");
  PRINT_THROUGHPUT("vec_dot_32", vec_32_MB);

  TIME_START("vec_dot_32_64");
  for (size_t i = 0; i < experiments; i++)
    tot_sum += vec_dot_32_64(a_32.data(), b_32.data(), a_32.size());
  TIME_END("vec_dot_32_64");
  PRINT_THROUGHPUT("vec_dot_32_64", vec_32_MB);

  TIME_START("vec_dot_64");
  for (size_t i = 0; i < experiments; i++)
    tot_sum += vec_dot_64(a_64.data(), b_64.data(), a_64.size());
  TIME_END("vec_dot_64");
  PRINT_THROUGHPUT("vec_dot_64", vec_64_MB);

  TIME_START("vec_dot_64_128");
  for (size_t i = 0; i < experiments; i++)
    tot_sum += vec_dot_64_128(a_64.data(), b_64.data(), a_64.size());
  TIME_END("vec_dot_64_128");
  PRINT_THROUGHPUT("vec_dot_64_128", vec_64_MB);

  std::cout << "tot_sum: " << tot_sum << std::endl;

  return 0;
}
