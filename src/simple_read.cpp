#include "aligned_allocator.h"
#include "TimeLogger.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <algorithm>

u32 simple_read_32(const u32 *const __restrict a, const size_t size) {
  u32 sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}

static inline u64 simple_read_64(const u64 *const __restrict a, const size_t size) {
  u64 sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}


int main() {

  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64(rows_64 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32(rows_32 * cols);
  std::generate(A_32.begin(), A_32.end(), std::rand);
  std::generate(A_64.begin(), A_64.end(), std::rand);
  const double mat_32_MB = experiments * (A_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  const double mat_64_MB = experiments * (A_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);

  size_t tot_sum = 0;
  std::cout << "A_32.size(): " << A_32.size() << std::endl;

  TIME_START("simple_read_32");
  for (int i = 0; i < experiments; i++)
    tot_sum += simple_read_32(A_32.data(), A_32.size());
  TIME_END("simple_read_32");
  std::cout << "Matrix A 32b Size: " << mat_32_MB << " MB" << std::endl;
  PRINT_THROUGHPUT("simple_read_32", mat_32_MB);



  TIME_START("simple_read_64");
  for (int i = 0; i < experiments; i++)
    tot_sum += simple_read_64(A_64.data(), A_64.size());
  TIME_END("simple_read_64");
  std::cout << "Matrix A 64b Size: " << mat_64_MB << " MB" << std::endl;  
  PRINT_THROUGHPUT("simple_read_64", mat_64_MB);


  std::cout << "ignore this: " << tot_sum << std::endl;
  return 0;
}


