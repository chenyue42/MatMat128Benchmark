#include "TimeLogger.h"
#include "kernels.h"
#include <iostream>
#include <vector>

int main() {
  constexpr size_t experiments = 10;
  constexpr size_t cols = 473;
  constexpr size_t rows_64 = 1<<17;
  constexpr size_t rows_32 = rows_64 * 2;

  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> a_64(rows_64 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> b_64(rows_64 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> a_32(rows_32 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> b_32(rows_32 * cols);

  for (size_t i = 0; i < a_64.size(); i++) { a_64[i] = rand(); }
  for (size_t i = 0; i < b_64.size(); i++) { b_64[i] = rand(); }
  for (size_t i = 0; i < a_32.size(); i++) { a_32[i] = rand(); }
  for (size_t i = 0; i < b_32.size(); i++) { b_32[i] = rand(); }

  size_t tot_sum = 0;

  TIME_START("vec_dot_32");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_32(a_32.data(), b_32.data(), a_32.size());
  }
  TIME_END("vec_dot_32");

  TIME_START("vec_dot_32_64");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_32_64(a_32.data(), b_32.data(), a_32.size());
  }
  TIME_END("vec_dot_32_64");

  TIME_START("vec_dot_64");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_64(a_64.data(), b_64.data(), a_64.size());
  }
  TIME_END("vec_dot_64");

  TIME_START("vec_dot_64_128");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_64_128(a_64.data(), b_64.data(), a_64.size());
  }
  TIME_END("vec_dot_64_128");

  std::cout << "tot_sum: " << tot_sum << std::endl;

  const double vec_32_MB = (a_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  const double vec_64_MB = (a_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
  PRINT_THROUGHPUT("vec_dot_32", vec_32_MB * experiments);
  PRINT_THROUGHPUT("vec_dot_32_64", vec_32_MB * experiments);
  PRINT_THROUGHPUT("vec_dot_64", vec_64_MB * experiments);
  PRINT_THROUGHPUT("vec_dot_64_128", vec_64_MB * experiments);
  return 0;
}


