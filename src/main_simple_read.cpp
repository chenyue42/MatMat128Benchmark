#include "TimeLogger.h"
#include "kernels.h"
#include <iostream>
#include <vector>

int main() {
  constexpr size_t experiments = 10;
  constexpr size_t cols = 473;
  constexpr size_t rows_64 = 1<<17;
  constexpr size_t rows_32 = rows_64 * 2;

  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64(rows_64 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32(rows_32 * cols);

  for (size_t i = 0; i < A_64.size(); i++) { A_64[i] = rand(); }
  for (size_t i = 0; i < A_32.size(); i++) { A_32[i] = rand(); }

  size_t tot_sum = 0;
  std::cout << "A_32.size(): " << A_32.size() << std::endl;

  TIME_START("simple_read_32");
  for (int i = 0; i < experiments; i++)
    tot_sum += simple_read_32(A_32.data(), A_32.size());
  TIME_END("simple_read_32");

  TIME_START("simple_read_64");
  for (int i = 0; i < experiments; i++)
    tot_sum += simple_read_64(A_64.data(), A_64.size());
  TIME_END("simple_read_64");

  std::cout << "tot_sum: " << tot_sum << std::endl;

  const double mat_32_MB = (A_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  const double mat_64_MB = (A_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
  std::cout << "rows: " << rows_64 << ", cols: " << cols << std::endl;
  std::cout << "Matrix A 32b Size: " << mat_32_MB << " MB" << std::endl;
  std::cout << "Matrix A 64b Size: " << mat_64_MB << " MB" << std::endl;

  PRINT_THROUGHPUT("simple_read_32", mat_32_MB * experiments);
  PRINT_THROUGHPUT("simple_read_64", mat_64_MB * experiments);
  return 0;
}


