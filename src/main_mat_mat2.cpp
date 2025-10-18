#include "TimeLogger.h"
#include "kernels.h"
#include <iostream>
#include <vector>

int main() {
  constexpr size_t experiments = 10;
  constexpr size_t cols = 473;
  constexpr size_t rows_64 = 1<<17;
  constexpr size_t rows_32 = rows_64 * 2;
  constexpr size_t bcols = 2;

  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64(rows_64 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> B_64(cols * bcols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> out_64(rows_32 * bcols);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32(rows_32 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> B_32(cols * bcols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> out_32(rows_32 * bcols);
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> out_128(rows_64 * bcols);

  for (size_t i = 0; i < A_64.size(); i++) { A_64[i] = rand(); }
  for (size_t i = 0; i < B_64.size(); i++) { B_64[i] = rand(); }
  for (size_t i = 0; i < A_32.size(); i++) { A_32[i] = rand(); }
  for (size_t i = 0; i < B_32.size(); i++) { B_32[i] = rand(); }


  std::cout << "A_32 size in MB: " << (A_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0) << std::endl;

  size_t tot_sum = 0;

  TIME_START("mat_mat_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_64(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  tot_sum += out_64[rand() % rows_64];
  TIME_END("mat_mat_64");

  TIME_START("mat_mat_64_128");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  tot_sum += out_128[rand() % out_128.size()];
  TIME_END("mat_mat_64_128");

  TIME_START("mat_mat_32_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64");

  // ================== Vertical-blocked AVX2-ish matrix multiplication (32->64-bit).
  TIME_START("mat_mat_vert_32_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_vert_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_vert_32_64");

  TIME_START("mat_mat_vert8_scalar");
  for (int i = 0; i < experiments; i++)
    mat_mat_vert8_scalar(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_vert8_scalar");

  TIME_START("mat_mat_vert8_avx512");
  for (int i = 0; i < experiments; i++)
    mat_mat_vert8_avx512(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_vert8_avx512");


  TIME_START("mat_mat_32_64_Eigen");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64_Eigen(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64_Eigen");

  TIME_START("mat_mat_32_64_avx2");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64_avx2(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64_avx2");

  TIME_START("mat_mat_32_64_avx512");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64_avx512(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64_avx512");

  std::cout << "tot_sum: " << tot_sum << std::endl;

  const double mat_32_MB = experiments * (A_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  const double mat_64_MB = experiments * (A_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
  PRINT_THROUGHPUT("mat_mat_64", mat_64_MB);
  PRINT_THROUGHPUT("mat_mat_64_128", mat_64_MB);
  PRINT_THROUGHPUT("mat_mat_32_64", mat_32_MB);
  PRINT_THROUGHPUT("mat_mat_vert_32_64", mat_32_MB);
  PRINT_THROUGHPUT("mat_mat_vert8_scalar", mat_32_MB);
  PRINT_THROUGHPUT("mat_mat_vert8_avx512", mat_32_MB);
  PRINT_THROUGHPUT("mat_mat_32_64_Eigen", mat_32_MB);
  PRINT_THROUGHPUT("mat_mat_32_64_avx2", mat_32_MB);
  PRINT_THROUGHPUT("mat_mat_32_64_avx512", mat_32_MB);
  return 0;
}


