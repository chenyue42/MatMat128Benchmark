#include "TimeLogger.h"
#include "aligned_allocator.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
// #include <blaze/Math.h>

// Define uint128_t for GCC.
typedef unsigned __int128 uint128_t;

// ./mlc --memory_bandwidth_scan -t1
// 1MB = 1,000,000 bytes
// result is about 13000 MB = 13000 * 1,000,000 / (2^30) ~= 12.1 GB
void simple_read(uint64_t *__restrict A, const size_t size, const uint64_t val) {
  for (size_t i = 0; i < size; i++) {
    A[i] += val;
  }
}

void simple_read_128(uint128_t *__restrict A, const size_t size, const uint128_t val) {
  for (size_t i = 0; i < size; i++) {
    A[i] += val;
  }
}

void mat_vec_64(const uint64_t *const __restrict A,
                const uint64_t *const __restrict B, uint64_t *__restrict out,
                const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint64_t t = 0;
    const size_t offset = i * cols;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      t += A[offset + k] * B[k];
    }
    out[i] = t;
  }
}

void mat_vec_128(const uint64_t *const __restrict A,
                 const uint64_t *const __restrict B, uint128_t *__restrict out,
                 const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint128_t t = 0;
    const size_t offset = i * cols;
#pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t += (uint128_t)A[offset + k] * B[k];
    }
    out[i] = t;
  }
}

void mat_mat_64(const uint64_t *__restrict A, const uint64_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
#pragma GCC ivdep unroll 128
    for (size_t k = 0; k < cols; k++) {
      t0 += A[offset + k] * B[2 * k];
      t1 += A[offset + k] * B[2 * k + 1];
    }
    out[2 * i] = t0;
    out[2 * i + 1] = t1;
  }
}

void mat_mat_128(const uint64_t *__restrict A, const uint64_t *__restrict B,
                 uint128_t *__restrict out, const size_t rows,
                 const size_t cols) {
  uint128_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += A[i * cols + k] * (uint128_t)(B[2 * k]);
      t1 += A[i * cols + k] * (uint128_t)(B[2 * k + 1]);
    }
    out[2 * i] = t0;
    out[2 * i + 1] = t1;
  }
}

void mat_mat_64_Eigen(const uint64_t *A, const uint64_t *B, uint64_t *out,
                      size_t rows, size_t cols) {
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, cols, 2);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, 2, Eigen::RowMajor>> matOut(out, rows, 2);
    
    matOut.noalias() = matA * matB;
}

void mat_vec_64_Eigen(const uint64_t *A, const uint64_t *B, uint64_t *out,
                      size_t rows, size_t cols) {
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols); Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>> matB(B, cols);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>> matOut(out, rows);
    
    matOut.noalias() = matA * matB;
}

int main() {
  constexpr size_t experiments = 5;
  constexpr size_t rows = 1<<15;
  constexpr size_t cols = 1<<9;
  double size_MB = (rows * cols * sizeof(uint64_t)) / (1024.0 * 1024.0);
  std::cout << "rows: " << rows << ", cols: " << cols << std::endl;
  std::cout << "Matrix A Size: " << size_MB << " MB" << std::endl;

  // Allocate matrices with the aligned allocator.
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A(rows * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> B(2 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> out64(2 * rows);
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> out128(2 * rows);
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> A128(rows * cols);

  // print the address of the first element of each vector
  std::cout << "Address of A: " << static_cast<void*>(A.data()) << std::endl;
  std::cout << "Address of B: " << static_cast<void*>(B.data()) << std::endl;
  std::cout << "Address of out64: " << static_cast<void*>(out64.data()) << std::endl;

  // Initialize matrices with random values
  for (size_t i = 0; i < A.size(); i++) { A[i] = rand(); }
  for (size_t i = 0; i < B.size(); i++) { B[i] = rand(); }
  for (size_t i = 0; i < A128.size(); i++) { A128[i] = rand(); }

  // ================== Simple read.
  for (int i = 0; i < 3; i++)
    simple_read(A.data(), A.size(), rand());
  TIME_START("simple_read");
  for (int i = 0; i < experiments; i++)
    simple_read(A.data(), A.size(), rand());
  TIME_END("simple_read");

  // ================== Simple read 128-bit.
  for (int i = 0; i < 3; i++)
    simple_read_128(A128.data(), A.size(), rand());
  TIME_START("simple_read_128");
  for (int i = 0; i < experiments; i++)
    simple_read_128(A128.data(), A.size(), rand());
  TIME_END("simple_read_128");

  // ================== 64-bit matrix vector multiplication.
  TIME_START("mat_vec_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_64(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_vec_64");

  // ================== 128-bit matrix vector multiplication.
  TIME_START("mat_vec_128");
  for (int i = 0; i < experiments; i++)
    mat_vec_128(A.data(), B.data(), out128.data(), rows, cols);
  TIME_END("mat_vec_128");

  // ================== 64-bit matrix multiplication.
  TIME_START("mat_mat_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_64(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_mat_64");

  // ================== Naive matrix multiplication (128-bit).
  TIME_START("mat_mat_128");
  for (int i = 0; i < experiments; i++)
    mat_mat_128(A.data(), B.data(), out128.data(), rows, cols);
  TIME_END("mat_mat_128");


  // ================== Eigen matrix multiplication.
  TIME_START("mat_mat_eigen");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_Eigen(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_mat_eigen");

  // ================== Eigen matrix vector multiplication.
  TIME_START("mat_vec_eigen");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_Eigen(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_vec_eigen");

  // ================== Performance analysis.
  const size_t tot_size = size_MB * experiments;
  PRINT_THROUGHPUT("simple_read", tot_size);
  PRINT_THROUGHPUT("simple_read_128", 2 * tot_size);
  PRINT_THROUGHPUT("mat_vec_64", tot_size);
  PRINT_THROUGHPUT("mat_vec_128", tot_size);
  PRINT_THROUGHPUT("mat_mat_64", tot_size);
  PRINT_THROUGHPUT("mat_mat_128", tot_size);
  PRINT_THROUGHPUT("mat_mat_eigen", tot_size);
  PRINT_THROUGHPUT("mat_vec_eigen", tot_size);
  
  return 0;
}
