#include "TimeLogger.h"
#include <cstdlib>
#include <iostream>
#include <vector>

// Define uint128_t for GCC.
typedef unsigned __int128 uint128_t;


void simple_read(uint64_t *__restrict A, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    A[i] ^= 0xdeadbeef;
  }
}

// Matrix multiplication function.
// Computes out = A * B, where B is assumed to have two columns.
void mat_mat_128(const uint64_t *__restrict A, const uint64_t *__restrict B,
                 uint128_t *__restrict out, const size_t rows,
                 const size_t cols) {
  uint128_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0;
    t1 = 0;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += A[i * cols + k] * (uint128_t)B[2 * k];
      t1 += A[i * cols + k] * (uint128_t)B[2 * k + 1];
    }
    out[2 * i] = t0;
    out[2 * i + 1] = t1;
  }
}

int main() {
  // Matrix dimensions.
  const size_t rows = 1<<19;
  const size_t cols = 1<<8;

  // Allocate matrices:
  // - A: rows x cols (each element is 8 bytes)
  // - B: has 2 columns (stored as a flat array of length 2*cols)
  // - out: output array of size 2*rows
  std::vector<uint64_t> A(rows * cols);
  std::vector<uint64_t> B(2 * cols);
  std::vector<uint128_t> out(2 * rows);

  // Initialize matrices with random values.
  for (size_t i = 0; i < A.size(); i++) { A[i] = rand(); }
  for (size_t i = 0; i < B.size(); i++) { B[i] = rand(); }


  // Start timing the simple read.
  TIME_START("simple_read");
  simple_read(A.data(), A.size());
  TIME_END("simple_read");

  // Start timing the matrix multiplication.
  TIME_START("mat_mat_128");
  mat_mat_128(A.data(), B.data(), out.data(), rows, cols);
  TIME_END("mat_mat_128");

  // Compute the size of matrix A in megabytes.
  double size_MB = (rows * cols * sizeof(uint64_t)) / (1024.0 * 1024.0);

  // Get the elapsed time in seconds.
  double simple_read_ms = GET_TIME("simple_read");
  double mat_mat_ms = GET_TIME("mat_mat_128");

  // Throughput in MB/s.
  double simple_read_throughput = size_MB / simple_read_ms * 1000.0;
  double mat_mat_throughput = size_MB / mat_mat_ms * 1000.0;

  std::cout << "Matrix A Size: " << size_MB << " MB" << std::endl;
  std::cout << "simple read throughput: " << simple_read_throughput << " MB/s" << std::endl;
  std::cout << "mat-mat throughput: " << mat_mat_throughput << " MB/s" << std::endl;

  return 0;
}
