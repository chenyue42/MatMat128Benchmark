#include "TimeLogger.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <blaze/Math.h>


// Define uint128_t for GCC.
typedef unsigned __int128 uint128_t;

struct uint128_sim {
  uint64_t lo;
  uint64_t hi;
};

inline uint128_sim mul64(uint64_t a, uint64_t b) {
  // Split a and b into two 32-bit parts.
  uint64_t a_lo = (uint32_t)a;
  uint64_t a_hi = a >> 32;
  uint64_t b_lo = (uint32_t)b;
  uint64_t b_hi = b >> 32;

  // Multiply the parts.
  uint64_t p0 = a_lo * b_lo;
  uint64_t p1 = a_lo * b_hi;
  uint64_t p2 = a_hi * b_lo;
  uint64_t p3 = a_hi * b_hi;

  // Sum the cross products.
  uint64_t mid = p1 + p2;
  // Check if there was a carry in the cross terms.
  uint64_t carry_mid = (mid < p1) ? 1ULL : 0;

  // Add the lower 32 bits of mid (shifted left by 32) to p0.
  uint64_t lo = p0 + (mid << 32);
  uint64_t carry_lo = (lo < p0) ? 1ULL : 0;

  // The high part is:
  uint64_t hi = p3 + (mid >> 32) + carry_mid + carry_lo;

  return {lo, hi};
}

inline uint128_sim add128(const uint128_sim &a, const uint128_sim &b) {
    uint64_t lo = a.lo + b.lo;
    uint64_t carry = (lo < a.lo) ? 1ULL : 0;
    uint64_t hi = a.hi + b.hi + carry;
    return { lo, hi };
}


void simple_read(uint64_t *__restrict A, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    A[i] ^= 0xdeadbeef;
  }
}

void mat_vec_64(const uint64_t *const __restrict A,
                const uint64_t *const __restrict B, uint64_t *__restrict out,
                const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint64_t t = 0;
    const size_t offset = i * cols;
// #pragma GCC unroll 128
    #pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      t += A[offset + k] * B[k];
    }
    out[i] = t;
  }
}

void mat_mat_64(const uint64_t *__restrict A, const uint64_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0;
    t1 = 0;
    #pragma GCC unroll 64
    for (size_t k = 0; k < cols; k++) {
      t0 += A[i * cols + k] * B[2 * k];
      t1 += A[i * cols + k] * B[2 * k + 1];
    }
    out[2 * i] = t0;
    out[2 * i + 1] = t1;
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
    #pragma GCC unroll 64
    for (size_t k = 0; k < cols; k++) {
      t0 += A[i * cols + k] * (uint128_t)B[2 * k];
      t1 += A[i * cols + k] * (uint128_t)B[2 * k + 1];
    }
    out[2 * i] = t0;
    out[2 * i + 1] = t1;
  }
}

void mat_mat_128_sim(const uint64_t *__restrict A, const uint64_t *__restrict B,
                     uint128_sim *__restrict out, const size_t rows,
                     const size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    uint128_sim t0 = {0, 0};
    uint128_sim t1 = {0, 0};
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      // Multiply A[i*cols+k] with the corresponding B entries using simulated
      // 128-bit multiplication.
      uint128_sim prod0 = mul64(A[i * cols + k], B[2 * k]);
      uint128_sim prod1 = mul64(A[i * cols + k], B[2 * k + 1]);

      // Accumulate the products.
      t0 = add128(t0, prod0);
      t1 = add128(t1, prod1);
    }
    out[2 * i] = t0;
    out[2 * i + 1] = t1;
  }
}

void mat_mat_64_Eigen(const uint64_t *A, const uint64_t *B, uint64_t *out,
                      size_t rows, size_t cols) {
    // Map A as a rows x cols matrix.
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
    // Map B as a cols x 2 matrix.
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, cols, 2);
    // Directly map the output buffer as a rows x 2 matrix.
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, 2, Eigen::RowMajor>> matOut(out, rows, 2);
    
    // Perform the multiplication directly into the mapped output.
    matOut.noalias() = matA * matB;
}

void mat_vec_64_Eigen(const uint64_t *A, const uint64_t *B, uint64_t *out,
                      size_t rows, size_t cols) {
    // Map A as a rows x cols matrix.
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
    // Map B as a cols x 1 matrix.
    Eigen::Map<const Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>> matB(B, cols);
    // Directly map the output buffer as a rows x 1 matrix.
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>> matOut(out, rows);
    
    // Perform the multiplication directly into the mapped output.
    matOut.noalias() = matA * matB;
}

int main() {
  // Matrix dimensions.
  constexpr size_t rows = 1<<19;
  constexpr size_t cols = 1<<8;
  constexpr size_t experiments = 10;

  // Allocate matrices:
  // - A: rows x cols (each element is 8 bytes)
  // - B: has 2 columns (stored as a flat array of length 2*cols)
  // - out: output array of size 2*rows
  std::vector<uint64_t> A(rows * cols);
  std::vector<uint64_t> B(2 * cols);
  std::vector<uint64_t> out64(2 * rows);
  std::vector<uint128_t> out(2 * rows);
  std::vector<uint128_sim> out_sim(2 * rows);

  // Initialize matrices with random values.
  for (size_t i = 0; i < A.size(); i++) { A[i] = rand(); }
  for (size_t i = 0; i < B.size(); i++) { B[i] = rand(); }

  // ================== Simple read.
  TIME_START("simple_read");
  for (int i = 0; i < experiments; i++)
    simple_read(A.data(), A.size());
  TIME_END("simple_read");

  // ================== 64-bit matrix vector multiplication.
  TIME_START("mat_vec_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_64(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_vec_64");

  // ================== 64-bit matrix multiplication.
  TIME_START("mat_mat_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_64(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_mat_64");

  // ================== Naive matrix multiplication.
  TIME_START("mat_mat_128");
  for (int i = 0; i < experiments; i++)
    mat_mat_128(A.data(), B.data(), out.data(), rows, cols);
  TIME_END("mat_mat_128");

  // ================== Simulated 128-bit multiplication.
  TIME_START("mat_mat_128_sim");
  for (int i = 0; i < experiments; i++)
    mat_mat_128_sim(A.data(), B.data(), out_sim.data(), rows, cols);
  TIME_END("mat_mat_128_sim");

  // ================== Eigen matrix multiplication.
  TIME_START("mat_mat_eigen");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_Eigen(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_mat_eigen");

  // ================== Eigen matrix vec multiplication.
  TIME_START("mat_vec_eigen");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_Eigen(A.data(), B.data(), out64.data(), rows, cols);
  TIME_END("mat_vec_eigen");

  // ================== blaze matrix multiplication.
  // create blaze matrices
  blaze::DynamicMatrix<uint64_t> blaze_A(rows, cols, A.data());
  blaze::DynamicMatrix<uint64_t> blaze_B(cols, 2, B.data());
  blaze::DynamicMatrix<uint64_t> blaze_out(rows, 2, out64.data());
  TIME_START("mat_mat_blaze");
  for (int i = 0; i < experiments; i++)
    blaze_out = blaze_A * blaze_B;
  TIME_END("mat_mat_blaze");


  // ================== Performance analysis.
  // Compute the size of matrix A in megabytes.
  double size_MB = (rows * cols * sizeof(uint64_t)) / (1024.0 * 1024.0);

  // Get the elapsed time in seconds.
  double simple_read_ms = GET_TIME("simple_read");
  double mat_vec_ms = GET_TIME("mat_vec_64");
  double mat_mat64_ms = GET_TIME("mat_mat_64");
  double mat_mat_ms = GET_TIME("mat_mat_128");
  double mat_mat_sim_ms = GET_TIME("mat_mat_128_sim");
  double mat_mat_eigen_ms = GET_TIME("mat_mat_eigen");
  double mat_vec_eigen_ms = GET_TIME("mat_vec_eigen");
  double mat_mat_blaze_ms = GET_TIME("mat_mat_blaze");

  // Throughput in MB/s.
  double simple_read_throughput = size_MB * experiments / simple_read_ms * 1000.0;
  double mat_vec_throughput = size_MB * experiments / mat_vec_ms * 1000.0;
  double mat_mat64_throughput = size_MB * experiments / mat_mat64_ms * 1000.0;
  double mat_mat_throughput = size_MB * experiments/ mat_mat_ms * 1000.0;
  double mat_mat_sim_throughput = size_MB * experiments / mat_mat_sim_ms * 1000.0;
  double mat_mat_eigen_throughput = size_MB * experiments / mat_mat_eigen_ms * 1000.0;
  double mat_vec_eigen_throughput = size_MB * experiments / mat_vec_eigen_ms * 1000.0;
  double mat_mat_blaze_throughput = size_MB * experiments / mat_mat_blaze_ms * 1000.0;

  std::cout << "Matrix A Size: " << size_MB << " MB" << std::endl;
  std::cout << "simple read throughput: " << simple_read_throughput << " MB/s" << std::endl;
  std::cout << "mat-vec throughput: " << mat_vec_throughput << " MB/s" << std::endl;
  std::cout << "mat-mat-64 throughput: " << mat_mat64_throughput << " MB/s" << std::endl;
  std::cout << "mat-mat throughput: " << mat_mat_throughput << " MB/s" << std::endl;
  std::cout << "mat-mat-sim throughput: " << mat_mat_sim_throughput << " MB/s" << std::endl;
  std::cout << "mat-mat-eigen throughput: " << mat_mat_eigen_throughput << " MB/s" << std::endl;
  std::cout << "mat-vec-eigen throughput: " << mat_vec_eigen_throughput << " MB/s" << std::endl;
  std::cout << "mat-mat-blaze throughput: " << mat_mat_blaze_throughput << " MB/s" << std::endl;

  return 0;
}
