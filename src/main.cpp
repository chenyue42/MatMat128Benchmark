#include "TimeLogger.h"
#include "aligned_allocator.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Define uint128_t for GCC.
typedef unsigned __int128 uint128_t;

constexpr size_t b_cols = 2;  // BFV has two columns.


void simple_read_32(uint32_t *__restrict A, const size_t size, const uint32_t val) {
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    A[i] += val;
  }
}

// ./mlc --memory_bandwidth_scan -t1
// 1MB = 1,000,000 bytes
// result is about 13000 MB = 13000 * 1,000,000 / (2^30) ~= 12.1 GB
void simple_read(uint64_t *__restrict A, const size_t size, const uint64_t val) {
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    A[i] += val;
  }
}

void simple_read_128(uint128_t *__restrict A, const size_t size, const uint128_t val) {
  for (size_t i = 0; i < size; i++) {
    A[i] += val;
  }
}

void mat_vec_fp(const float *const __restrict A,
                const float *const __restrict B, float *const __restrict out,
                const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    float tmp = 0;
    const size_t offset = i * cols;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}

void mat_vec_double(const double *const __restrict A,
                const double *const __restrict B, double *const __restrict out,
                const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    double tmp = 0;
    const size_t offset = i * cols;
#pragma GCC unroll 64
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}


void mat_vec_32(const uint32_t *const __restrict A,
                const uint32_t *const __restrict B, uint32_t *__restrict out,
                const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint32_t tmp = 0;
    const size_t offset = i * cols;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}

void mat_vec_32_64(const uint32_t *const __restrict A,
                   const uint32_t *const __restrict B, uint64_t *__restrict out,
                   const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint64_t tmp = 0;
    const size_t offset = i * cols;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * (uint64_t)B[k];
    }
    out[i] = tmp;
  }
}


void mat_vec_64(const uint64_t *const __restrict A,
                const uint64_t *const __restrict B, uint64_t *__restrict out,
                const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint64_t tmp = 0;
    const size_t offset = i * cols;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}

void mat_vec_64_128(const uint64_t *const __restrict A,
                 const uint64_t *const __restrict B, uint128_t *__restrict out,
                 const size_t rows, const size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    uint128_t tmp = 0;
    const size_t offset = i * cols;
#pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      tmp += (uint128_t)A[offset + k] * B[k];
    }
    out[i] = tmp;
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
      t0 += A[offset + k] * B[b_cols * k];
      t1 += A[offset + k] * B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

void mat_mat_64_128(const uint64_t *__restrict A, const uint64_t *__restrict B,
                 uint128_t *__restrict out, const size_t rows,
                 const size_t cols) {
  uint128_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += A[offset + k] * (uint128_t)B[b_cols * k];
      t1 += A[offset + k] * (uint128_t)B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
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
  constexpr size_t experiments = 3;
  constexpr size_t cols = 1<<8;
  constexpr size_t rows = 1<<19;
  constexpr size_t rows_32 = 1<<20; // for 32-bit
  double size_32_MB = (rows_32 * cols * sizeof(uint32_t)) / (1024.0 * 1024.0);
  double size_64_MB = (rows * cols * sizeof(uint64_t)) / (1024.0 * 1024.0);
  double size_128_MB = (rows * cols * sizeof(uint128_t)) / (1024.0 * 1024.0);
  double size_f_MB = (rows_32 * cols * sizeof(float)) / (1024.0 * 1024.0);
  double size_d_MB = (rows * cols * sizeof(double)) / (1024.0 * 1024.0);
  std::cout << "rows: " << rows << ", cols: " << cols << std::endl;
  std::cout << "Matrix A 32b Size: " << size_32_MB << " MB" << std::endl;
  std::cout << "Matrix A 64b Size: " << size_64_MB << " MB" << std::endl;
  // std::cout << "Matrix A 128b Size: " << size_128_MB << " MB" << std::endl;
  std::cout << "Matrix A float Size: " << size_f_MB << " MB" << std::endl;
  std::cout << "Matrix A double Size: " << size_d_MB << " MB" << std::endl;

  // Allocate matrices with the aligned allocator.
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64(rows * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> B_64(b_cols * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> out_64(b_cols * rows);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32(rows_32 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> B_32(b_cols * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> out_32(b_cols * rows_32);

  std::vector<float, AlignedAllocator<float, 64>> A_f(rows_32 * cols);
  std::vector<float, AlignedAllocator<float, 64>> B_f(b_cols * cols);
  std::vector<float, AlignedAllocator<float, 64>> out_f(b_cols * rows_32);

  std::vector<double, AlignedAllocator<double, 64>> A_d(rows * cols);
  std::vector<double, AlignedAllocator<double, 64>> B_d(b_cols * cols);
  std::vector<double, AlignedAllocator<double, 64>> out_d(b_cols * rows);
  
  // std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> A_128(rows * cols);
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> out_128(b_cols * rows);

  // print the address of the first element of each vector
  std::cout << "Address of A: " << static_cast<void*>(A_64.data()) << std::endl;
  std::cout << "Address of B: " << static_cast<void*>(B_64.data()) << std::endl;
  std::cout << "Address of out64: " << static_cast<void*>(out_64.data()) << std::endl;

  // Initialize matrices with random values
  for (size_t i = 0; i < A_64.size(); i++) { A_64[i] = rand(); }
  for (size_t i = 0; i < B_64.size(); i++) { B_64[i] = rand(); }
  for (size_t i = 0; i < A_32.size(); i++) { A_32[i] = rand(); }
  for (size_t i = 0; i < B_32.size(); i++) { B_32[i] = rand(); }
  for (size_t i = 0; i < A_f.size(); i++) { A_f[i] = static_cast<float>(rand()) / RAND_MAX; }
  for (size_t i = 0; i < B_f.size(); i++) { B_f[i] = static_cast<float>(rand()) / RAND_MAX; }
  for (size_t i = 0; i < A_d.size(); i++) { A_d[i] = static_cast<double>(rand()) / RAND_MAX; }
  for (size_t i = 0; i < B_d.size(); i++) { B_d[i] = static_cast<double>(rand()) / RAND_MAX; }
  // for (size_t i = 0; i < A_128.size(); i++) { A_128[i] = rand(); }



  // ================== Simple read 32-bit.
  TIME_START("simple_read_32");
  for (int i = 0; i < experiments; i++)
    simple_read_32(A_32.data(), A_32.size(), rand());
  TIME_END("simple_read_32");

  // ================== Simple read.
  TIME_START("simple_read_64");
  for (int i = 0; i < experiments; i++)
    simple_read(A_64.data(), A_64.size(), rand());
  TIME_END("simple_read_64");

  // // ================== Simple read 128-bit.
  // TIME_START("simple_read_128");
  // for (int i = 0; i < experiments; i++)
  //   simple_read_128(A_128.data(), A_64.size(), rand());
  // TIME_END("simple_read_128");

  // ================== float matrix vector multiplication.
  std::cout << "sizeof(float): " << sizeof(float) << std::endl;
  TIME_START("mat_vec_float");
  for (int i = 0; i < experiments; i++)
    mat_vec_fp(A_f.data(), B_f.data(), out_f.data(), rows, cols);
  TIME_END("mat_vec_float");

  // ================== double matrix vector multiplication.
  std::cout << "sizeof(double): " << sizeof(double) << std::endl;
  TIME_START("mat_vec_double");
  for (int i = 0; i < experiments; i++)
    mat_vec_double(A_d.data(), B_d.data(), out_d.data(), rows, cols);
  TIME_END("mat_vec_double");

  // ================== 32-bit matrix vector multiplication.
  TIME_START("mat_vec_32");
  for (int i = 0; i < experiments; i++)
    mat_vec_32(A_32.data(), B_32.data(), out_32.data(), rows, cols);
  TIME_END("mat_vec_32");

  // ================== 32-bit matrix vector multiplication (64-bit).
  TIME_START("mat_vec_32_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_32_64(A_32.data(), B_32.data(), out_64.data(), rows, cols);
  TIME_END("mat_vec_32_64");

  // ================== 64-bit matrix vector multiplication.
  TIME_START("mat_vec_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_64(A_64.data(), B_64.data(), out_64.data(), rows, cols);
  TIME_END("mat_vec_64");

  // ================== 128-bit matrix vector multiplication.
  TIME_START("mat_vec_64_128");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_128(A_64.data(), B_64.data(), out_128.data(), rows, cols);
  TIME_END("mat_vec_64_128");

  // ================== 64-bit matrix multiplication.
  TIME_START("mat_mat_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_64(A_64.data(), B_64.data(), out_64.data(), rows, cols);
  TIME_END("mat_mat_64");

  // ================== Naive matrix multiplication (128-bit).
  TIME_START("mat_mat_64_128");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_128(A_64.data(), B_64.data(), out_128.data(), rows, cols);
  TIME_END("mat_mat_64_128");

  // ================== Eigen matrix multiplication.
  TIME_START("mat_mat_eigen");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_Eigen(A_64.data(), B_64.data(), out_64.data(), rows, cols);
  TIME_END("mat_mat_eigen");

  // ================== Eigen matrix vector multiplication.
  TIME_START("mat_vec_eigen");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_Eigen(A_64.data(), B_64.data(), out_64.data(), rows, cols);
  TIME_END("mat_vec_eigen");

  // ================== Performance analysis.
  const size_t db_sz_64 = size_64_MB * experiments;  // 8 bytes per element
  const size_t db_sz_32 = size_32_MB * experiments;  // 4 bytes per element
  const size_t db_sz_128 = size_128_MB * experiments; // 16 bytes per element
  const size_t db_sz_f = size_f_MB * experiments;   // 4 bytes per element
  const size_t db_sz_d = size_d_MB * experiments;   // 8 bytes per element
  std::cout << "====== reading DB and add constant =======" << std::endl;
  PRINT_THROUGHPUT("simple_read_32", db_sz_32);
  PRINT_THROUGHPUT("simple_read_64", db_sz_64);
  // PRINT_THROUGHPUT("simple_read_128", db_sz_128);
  std::cout << "====== matrix * vector =======" << std::endl;
  PRINT_THROUGHPUT("mat_vec_float", db_sz_f);
  PRINT_THROUGHPUT("mat_vec_double", db_sz_d);
  PRINT_THROUGHPUT("mat_vec_32", db_sz_32);
  PRINT_THROUGHPUT("mat_vec_32_64", db_sz_32);
  PRINT_THROUGHPUT("mat_vec_64", db_sz_64);
  PRINT_THROUGHPUT("mat_vec_64_128", db_sz_64);
  std::cout << "====== matrix * 2 columns =======" << std::endl;
  PRINT_THROUGHPUT("mat_mat_64", db_sz_64);
  PRINT_THROUGHPUT("mat_mat_64_128", db_sz_64);
  PRINT_THROUGHPUT("mat_mat_eigen", db_sz_64);
  PRINT_THROUGHPUT("mat_vec_eigen", db_sz_64);
  
  return 0;
}
