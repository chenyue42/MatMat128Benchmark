#include "TimeLogger.h"
#include "aligned_allocator.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

// Define uint128_t for GCC.
typedef unsigned __int128 uint128_t;

constexpr size_t b_cols = 2;  // BFV has two columns.

// ./mlc --memory_bandwidth_scan -t1
// 1MB = 1,000,000 bytes
// result is about 13000 MB = 13000 * 1,000,000 / (2^30) ~= 12.1 GB
uint32_t simple_read_32(const uint32_t *const __restrict A, const size_t size) {
  uint32_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}


uint64_t simple_read_64(const uint64_t *const __restrict A, const size_t size) {
  uint64_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}

float simple_read_float(const float *const __restrict A, const size_t size) {
  float sum = 0;
  // #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}

void simple_read_128(uint128_t *__restrict A, const size_t size, const uint128_t val) {
  for (size_t i = 0; i < size; i++) {
    A[i] += val;
  }
}



// ========================== vector squared ==========================
void vev_square_fp(const float *const __restrict A, float *__restrict out,
                   const size_t size) {
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    out[i] = A[i] * A[i];
  }
}

void vec_square_double(const double *const __restrict A, double *__restrict out,
                   const size_t size) {
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    out[i] = A[i] * A[i];
  }
}

void vec_square_32(const uint32_t *const __restrict A, uint32_t *__restrict out,
                   const size_t size) {
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    out[i] = A[i] * A[i];
  }
}

void vec_square_32_64(const uint32_t *const __restrict A, uint64_t *__restrict out,
                   const size_t size) {
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    out[i] = (uint64_t)A[i] *(uint64_t) A[i];
  }
}

void vec_square_64(const uint64_t *const __restrict A, uint64_t *__restrict out,
                   const size_t size){
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    out[i] = A[i] * A[i];
  }
}

void vec_square_64_128(const uint64_t *const __restrict A, uint128_t *__restrict out,
                   const size_t size) {
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    out[i] = (uint128_t)A[i] * A[i];
  }
}


// ========================== matrix vector multiplication ==========================

void mat_vec_fp(const float *const __restrict A,
                const float *const __restrict B, float *const __restrict out,
                const size_t rows, const size_t cols) {
  float tmp = 0;
// #pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
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
  double tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC unroll 64
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}

void mat_vec_16(const uint16_t *const __restrict A,
                const uint16_t *const __restrict B, uint16_t *__restrict out,
                const size_t rows, const size_t cols) {
  uint16_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}

void mat_vec_16_32(const uint16_t *const __restrict A,
                   const uint16_t *const __restrict B, uint32_t *__restrict out,
                   const size_t rows, const size_t cols){
  uint32_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * (uint32_t)B[k];
    }
    out[i] = tmp;
  }
}



void mat_vec_32(const uint32_t *const __restrict A,
                const uint32_t *const __restrict B, uint32_t *__restrict out,
                const size_t rows, const size_t cols) {
  uint32_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
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
  uint64_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
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
  uint64_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
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
  uint128_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      tmp += (uint128_t)A[offset + k] * B[k];
    }
    out[i] = tmp;
  }
}


void mat_mat_fp(const float *__restrict A, const float *__restrict B,
                float *__restrict out, const size_t rows, const size_t cols) {
  float t0, t1;
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

void mat_mat_64(const uint64_t *__restrict A, const uint64_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
// #pragma GCC ivdep unroll 128
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

void mat_mat_32_64(const uint32_t *__restrict A, const uint64_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
    // #pragma GCC ivdep unroll 64
    for (size_t k = 0; k < cols; k++) {
      t0 += (uint64_t)A[offset + k] * B[b_cols * k];
      t1 += (uint64_t)A[offset + k] * B[b_cols * k + 1];
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
  constexpr size_t cols = (1<<8) + 123;
  // constexpr size_t rows_64 = 1<<18;
  constexpr size_t rows_64 = (1<<18) - 2043;
  // constexpr size_t rows_32 = 1<<19; // for 32-bit
  constexpr size_t rows_32 = (1<<18) - 2043;
  constexpr size_t rows_16 = 1<<20; // for 16-bit

  // Allocate matrices with the aligned allocator.
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64(rows_64 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> B_64(b_cols * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> out_64(b_cols * rows_64);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> out_32_64(b_cols * rows_32);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64_sqr_out(rows_64 * cols);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32(rows_32 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> B_32(b_cols * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> out_32(b_cols * rows_32);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32_sqr_out(rows_32 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_32_64_sqr_out(rows_32 * cols);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> A_16(rows_16 * cols);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> B_16(b_cols * cols);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> out_16(b_cols * rows_16);

  std::vector<float, AlignedAllocator<float, 64>> A_f(rows_32 * cols);
  std::vector<float, AlignedAllocator<float, 64>> B_f(b_cols * cols);
  std::vector<float, AlignedAllocator<float, 64>> out_f(b_cols * rows_32);
  std::vector<float, AlignedAllocator<float, 64>> A_f_sqr_out(rows_32 * cols);

  std::vector<double, AlignedAllocator<double, 64>> A_d(rows_64 * cols);
  std::vector<double, AlignedAllocator<double, 64>> B_d(b_cols * cols);
  std::vector<double, AlignedAllocator<double, 64>> out_d(b_cols * rows_64);
  std::vector<double, AlignedAllocator<double, 64>> A_d_sqr_out(rows_64 * cols);

  
  // std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> A_128(rows * cols);
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> out_128(b_cols * rows_64);
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> A_64_128_sqr_out(rows_64 * cols);

  double size_16_MB = (A_16.size() * sizeof(uint16_t)) / (1024.0 * 1024.0);
  double size_32_MB = (A_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  double size_64_MB = (A_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
  // double size_128_MB = (A_128.size() * cols * sizeof(uint128_t)) / (1024.0 * 1024.0);
  double size_f_MB = (A_f.size() * sizeof(float)) / (1024.0 * 1024.0);
  double size_d_MB = (A_d.size() * sizeof(double)) / (1024.0 * 1024.0);
  std::cout << "rows: " << rows_64 << ", cols: " << cols << std::endl;
  std::cout << "Matrix A 16b Size: " << size_16_MB << " MB" << std::endl;
  std::cout << "Matrix A 32b Size: " << size_32_MB << " MB" << std::endl;
  std::cout << "Matrix A 64b Size: " << size_64_MB << " MB" << std::endl;
  // std::cout << "Matrix A 128b Size: " << size_128_MB << " MB" << std::endl;
  std::cout << "Matrix A float Size: " << size_f_MB << " MB" << std::endl;
  std::cout << "Matrix A double Size: " << size_d_MB << " MB" << std::endl;


  // print the address of the first element of each vector
  // std::cout << "Address of A: " << static_cast<void*>(A_64.data()) << std::endl;
  // std::cout << "Address of B: " << static_cast<void*>(B_64.data()) << std::endl;
  // std::cout << "Address of out64: " << static_cast<void*>(out_64.data()) << std::endl;

  // Initialize matrices with random values
  for (size_t i = 0; i < A_64.size(); i++) { A_64[i] = rand(); }
  for (size_t i = 0; i < B_64.size(); i++) { B_64[i] = rand(); }
  for (size_t i = 0; i < A_32.size(); i++) { A_32[i] = rand(); }
  for (size_t i = 0; i < B_32.size(); i++) { B_32[i] = rand(); }
  for (size_t i = 0; i < A_16.size(); i++) { A_16[i] = rand(); }
  for (size_t i = 0; i < B_16.size(); i++) { B_16[i] = rand(); }
  for (size_t i = 0; i < A_f.size(); i++) { A_f[i] = static_cast<float>(rand()); }
  for (size_t i = 0; i < B_f.size(); i++) { B_f[i] = static_cast<float>(rand()); }
  for (size_t i = 0; i < A_d.size(); i++) { A_d[i] = static_cast<double>(rand()); }
  for (size_t i = 0; i < B_d.size(); i++) { B_d[i] = static_cast<double>(rand()); }
  // for (size_t i = 0; i < A_128.size(); i++) { A_128[i] = rand(); }

  // ================== Simple read 32-bit.
  std::cout << "A_32.size(): " << A_32.size() << std::endl; 
  size_t tot_sum = 0;
  TIME_START("simple_read_32");
  for (int i = 0; i < experiments; i++)
    // simple_read_32(A_32.data(), A_32.size(), rand());
    tot_sum += simple_read_32(A_32.data(), A_32.size());
  TIME_END("simple_read_32");

  // ================== Simple read.
  TIME_START("simple_read_64");
  for (int i = 0; i < experiments; i++)
    // simple_read(A_64.data(), A_64.size(), rand());
    tot_sum += simple_read_64(A_64.data(), A_64.size());
  TIME_END("simple_read_64");

  // // ================== Simple read 128-bit.
  // TIME_START("simple_read_128");
  // for (int i = 0; i < experiments; i++)
  //   simple_read_128(A_128.data(), A_64.size(), rand());
  // TIME_END("simple_read_128");

  // ================== float vector squared 
  TIME_START("vec_square_float");
  for (int i = 0; i < experiments; i++) {
    vev_square_fp(A_f.data(), A_f_sqr_out.data(), A_f.size());
    tot_sum += A_f_sqr_out[rand() % A_f_sqr_out.size()];
  }
  TIME_END("vec_square_float");

  // ================== double vector squared
  TIME_START("vec_square_double");
  for (int i = 0; i < experiments; i++) {
    vec_square_double(A_d.data(), A_d_sqr_out.data(), A_d.size());
    tot_sum += A_d_sqr_out[rand() % A_d_sqr_out.size()];
  }
  TIME_END("vec_square_double");

  // ================== 32-bit vector squared
  TIME_START("vec_square_32");
  for (int i = 0; i < experiments; i++) {
    vec_square_32(A_32.data(), A_32_sqr_out.data(), A_32.size());
    tot_sum += A_32_sqr_out[rand() % A_32_sqr_out.size()];
  }
  TIME_END("vec_square_32");

  // ================== 32-bit vector squared (64-bit)
  TIME_START("vec_square_32_64");
  for (int i = 0; i < experiments; i++) {
    vec_square_32_64(A_32.data(), A_32_64_sqr_out.data(), A_32.size());
    tot_sum += A_32_64_sqr_out[rand() % A_32_64_sqr_out.size()];
  }
  TIME_END("vec_square_32_64");

  // ================== 64-bit vector squared
  TIME_START("vec_square_64");
  for (int i = 0; i < experiments; i++) {
    vec_square_64(A_64.data(), A_64_sqr_out.data(), A_64.size());
    tot_sum += A_64_sqr_out[rand() % A_64_sqr_out.size()];
  }
  TIME_END("vec_square_64");

  // ================== 64-bit vector squared (128-bit)
  TIME_START("vec_square_64_128");
  for (int i = 0; i < experiments; i++) {
    vec_square_64_128(A_64.data(), A_64_128_sqr_out.data(), A_64.size());
    tot_sum += A_64_128_sqr_out[rand() % A_64_128_sqr_out.size()];
  }
  TIME_END("vec_square_64_128");


  // ================== float matrix vector multiplication.
  std::cout << "sizeof(float): " << sizeof(float) << std::endl;
  TIME_START("mat_vec_float");
  for (int i = 0; i < experiments; i++)
    mat_vec_fp(A_f.data(), B_f.data(), out_f.data(), rows_32, cols);
  TIME_END("mat_vec_float");

  // ================== double matrix vector multiplication.
  std::cout << "sizeof(double): " << sizeof(double) << std::endl;
  TIME_START("mat_vec_double");
  for (int i = 0; i < experiments; i++)
    mat_vec_double(A_d.data(), B_d.data(), out_d.data(), rows_64, cols);
  TIME_END("mat_vec_double");

  // ================== 16-bit matrix vector multiplication.
  std::cout << "sizeof(uint16_t): " << sizeof(uint16_t) << std::endl;
  TIME_START("mat_vec_16");
  for (int i = 0; i < experiments; i++)
    mat_vec_16(A_16.data(), B_16.data(), out_16.data(), rows_16, cols);
  TIME_END("mat_vec_16");

  // ================== 16-bit matrix vector multiplication (32-bit).
  TIME_START("mat_vec_16_32");
  for (int i = 0; i < experiments; i++)
    mat_vec_16_32(A_16.data(), B_16.data(), out_32.data(), rows_16, cols);
  TIME_END("mat_vec_16_32");
  
  // ================== 32-bit matrix vector multiplication.
  TIME_START("mat_vec_32");
  for (int i = 0; i < experiments; i++)
    mat_vec_32(A_32.data(), B_32.data(), out_32.data(), rows_32, cols);
  TIME_END("mat_vec_32");

  // ================== 32-bit matrix vector multiplication (64-bit).
  TIME_START("mat_vec_32_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  TIME_END("mat_vec_32_64");

  // ================== 64-bit matrix vector multiplication.
  TIME_START("mat_vec_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_64(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  TIME_END("mat_vec_64");

  // ================== 128-bit matrix vector multiplication.
  TIME_START("mat_vec_64_128");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  TIME_END("mat_vec_64_128");

  // ================== 64-bit matrix multiplication.
  TIME_START("mat_mat_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_64(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  TIME_END("mat_mat_64");

  // ================== Naive matrix multiplication (128-bit).
  TIME_START("mat_mat_64_128");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  TIME_END("mat_mat_64_128");

  // ================== Naive matrix multiplication (32-bit).
  TIME_START("mat_mat_32_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64(A_32.data(), B_64.data(), out_32_64.data(), rows_32, cols);
  TIME_END("mat_mat_32_64");

  // ================== Eigen matrix multiplication.
  TIME_START("mat_mat_eigen");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_Eigen(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  TIME_END("mat_mat_eigen");

  // ================== Eigen matrix vector multiplication.
  TIME_START("mat_vec_eigen");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_Eigen(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  TIME_END("mat_vec_eigen");

  // ================== Performance analysis.
  const size_t db_sz_64 = size_64_MB * experiments;  // 8 bytes per element
  const size_t db_sz_32 = size_32_MB * experiments;  // 4 bytes per element
  const size_t db_sz_16 = size_16_MB * experiments;  // 2 bytes per element
  // const size_t db_sz_128 = size_128_MB * experiments; // 16 bytes per element
  const size_t db_sz_f = size_f_MB * experiments;   // 4 bytes per element
  const size_t db_sz_d = size_d_MB * experiments;   // 8 bytes per element


  // std::cout << "mat_vec_float: " << GET_TIME("mat_vec_float") / experiments << " ms" << std::endl;
  std::cout << "tot_sum: " << tot_sum << std::endl;



  std::cout << "====== reading DB and add constant =======" << std::endl;
  PRINT_THROUGHPUT("simple_read_32", db_sz_32);
  PRINT_THROUGHPUT("simple_read_64", db_sz_64);
  PRINT_THROUGHPUT("simple_read_float", db_sz_f);
  // PRINT_THROUGHPUT("simple_read_128", db_sz_128);
  std::cout << "====== vector squared =======" << std::endl;
  PRINT_THROUGHPUT("vec_square_float", db_sz_f);
  PRINT_THROUGHPUT("vec_square_double", db_sz_d);
  PRINT_THROUGHPUT("vec_square_32", db_sz_32);
  PRINT_THROUGHPUT("vec_square_32_64", db_sz_32);
  PRINT_THROUGHPUT("vec_square_64", db_sz_64);
  PRINT_THROUGHPUT("vec_square_64_128", db_sz_64);
  std::cout << "====== matrix * vector =======" << std::endl;
  PRINT_THROUGHPUT("mat_vec_float", db_sz_f);
  PRINT_THROUGHPUT("mat_vec_double", db_sz_d);
  PRINT_THROUGHPUT("mat_vec_16", db_sz_16);
  PRINT_THROUGHPUT("mat_vec_16_32", db_sz_16);
  PRINT_THROUGHPUT("mat_vec_32", db_sz_32);
  PRINT_THROUGHPUT("mat_vec_32_64", db_sz_32);
  PRINT_THROUGHPUT("mat_vec_64", db_sz_64);
  PRINT_THROUGHPUT("mat_vec_64_128", db_sz_64);
  std::cout << "====== matrix * 2 columns =======" << std::endl;
  PRINT_THROUGHPUT("mat_mat_64", db_sz_64);
  PRINT_THROUGHPUT("mat_mat_64_128", db_sz_64);
  PRINT_THROUGHPUT("mat_mat_32_64", db_sz_32);
  PRINT_THROUGHPUT("mat_mat_eigen", db_sz_64);
  PRINT_THROUGHPUT("mat_vec_eigen", db_sz_64);
  
  return 0;
}
