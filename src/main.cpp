#include "TimeLogger.h"
#include "aligned_allocator.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <Eigen/Dense>

// Define uint128_t for GCC.
using uint128_t = unsigned __int128;
constexpr size_t b_cols = 2;

// ./mlc --memory_bandwidth_scan -t1
// 1MB = 1,000,000 bytes
// result is about 13000 MB = 13000 * 1,000,000 / (2^30) ~= 12.1 GB
static inline uint32_t simple_read_32(const uint32_t *const __restrict a, const size_t size) {
  uint32_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}


static inline uint64_t simple_read_64(const uint64_t *const __restrict a, const size_t size) {
  uint64_t sum = 0;
  #pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}


// ========================== vector dot product ==========================
static inline uint32_t vec_dot_32(const uint32_t *const __restrict a, const uint32_t *const __restrict b,
                   const size_t size) {
  uint32_t sum = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static inline uint64_t vec_dot_32_64(const uint32_t *const __restrict a, const uint32_t *const __restrict b,
                   const size_t size) {
  uint64_t sum = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += (uint64_t)a[i] *(uint64_t) b[i];
  }
  return sum;
}

static inline uint64_t vec_dot_64(const uint64_t *const __restrict a, const uint64_t *const __restrict b,
                   const size_t size){
  uint64_t sum = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static inline uint128_t vec_dot_64_128(const uint64_t *const __restrict a, const uint64_t *const __restrict b,
                   const size_t size) {
  uint128_t sum = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < size; i++) {
    sum += (uint128_t)a[i] * b[i];
  }
  return sum;
}

// AVX2 kernel: dot product of two u32 vectors, accumulate into uint64_t
static inline uint64_t dot_u32_u64_avx2(const uint32_t* __restrict a,
  const uint32_t* __restrict b,
  size_t n)
{
#if defined(__AVX2__)
__m256i acc_even = _mm256_setzero_si256(); // sums a0*b0, a2*b2, ...
__m256i acc_odd  = _mm256_setzero_si256(); // sums a1*b1, a3*b3, ...
size_t i = 0;

// process 8 elements per iter -> 4x 64-bit products per mul instruction
for (; i + 8 <= n; i += 8) {
__m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
__m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

// even lanes: [a0,a2,a4,a6] * [b0,b2,b4,b6] -> 4x uint64_t
__m256i prod_even = _mm256_mul_epu32(va, vb);

// odd lanes: shift each 64-bit lane down by 32 to get [a1,a3,a5,a7], ditto for b
__m256i va_hi = _mm256_srli_epi64(va, 32);
__m256i vb_hi = _mm256_srli_epi64(vb, 32);
__m256i prod_odd = _mm256_mul_epu32(va_hi, vb_hi);

acc_even = _mm256_add_epi64(acc_even, prod_even);
acc_odd  = _mm256_add_epi64(acc_odd,  prod_odd);
}

// horizontal add the two accumulators
alignas(32) uint64_t buf[4];
uint64_t sum = 0;
_mm256_store_si256((__m256i*)buf, acc_even);
sum += buf[0] + buf[1] + buf[2] + buf[3];
_mm256_store_si256((__m256i*)buf, acc_odd);
sum += buf[0] + buf[1] + buf[2] + buf[3];

// tail
for (; i < n; ++i) sum += (uint64_t)a[i] * (uint64_t)b[i];
return sum;
#else
// compiled without AVX2: fall back to scalar
uint64_t acc0=0, acc1=0, acc2=0, acc3=0;
size_t i=0;
for (; i+4<=n; i+=4) {
acc0 += (uint64_t)a[i+0]*b[i+0];
acc1 += (uint64_t)a[i+1]*b[i+1];
acc2 += (uint64_t)a[i+2]*b[i+2];
acc3 += (uint64_t)a[i+3]*b[i+3];
}
for (; i<n; ++i) acc0 += (uint64_t)a[i]*b[i];
return (acc0+acc1)+(acc2+acc3);
#endif
}


// ========================== matrix vector multiplication ==========================
void mat_vec_32(const uint32_t *const __restrict A,
                const uint32_t *const __restrict b, uint32_t *__restrict out,
                const size_t rows, const size_t cols) {
  uint32_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}


void mat_vec_32_64(const uint32_t *const __restrict A,
                   const uint32_t *const __restrict b, uint64_t *__restrict out,
                   const size_t rows, const size_t cols) {
  uint64_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * (uint64_t)b[k];
    }
    out[i] = tmp;
  }
}

static inline void mat_vec_32_64_Eigen(const uint32_t *A, const uint32_t *b, uint64_t *out, size_t rows, size_t cols) {
    Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
    Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>> matB(b, cols);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>> matOut(out, rows);
    
    matOut.noalias() = matA.cast<uint64_t>() * matB.cast<uint64_t>();
}


// Optimized mat-vec: A is row-major rows×cols (u32), B has length cols (u32), out has length rows (uint64_t)
void mat_vec_32_64_avx2(const uint32_t* __restrict A,
  const uint32_t* __restrict b,
  uint64_t* __restrict out,
  size_t rows, size_t cols) {
#pragma GCC ivdep
  for (size_t i = 0; i < rows; ++i) {
    const uint32_t* __restrict arow = A + i * cols;
    out[i] = dot_u32_u64_avx2(arow, b, cols);
  }
}

void mat_vec_64(const uint64_t *const __restrict A,
                const uint64_t *const __restrict b, uint64_t *__restrict out,
                const size_t rows, const size_t cols) {
  uint64_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC ivdep
    for (size_t k = 0; k < cols; k++) {
      tmp += A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}


void mat_vec_64_128(const uint64_t *const __restrict A,
                 const uint64_t *const __restrict b, uint128_t *__restrict out,
                 const size_t rows, const size_t cols) {
  uint128_t tmp = 0;
#pragma GCC ivdep
  for (size_t i = 0; i < rows; i++) {
    const size_t offset = i * cols;
    tmp = 0;
#pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      tmp += (uint128_t)A[offset + k] * b[k];
    }
    out[i] = tmp;
  }
}


// ========================== matrix matrix multiplication ==========================
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


static inline void mat_mat_64_128(const uint64_t *__restrict A, const uint64_t *__restrict B,
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

static inline void mat_mat_32_64(const uint32_t *__restrict A, const uint32_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  uint64_t t0, t1;
  for (size_t i = 0; i < rows; i++) {
    t0 = 0; t1 = 0;
    const size_t offset = i * cols;
    #pragma GCC unroll 32
    for (size_t k = 0; k < cols; k++) {
      t0 += (uint64_t)A[offset + k] * B[b_cols * k];
      t1 += (uint64_t)A[offset + k] * B[b_cols * k + 1];
    }
    out[b_cols * i] = t0;
    out[b_cols * i + 1] = t1;
  }
}

static inline void mat_mat_32_64_Eigen(const uint32_t *__restrict A, const uint32_t *__restrict B,
                uint64_t *__restrict out, const size_t rows, const size_t cols) {
  Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, rows, cols);
  Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, cols, b_cols);
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matOut(out, rows, b_cols);
  // Two casts allocate temporaries each call; very expensive in tight loop.
  matOut.noalias() = matA.cast<uint64_t>() * matB.cast<uint64_t>();
}

void mat_mat_32_64_avx512(const uint32_t *A, const uint32_t *B, uint64_t *out,
                          size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    const uint32_t *arow = A + i * cols;
    __m512i acc0 = _mm512_setzero_si512(); // accumulates A*B[:,0]
    __m512i acc1 = _mm512_setzero_si512(); // accumulates A*B[:,1]

    size_t k = 0;
    const size_t step = 8; // 8 x uint64 lanes => 8 pairs (16 uint32)
    for (; k + step <= cols; k += step) {
      // Load 8 A values (uint32) and replicate each into both half-words of a
      // 64-bit lane
      __m256i a32 = _mm256_loadu_si256((const __m256i *)(arow + k));
      __m512i a64 = _mm512_cvtepu32_epi64(a32);
      __m512i a64_hi = _mm512_slli_epi64(a64, 32);
      __m512i a_rep = _mm512_or_si512(a64, a64_hi); // [a|a] per 64-bit lane

      // Load 8 pairs [B0, B1, B0, B1, ...] as 64-byte chunk
      const uint32_t *bptr = B + (k * b_cols);
      __m512i b_pairs = _mm512_loadu_si512(
          (const void *)bptr); // each 64-bit lane: low=B0, high=B1

      // Multiply low 32 bits: A * B0
      acc0 = _mm512_add_epi64(acc0, _mm512_mul_epu32(a_rep, b_pairs));
      // Multiply high 32 bits: A * B1 (shift B's high 32 down to low)
      __m512i b_hi = _mm512_srli_epi64(b_pairs, 32);
      acc1 = _mm512_add_epi64(acc1, _mm512_mul_epu32(a_rep, b_hi));
    }

    // horizontal sum of 8x u64 in each accumulator
    alignas(64) uint64_t tmp0[8];
    alignas(64) uint64_t tmp1[8];
    _mm512_store_si512((__m512i *)tmp0, acc0);
    _mm512_store_si512((__m512i *)tmp1, acc1);
    uint64_t t0 = 0, t1 = 0;
    for (int x = 0; x < 8; ++x) {
      t0 += tmp0[x];
      t1 += tmp1[x];
    }

    // tail
    for (; k < cols; ++k) {
      uint64_t a = arow[k];
      t0 += a * (uint64_t)B[k * b_cols + 0];
      t1 += a * (uint64_t)B[k * b_cols + 1];
    }

    out[b_cols * i + 0] = t0;
    out[b_cols * i + 1] = t1;
  }
}

int main() {
  constexpr size_t experiments = 20;
  constexpr size_t cols = 1<<8; 
  constexpr size_t rows_64 = 1<<17;
  constexpr size_t rows_32 = rows_64 * 2;
  constexpr size_t b_cols = 2;


  // Allocate vectors for dot product tests
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> a_64(rows_64 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> b_64(rows_64 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> a_32(rows_32 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> b_32(rows_32 * cols);

  // Allocate matrics with the aligned allocator. 
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> A_64(rows_64 * cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> B_64(cols * b_cols);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> out_64(rows_32 * b_cols); // allocate more for mat_mat_32_64

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> A_32(rows_32 * cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> B_32(cols * b_cols);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> out_32(rows_32 * b_cols);
  
  std::vector<uint128_t, AlignedAllocator<uint128_t, 64>> out_128(rows_64 * b_cols );


  // Initialize matrices with random values
  for (size_t i = 0; i < a_64.size(); i++) { a_64[i] = rand(); }
  for (size_t i = 0; i < a_32.size(); i++) { a_32[i] = rand(); }
  for (size_t i = 0; i < A_64.size(); i++) { A_64[i] = rand(); }
  for (size_t i = 0; i < B_64.size(); i++) { B_64[i] = rand(); }
  for (size_t i = 0; i < A_32.size(); i++) { A_32[i] = rand(); }
  for (size_t i = 0; i < B_32.size(); i++) { B_32[i] = rand(); }

  // ================== Simple read 32-bit.
  std::cout << "A_32.size(): " << A_32.size() << std::endl; 
  size_t tot_sum = 0;
  TIME_START("simple_read_32");
  for (int i = 0; i < experiments; i++)
    tot_sum += simple_read_32(A_32.data(), A_32.size());
  TIME_END("simple_read_32");

  // ================== Simple read 64-bit.
  TIME_START("simple_read_64");
  for (int i = 0; i < experiments; i++)
    tot_sum += simple_read_64(A_64.data(), A_64.size());
  TIME_END("simple_read_64");

  // -------------------------- vector dot product --------------------------
  // ================== 32-bit vector dot product
  TIME_START("vec_dot_32");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_32(a_32.data(), b_32.data(), a_32.size());
  }
  TIME_END("vec_dot_32");

  // ================== 32-bit vector dot product (64-bit)
  TIME_START("vec_dot_32_64");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_32_64(a_32.data(), b_32.data(), a_32.size());
  }
  TIME_END("vec_dot_32_64");

  // ================== 64-bit vector dot product
  TIME_START("vec_dot_64");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_64(a_64.data(), b_64.data(), a_64.size());
  }
  TIME_END("vec_dot_64");

  // ================== 64-bit vector dot product (128-bit)
  TIME_START("vec_dot_64_128");
  for (int i = 0; i < experiments; i++) {
    tot_sum += vec_dot_64_128(a_64.data(), b_64.data(), a_64.size());
  }
  TIME_END("vec_dot_64_128");

  std::cout << "====== matrix vector multiplication =======" << std::endl;
  // ================== 32-bit matrix vector multiplication.
  TIME_START("mat_vec_32");
  for (int i = 0; i < experiments; i++)
    mat_vec_32(A_32.data(), B_32.data(), out_32.data(), rows_32, cols);
  tot_sum += out_32[rand() % out_32.size()];
  TIME_END("mat_vec_32");

  // ================== 32-bit matrix vector multiplication (64-bit).
  TIME_START("mat_vec_32_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_vec_32_64");

  // ================== 32-bit matrix vector multiplication (AVX2).
  TIME_START("mat_vec_32_64_avx2");
  for (int i = 0; i < experiments; i++)
    mat_vec_32_64_avx2(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_vec_32_64_avx2");

  TIME_START("mat_vec_32_64_Eigen");
  for (int i = 0; i < experiments; i++)
    mat_vec_32_64_Eigen(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_vec_32_64_Eigen");

  // ================== 64-bit matrix vector multiplication.
  TIME_START("mat_vec_64");
  for (int i = 0; i < experiments; i++)
    mat_vec_64(A_64.data(), b_64.data(), out_64.data(), rows_64, cols);
  tot_sum += out_64[rand() % rows_64];
  TIME_END("mat_vec_64");

  // ================== 128-bit matrix vector multiplication.
  TIME_START("mat_vec_64_128");
  for (int i = 0; i < experiments; i++)
    mat_vec_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  tot_sum += out_128[rand() % rows_64];
  TIME_END("mat_vec_64_128");


  // -------------------------- matrix matrix multiplication --------------------------
  // ================== 64-bit matrix multiplication.
  TIME_START("mat_mat_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_64(A_64.data(), B_64.data(), out_64.data(), rows_64, cols);
  tot_sum += out_64[rand() % rows_64];
  TIME_END("mat_mat_64");

  // ================== Naive matrix multiplication (128-bit).
  TIME_START("mat_mat_64_128");
  for (int i = 0; i < experiments; i++)
    mat_mat_64_128(A_64.data(), B_64.data(), out_128.data(), rows_64, cols);
  tot_sum += out_128[rand() % out_128.size()];
  TIME_END("mat_mat_64_128");

  // ================== Naive matrix multiplication (32-bit).
  TIME_START("mat_mat_32_64");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64");

  TIME_START("mat_mat_32_64_avx512");
  for (int i = 0; i < experiments; i++)
    mat_mat_32_64_avx512(A_32.data(), B_32.data(), out_64.data(), rows_32, cols);
  tot_sum += out_64[rand() % out_64.size()];
  TIME_END("mat_mat_32_64_avx512");

  std::cout << "tot_sum: " << tot_sum << std::endl;
  // ================== Performance analysis.
  const double mat_32_MB = (A_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  const double mat_64_MB = (A_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
  const double vec_32_MB = (a_32.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
  const double vec_64_MB = (a_64.size() * sizeof(uint64_t)) / (1024.0 * 1024.0);
  std::cout << "rows: " << rows_64 << ", cols: " << cols << std::endl;
  std::cout << "Vector A 64b Size: " << a_64.size() * sizeof(uint64_t) / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "Vector A 32b Size: " << a_32.size() * sizeof(uint32_t) / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "Matrix A 32b Size: " << mat_32_MB << " MB" << std::endl;
  std::cout << "Matrix A 64b Size: " << mat_64_MB << " MB" << std::endl;

  // Fair bandwidth metrics including A, B, and out for mat-vec and mat-mat
  const double MB = (1024.0 * 1024.0);
  const double total_size = MB / experiments;

  std::cout << "====== reading DB and add constant =======" << std::endl;
  PRINT_THROUGHPUT("simple_read_32", vec_32_MB * experiments);
  PRINT_THROUGHPUT("simple_read_64", vec_64_MB * experiments);
  std::cout << "====== vector dot product =======" << std::endl;
  PRINT_THROUGHPUT("vec_dot_32", vec_32_MB * experiments);
  PRINT_THROUGHPUT("vec_dot_32_64", vec_32_MB * experiments);
  PRINT_THROUGHPUT("vec_dot_64", vec_64_MB * experiments);
  PRINT_THROUGHPUT("vec_dot_64_128", vec_64_MB * experiments);
  std::cout << "====== matrix * vector =======" << std::endl;
  PRINT_THROUGHPUT("mat_vec_32", mat_32_MB * experiments);
  PRINT_THROUGHPUT("mat_vec_32_64", mat_32_MB * experiments);
  PRINT_THROUGHPUT("mat_vec_32_64_Eigen", mat_32_MB * experiments);
  PRINT_THROUGHPUT("mat_vec_32_64_avx2", mat_32_MB * experiments);
  PRINT_THROUGHPUT("mat_vec_64", mat_64_MB * experiments);
  PRINT_THROUGHPUT("mat_vec_64_128", mat_64_MB * experiments);
  std::cout << "====== matrix * matrix =======" << std::endl;
  PRINT_THROUGHPUT("mat_mat_64_128", mat_64_MB * experiments);
  PRINT_THROUGHPUT("mat_mat_32_64", mat_32_MB * experiments);  
  PRINT_THROUGHPUT("mat_mat_32_64_avx512", mat_32_MB * experiments);
  return 0;
}
