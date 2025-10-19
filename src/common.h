#ifndef COMMON_H
#define COMMON_H
#include <cstdint>
#include <cstddef>
#include <random>
#include <immintrin.h>

constexpr size_t b_cols = 2;

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

constexpr size_t experiments = 10;
constexpr size_t cols = 473;
constexpr size_t rows_64 = 1<<18;
constexpr size_t rows_32 = rows_64 * 2;

#endif // COMMON_H