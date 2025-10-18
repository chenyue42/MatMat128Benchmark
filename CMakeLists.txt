cmake_minimum_required(VERSION 3.10)
project(MatMat128Benchmark)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "-O3 -MD -march=native -mtune=native")  # Enable all optimizations
set(CMAKE_CXX_FLAGS "-O3 -g -MD -march=native -mtune=native")  # Enable all optimizations

find_package(Eigen3 QUIET)

# If Eigen isn't available on the system, fetch it header-only (no root needed)
if (NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)
endif()

# Create executables for each benchmark type
add_executable(simple_read src/main_simple_read.cpp)
add_executable(vec_dot     src/main_vec_dot.cpp)
add_executable(mat_vec     src/main_mat_vec.cpp)
add_executable(mat_mat2    src/main_mat_mat2.cpp)
add_executable(full_bench  src/main.cpp)

# Link Eigen or include it for each target
set(_BENCH_TARGETS simple_read vec_dot mat_vec mat_mat2 full_bench)
foreach(t ${_BENCH_TARGETS})
  if (TARGET Eigen3::Eigen)
    target_link_libraries(${t} Eigen3::Eigen)
  else()
    if (eigen_SOURCE_DIR)
      target_include_directories(${t} PRIVATE ${eigen_SOURCE_DIR})
    endif()
  endif()
endforeach()

