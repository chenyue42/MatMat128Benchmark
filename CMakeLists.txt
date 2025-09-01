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

# Create executable from the source files
add_executable(benchmark src/main.cpp)

if (TARGET Eigen3::Eigen)
  target_link_libraries(benchmark Eigen3::Eigen)
else()
  # Fallback: include directory if target isn't defined (older Eigen)
  if (eigen_SOURCE_DIR)
    target_include_directories(benchmark PRIVATE ${eigen_SOURCE_DIR})
  endif()
endif()

