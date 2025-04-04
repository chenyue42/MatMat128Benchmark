cmake_minimum_required(VERSION 3.10)
project(MatMat128Benchmark)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3 -MD -march=native -mtune=native")  # Enable all optimizations

find_package(Eigen3 REQUIRED)

# Create executable from the source files
add_executable(benchmark src/main.cpp)

target_link_libraries(benchmark Eigen3::Eigen)

