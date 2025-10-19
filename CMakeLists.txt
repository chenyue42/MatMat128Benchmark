cmake_minimum_required(VERSION 3.10)
project(MatMat128Benchmark)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "-O2 -g -MD -march=native -mtune=native")  # -O2
set(CMAKE_CXX_FLAGS "-O3 -g -MD -march=native -mtune=native")  # Enable all optimizations

# Create executables for each benchmark type
add_executable(simple_read src/simple_read.cpp)
add_executable(vec_dot     src/vec_dot.cpp)
add_executable(mat_vec     src/mat_vec.cpp)
add_executable(mat_mat     src/mat_mat.cpp)