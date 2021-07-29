cmake_minimum_required(VERSION 3.16)
project(parallel_computing)

set(CMAKE_CXX_STANDARD 17)
# hack for cuda before cmake 3.18.3
if(${CMAKE_VERSION} VERSION_LESS "3.18.3")
  target_compile_options(cuda_test_prog PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
endif()