cmake_minimum_required(VERSION 3.16)
project(parallel_computing LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_SEPARABLE_COMPILATION ON)

add_executable(kmeans-cuda kmeans.cu)
# hack for cuda before cmake 3.18.3
if(${CMAKE_VERSION} VERSION_LESS "3.18.3")
  target_compile_options(kmeans-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17 -arch=sm_35>)
endif()

