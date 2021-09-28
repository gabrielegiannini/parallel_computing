cmake_minimum_required(VERSION 3.16)
project(parallel_computing LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "-arch=sm_60 -rdc=true" )

add_executable(kmeansSeq kmeans.cpp)
add_executable(kmeans-cuda kmeans.cu)
#target_compile_options(kmeans-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
#target_link_libraries(kmeans-cuda ${CUDA_cudadevrt_LIBRARY})
set_target_properties(kmeans-cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# hack for cuda before cmake 3.18.3
if(${CMAKE_VERSION} VERSION_LESS "3.18.3")
  target_compile_options(kmeans-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
endif()

