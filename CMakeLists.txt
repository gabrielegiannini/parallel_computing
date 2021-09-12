cmake_minimum_required(VERSION 3.16)
project(parallel_computing)

include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package(Threads REQUIRED)
add_executable(ngrams main.cpp)
add_executable(ngrams-omp mainOMP.cpp)
add_executable(ngrams-seq mainSeq.cpp)
target_link_libraries(ngrams PRIVATE Threads::Threads)
file( GLOB INPUT_TXT "analyze/*" )
file( COPY ${INPUT_TXT} DESTINATION analyze)