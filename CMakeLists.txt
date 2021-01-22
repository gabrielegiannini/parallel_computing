cmake_minimum_required(VERSION 3.16)
project(parallel_computing)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)
add_executable(ngrams main.cpp)
target_link_libraries(ngrams PRIVATE Threads::Threads)
file( GLOB INPUT_TXT "analyze/*" )
file( COPY ${INPUT_TXT} DESTINATION analyze)