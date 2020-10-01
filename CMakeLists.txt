cmake_minimum_required(VERSION 3.10)
project(tensor-library)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-g3"
)

add_executable(tensor tensor.cc)
target_link_libraries(tensor stdc++)

add_executable(einstein einstein.cc)
target_link_libraries(einstein stdc++)
