cmake_minimum_required(VERSION 3.10)
project(tensor-library)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-g3"
)

add_executable(tensor-library main.cc)
target_link_libraries(tensor-library stdc++)
