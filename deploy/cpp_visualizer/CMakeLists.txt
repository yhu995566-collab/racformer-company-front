cmake_minimum_required(VERSION 3.16)
project(racformer_visualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

add_library(racformer_visualizer SHARED
    src/c_api.cpp
    src/visualizer.cpp)
target_include_directories(racformer_visualizer
    PUBLIC include
    PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(racformer_visualizer PRIVATE
    ${OpenCV_LIBS} Threads::Threads)
target_compile_options(racformer_visualizer PRIVATE
    -Wall -Wextra -Wpedantic)

install(TARGETS racformer_visualizer LIBRARY DESTINATION lib)
install(FILES include/racformer/visualizer_c_api.h
    DESTINATION include/racformer)
