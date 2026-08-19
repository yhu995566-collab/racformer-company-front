cmake_minimum_required(VERSION 3.16)
project(racformer_runtime LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JPEG REQUIRED)
find_package(Threads REQUIRED)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS /usr/local/cuda/include /usr/local/cuda-11.4/include
          /usr/local/cuda/targets/aarch64-linux/include)
find_library(CUDART_LIBRARY cudart
    HINTS /usr/local/cuda/lib64 /usr/local/cuda-11.4/lib64
          /usr/local/cuda/targets/aarch64-linux/lib)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA headers/library were not found")
endif()
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/libraries were not found")
endif()

add_library(racformer_runtime SHARED
    src/c_api.cpp
    src/preprocess.cpp
    src/runtime.cpp
    src/tensor_store.cpp
    src/trt_engine.cpp)
target_include_directories(racformer_runtime
    PUBLIC include
    PRIVATE src "${CUDA_INCLUDE_DIR}" "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(racformer_runtime PRIVATE
    "${CUDART_LIBRARY}" JPEG::JPEG Threads::Threads dl
    "${TENSORRT_LIBRARY}" "${TENSORRT_PLUGIN_LIBRARY}")
target_compile_options(racformer_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_executable(racformer_offline_demo apps/offline_demo.cpp)
target_link_libraries(racformer_offline_demo PRIVATE racformer_runtime Threads::Threads)

if(BUILD_TESTING)
    add_executable(temporal_ring_test tests/temporal_ring_test.cpp)
    target_include_directories(temporal_ring_test PRIVATE src)
    target_compile_options(temporal_ring_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME temporal_ring_test COMMAND temporal_ring_test)
endif()

install(TARGETS racformer_runtime racformer_offline_demo
    LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/racformer/c_api.h DESTINATION include/racformer)
