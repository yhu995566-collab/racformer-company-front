cmake_minimum_required(VERSION 3.18)
project(racformer_bev_pool_v2_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT root containing include/ and lib/")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}/include $ENV{TENSORRT_ROOT}/include
        $ENV{CONDA_PREFIX}/include /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_NVINFER_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
        $ENV{TENSORRT_ROOT}/lib $ENV{TENSORRT_ROOT}/lib64
        $ENV{CONDA_PREFIX}/lib /usr/lib/x86_64-linux-gnu /usr/lib)
find_package(CUDAToolkit REQUIRED)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT headers or libnvinfer were not found; set TENSORRT_ROOT")
endif()

add_library(racformer_bev_pool_v2_trt SHARED
  bev_pool_v2_plugin.cpp
  bev_pool_v2_kernel.cu
  msmv_sampling_plugin.cpp
  msmv_sampling_kernel.cu)
target_include_directories(racformer_bev_pool_v2_trt PRIVATE
  ${TENSORRT_INCLUDE_DIR})
target_link_libraries(racformer_bev_pool_v2_trt PRIVATE
  ${TENSORRT_NVINFER_LIBRARY}
  CUDA::cudart)
set_target_properties(racformer_bev_pool_v2_trt PROPERTIES
  OUTPUT_NAME racformer_bev_pool_v2_trt)
