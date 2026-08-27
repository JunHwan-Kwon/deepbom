cmake_minimum_required(VERSION 3.24)
project(deepbom_tensorrt_parser_collector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED TENSORRT_ROOT)
  if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
  else()
    message(FATAL_ERROR "Set TENSORRT_ROOT to the selected TensorRT SDK. The collector never downloads or silently selects a runtime build.")
  endif()
endif()

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}/include" REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" REQUIRED)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" REQUIRED)

add_executable(deepbom-tensorrt-parser-collector src/main.cc)
target_include_directories(deepbom-tensorrt-parser-collector SYSTEM PRIVATE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(deepbom-tensorrt-parser-collector PRIVATE
  "${TENSORRT_LIBRARY}"
  "${TENSORRT_ONNX_PARSER_LIBRARY}"
  CUDA::cudart
  ${CMAKE_DL_LIBS}
)

if(MSVC)
  target_link_libraries(deepbom-tensorrt-parser-collector PRIVATE bcrypt)
  target_compile_options(deepbom-tensorrt-parser-collector PRIVATE /W4 /WX /permissive-)
else()
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  target_link_libraries(deepbom-tensorrt-parser-collector PRIVATE OpenSSL::Crypto)
  target_compile_options(deepbom-tensorrt-parser-collector PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
