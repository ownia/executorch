# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(supertonic_runner)

include(GNUInstallDirs)

option(EXECUTORCH_BUILD_VULKAN "Build the Supertonic runner with Vulkan." OFF)

if(NOT EXECUTORCH_BUILD_VULKAN)
  if(NOT APPLE)
    message(
      FATAL_ERROR "The Supertonic MLX runner is supported only on Darwin. "
                  "Use the supertonic-vulkan preset on Linux."
    )
  endif()
  set(_supertonic_target_processor "${CMAKE_OSX_ARCHITECTURES}")
  if(NOT _supertonic_target_processor)
    set(_supertonic_target_processor "${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(NOT _supertonic_target_processor STREQUAL "arm64")
    message(
      FATAL_ERROR "The Supertonic MLX runner requires an arm64 Darwin target; "
                  "detected '${_supertonic_target_processor}'."
    )
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_SOURCE_ROOT}/tools/cmake/Utils.cmake)

string(CONCAT _json_include
              "${EXECUTORCH_SOURCE_ROOT}/extension/llm/tokenizers/"
              "third-party/json/single_include"
)
if(APPLE)
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
  set(_unicode_libraries ${COREFOUNDATION_FRAMEWORK})
else()
  find_package(ICU REQUIRED COMPONENTS uc)
  set(_unicode_libraries ICU::uc)
endif()

set(_helper_sources runtime/supertonic_runner.cpp runtime/text_processor.cpp
                    runtime/style_loader.cpp runtime/wav_writer.cpp
)

enable_testing()
add_executable(
  supertonic_runtime_test runtime/tests/supertonic_runtime_test.cpp
                          ${_helper_sources}
)
target_compile_definitions(
  supertonic_runtime_test PRIVATE SUPERTONIC_PURE_HELPERS_ONLY
)
target_include_directories(
  supertonic_runtime_test PRIVATE runtime ${_json_include}
)
target_link_libraries(supertonic_runtime_test PRIVATE ${_unicode_libraries})
add_test(NAME supertonic_runtime_helpers COMMAND supertonic_runtime_test)

find_package(
  gflags REQUIRED CONFIG HINTS
  ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags
)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

if(EXECUTORCH_BUILD_VULKAN AND NOT TARGET vulkan_backend)
  message(
    FATAL_ERROR "ExecuTorch must be installed with EXECUTORCH_BUILD_VULKAN=ON."
  )
elseif(NOT EXECUTORCH_BUILD_VULKAN AND (NOT TARGET mlxdelegate OR NOT TARGET mlx
                                       )
)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with the MLX delegate and runtime targets."
  )
endif()

foreach(required_target extension_module extension_data_loader extension_tensor
                        extension_flat_tensor
)
  if(NOT TARGET ${required_target})
    message(
      FATAL_ERROR
        "ExecuTorch installation is missing required target ${required_target}."
    )
  endif()
endforeach()

add_executable(supertonic_runner runtime/main.cpp ${_helper_sources})
target_include_directories(
  supertonic_runner PRIVATE ${EXECUTORCH_SOURCE_ROOT}/.. runtime
                            ${_json_include}
)
target_link_libraries(
  supertonic_runner
  PRIVATE executorch
          extension_module
          extension_data_loader
          extension_tensor
          extension_flat_tensor
          gflags
          ${_unicode_libraries}
)
if(EXECUTORCH_BUILD_VULKAN)
  target_compile_definitions(supertonic_runner PRIVATE EXECUTORCH_BUILD_VULKAN)
  target_link_libraries(supertonic_runner PRIVATE vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
  add_executable(
    supertonic_vulkan_test runtime/tests/supertonic_vulkan_test.cpp
  )
  target_include_directories(
    supertonic_vulkan_test PRIVATE ${EXECUTORCH_SOURCE_ROOT}/..
  )
  target_link_libraries(
    supertonic_vulkan_test PRIVATE vulkan_backend executorch_core
  )
  add_test(NAME supertonic_vulkan_copies COMMAND supertonic_vulkan_test)
  set_tests_properties(supertonic_vulkan_copies PROPERTIES TIMEOUT 120)
  if(TARGET optimized_native_cpu_ops_lib)
    target_link_libraries(
      supertonic_runner PRIVATE optimized_native_cpu_ops_lib optimized_kernels
                                portable_kernels cpublas eigen_blas
    )
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  elseif(TARGET portable_ops_lib)
    target_link_libraries(
      supertonic_runner PRIVATE portable_ops_lib portable_kernels
    )
    executorch_target_link_options_shared_lib(portable_ops_lib)
  else()
    message(
      FATAL_ERROR "ExecuTorch must include portable or optimized CPU kernels."
    )
  endif()
  if(TARGET quantized_ops_lib)
    target_link_libraries(
      supertonic_runner PRIVATE quantized_ops_lib quantized_kernels
    )
    get_target_property(_quantized_imported quantized_ops_lib IMPORTED)
    if(NOT _quantized_imported)
      executorch_target_link_options_shared_lib(quantized_ops_lib)
    endif()
  endif()
else()
  target_compile_definitions(supertonic_runner PRIVATE EXECUTORCH_BUILD_MLX)
  target_link_libraries(supertonic_runner PRIVATE mlxdelegate mlx)
  executorch_target_link_options_shared_lib(mlxdelegate)
  set(EXECUTORCH_BUILD_MLX ON)
  if(NOT DEFINED MLX_METALLIB_PATH OR NOT EXISTS "${MLX_METALLIB_PATH}")
    message(
      FATAL_ERROR "The installed MLX package did not provide mlx.metallib."
    )
  endif()
  executorch_target_copy_mlx_metallib(supertonic_runner)
  install(
    FILES "${MLX_METALLIB_PATH}"
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    RENAME mlx.metallib
  )
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(supertonic_runner)
endif()
install(TARGETS supertonic_runner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(SUPERTONIC_INTEGRATION_PTE
    ""
    CACHE FILEPATH "Optional Supertonic PTE for the integration test."
)
set(SUPERTONIC_INTEGRATION_ASSET_DIR
    ""
    CACHE PATH "Optional published Supertonic asset directory for integration."
)
set(SUPERTONIC_INTEGRATION_STYLE
    ""
    CACHE FILEPATH "Optional published voice-style JSON for integration."
)
add_test(
  NAME supertonic_integration
  COMMAND
    ${CMAKE_COMMAND} -DRUNNER=$<TARGET_FILE:supertonic_runner>
    -DPTE=${SUPERTONIC_INTEGRATION_PTE}
    -DASSET_DIR=${SUPERTONIC_INTEGRATION_ASSET_DIR}
    -DSTYLE=${SUPERTONIC_INTEGRATION_STYLE}
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/supertonic-integration.wav -P
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/run_integration.cmake
)
set_tests_properties(
  supertonic_integration
  PROPERTIES TIMEOUT 1800 SKIP_REGULAR_EXPRESSION
             "SKIP: Supertonic integration assets are unavailable"
)
add_test(
  NAME supertonic_jsonl_integration
  COMMAND
    ${CMAKE_COMMAND} -DRUNNER=$<TARGET_FILE:supertonic_runner>
    -DPTE=${SUPERTONIC_INTEGRATION_PTE}
    -DASSET_DIR=${SUPERTONIC_INTEGRATION_ASSET_DIR}
    -DSTYLE=${SUPERTONIC_INTEGRATION_STYLE}
    -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/jsonl-integration -P
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/run_jsonl_server_integration.cmake
)
set_tests_properties(
  supertonic_jsonl_integration
  PROPERTIES TIMEOUT 1800 SKIP_REGULAR_EXPRESSION
             "SKIP: Supertonic JSONL server assets are unavailable"
)
