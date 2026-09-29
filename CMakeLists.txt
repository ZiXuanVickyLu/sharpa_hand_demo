cmake_minimum_required(VERSION 3.24)

project(contact_solver LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_ROOT}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# One runtime flavour everywhere (mixing static/shared cudart breaks device links).
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CS_USE_DOUBLE "Use double precision for `real` (default, the paper's precision)" ON)
option(CS_ENABLE_GUI "Build the polyscope viewer" OFF)
option(CS_BUILD_APPS "Build the applications" ON)

if(CS_USE_DOUBLE)
    message(STATUS "contact_solver: real = double")
    add_compile_definitions(CS_USE_DOUBLE)
else()
    message(STATUS "contact_solver: real = float")
endif()

# ---------------------------------------------------------------------------
# CUDA architecture (auto-detected from the installed GPU, sm_89 on the dev box)
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(auto_detect_cuda_arch)
if(CUDA_ARCHS)
    string(REGEX MATCH "compute_([0-9]+)" _cs_cc "${CUDA_ARCHS}")
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_MATCH_1} CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# CCCL (Thrust/CUB/libcu++) moved to include/cccl in CUDA 13; the vendored wheels
# switch their include paths on this definition.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    add_compile_definitions(CCCL_VERSION_GREATER_EQUAL_13_0)
endif()

# ---------------------------------------------------------------------------
# Project paths baked into the binaries (assets / outputs live next to the sources)
# ---------------------------------------------------------------------------
set(CS_ASSET_PATH "${CMAKE_SOURCE_DIR}/asset")
set(CS_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/output")
set(CS_CONFIG_PATH "${CMAKE_SOURCE_DIR}/app/config")
add_compile_definitions(CS_ASSET_PATH="${CS_ASSET_PATH}/")
add_compile_definitions(CS_OUTPUT_PATH="${CS_OUTPUT_PATH}/")
add_compile_definitions(CS_CONFIG_PATH="${CS_CONFIG_PATH}/")
file(MAKE_DIRECTORY ${CS_OUTPUT_PATH})

# ---------------------------------------------------------------------------
# Dependencies, libraries, apps
# ---------------------------------------------------------------------------
add_subdirectory(ext)
add_subdirectory(src)

if(CS_BUILD_APPS)
    add_subdirectory(app)
endif()
