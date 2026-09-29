include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# ---------------------------------------------------------------------------
# Eigen 3.4 (header-only; we bypass Eigen's own CMake and export an INTERFACE target)
# ---------------------------------------------------------------------------
message(STATUS "fetch eigen...")
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
add_library(Eigen3_Eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
target_include_directories(Eigen3_Eigen INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)

# ---------------------------------------------------------------------------
# nlohmann::json
# ---------------------------------------------------------------------------
message(STATUS "fetch json...")
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF)
FetchContent_MakeAvailable(json)

# ---------------------------------------------------------------------------
# spdlog (with its bundled fmt)
# ---------------------------------------------------------------------------
message(STATUS "fetch spdlog...")
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    GIT_SHALLOW TRUE
)
set(SPDLOG_BUILD_EXAMPLE OFF)
set(SPDLOG_BUILD_TESTS OFF)
FetchContent_MakeAvailable(spdlog)

# ---------------------------------------------------------------------------
# cuda-api-wrappers (RAII around the CUDA runtime/driver: unique_span, streams, events)
# ---------------------------------------------------------------------------
message(STATUS "fetch cuda-api-wrappers...")
FetchContent_Declare(
    cuda-api-wrappers
    GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
    GIT_TAG v0.8.2-b1
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(cuda-api-wrappers)

# ---------------------------------------------------------------------------
# zlib + cnpy (.npy keyframe sequences)
# ---------------------------------------------------------------------------
find_package(ZLIB REQUIRED)
set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS})
set(ZLIB_LIBRARIES ZLIB::ZLIB)
add_subdirectory(cnpy)

# ---------------------------------------------------------------------------
# HouGeoIO (bgeo reader/writer), vendored from coupled_solver
# ---------------------------------------------------------------------------
add_subdirectory(HouGeoIO)

# ---------------------------------------------------------------------------
# Warp's templated McAdams 3x3 SVD (Apache-2.0 + MIT notice), used by src/core/svd3.cuh
# ---------------------------------------------------------------------------
add_library(warp_svd INTERFACE)
target_include_directories(warp_svd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/warp_svd)

# ---------------------------------------------------------------------------
# Optional viewer
# ---------------------------------------------------------------------------
if(CS_ENABLE_GUI)
    message(STATUS "fetch polyscope...")
    FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG v2.3.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(polyscope)
endif()

message(STATUS "contact_solver: external dependencies ready")
