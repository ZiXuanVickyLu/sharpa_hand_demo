# Two libraries:
#   cs_host   - host-only code (geometry, scene description, I/O); Eigen + json + spdlog.
#   cs_device - every CUDA translation unit of the solver in ONE library, so that device
#               functions shared between modules (ACCD, distances, SVD, ...) never need
#               cross-library device linking. Relocatable device code stays inside cs_device.
# Both glob their directories: a new module file is picked up by re-running CMake, and no
# shared CMake file needs editing when a module grows.

find_package(cuda-api-wrappers CONFIG REQUIRED)

set(CS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Host library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CS_HOST_SOURCES CONFIGURE_DEPENDS
    ${CS_SRC_DIR}/core/*.cpp
    ${CS_SRC_DIR}/geometry/*.cpp
    ${CS_SRC_DIR}/scene/*.cpp
    ${CS_SRC_DIR}/io/*.cpp
)
add_library(cs_host STATIC ${CS_HOST_SOURCES})
target_include_directories(cs_host PUBLIC ${CS_SRC_DIR} ${CS_SRC_DIR}/core)
target_link_libraries(cs_host PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    HouGeoIO
    cnpy
    CUDA::cudart   # host code may call the runtime API (timers, checks); shared cudart everywhere
)
target_compile_features(cs_host PUBLIC cxx_std_20)
if(UNIX)
    target_compile_options(cs_host PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
endif()

# ---------------------------------------------------------------------------
# Device library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CS_DEVICE_SOURCES CONFIGURE_DEPENDS
    ${CS_SRC_DIR}/core/*.cu
    ${CS_SRC_DIR}/libculbvh/*.cu
    ${CS_SRC_DIR}/ccd/*.cu
    ${CS_SRC_DIR}/gpu/*.cu
    ${CS_SRC_DIR}/fem/*.cu
    ${CS_SRC_DIR}/contact/*.cu
    ${CS_SRC_DIR}/linsys/*.cu
    ${CS_SRC_DIR}/solver/*.cu
)
add_library(cs_device STATIC ${CS_DEVICE_SOURCES})
set_target_properties(cs_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
)
target_compile_options(cs_device PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(cs_device PUBLIC
    ${CS_SRC_DIR}
    ${CS_SRC_DIR}/core
    ${CS_SRC_DIR}/ccd
)
target_link_libraries(cs_device PUBLIC
    cs_host
    warp_svd
    cuda-api-wrappers::runtime-and-driver
    CUDA::cudart
)
target_compile_features(cs_device PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Umbrella
# ---------------------------------------------------------------------------
add_library(contact_solver INTERFACE)
target_link_libraries(contact_solver INTERFACE cs_device cs_host)
