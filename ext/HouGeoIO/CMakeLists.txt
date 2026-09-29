cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
project(BGEOIO)
set(PROJECT_DIR HouGeoIO)
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

add_subdirectory(extern)

file(GLOB INCS src/*.h)
file(GLOB SRCS src/*.cpp)

add_library(HouGeoIO ${INCS} ${SRCS} )

target_link_libraries(HouGeoIO PUBLIC HouJson Eigen3::Eigen)
target_include_directories(HouGeoIO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(HouGeoIO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern)
target_include_directories(HouGeoIO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)