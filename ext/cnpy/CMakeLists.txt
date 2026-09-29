# set all .cpp .hpp .h as source files
file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp" "*.h")

# Generate static library
add_library(cnpy STATIC ${SOURCES})
target_link_libraries(cnpy PUBLIC ${ZLIB_LIBRARIES})
target_include_directories(cnpy PUBLIC ${ZLIB_INCLUDE_DIRS})
target_compile_features(cnpy PUBLIC cxx_std_17)
