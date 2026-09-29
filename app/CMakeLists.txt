find_package(cuda-api-wrappers CONFIG REQUIRED)

# Headless runner: cs_run <config.json> [--frames N] [--output DIR]
add_executable(cs_run cs_run/main.cpp)
target_link_libraries(cs_run PRIVATE contact_solver nlohmann_json::nlohmann_json spdlog::spdlog)
target_compile_features(cs_run PRIVATE cxx_std_20)

if(CS_ENABLE_GUI)
    add_executable(cs_view cs_view/main.cpp)
    target_link_libraries(cs_view PRIVATE contact_solver nlohmann_json::nlohmann_json spdlog::spdlog polyscope)
    target_compile_features(cs_view PRIVATE cxx_std_20)
endif()
