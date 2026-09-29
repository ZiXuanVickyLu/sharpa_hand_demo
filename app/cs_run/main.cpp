// Headless runner: cs_run <config.json> [--frames N] [--output DIR] [--tets] [--quiet]
#include <solver/run.h>
#include <core/version.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    spdlog::info("contact_solver {}", cs::version_string());
    if (argc < 2) {
        spdlog::error("usage: cs_run <config.json> [--frames N] [--output DIR] [--tets] [--quiet]");
        return 2;
    }
    cs::RunOptions opt;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) opt.frames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) opt.output_dir = argv[++i];
        else if (std::strcmp(argv[i], "--tets") == 0) opt.export_tets = true;
        else if (std::strcmp(argv[i], "--quiet") == 0) opt.quiet = true;
        else {
            spdlog::error("unknown argument {}", argv[i]);
            return 2;
        }
    }
    try {
        return cs::run_simulation_file(argv[1], opt);
    } catch (const std::exception& e) {
        spdlog::error("fatal: {}", e.what());
        return 1;
    }
}
