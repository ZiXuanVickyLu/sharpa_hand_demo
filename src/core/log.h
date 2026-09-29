// log.h - thin wrapper over spdlog's default logger.
//
//   cs::log().info("step {} done", i);
//   cs::set_log_level("debug");   // trace|debug|info|warn|warning|error|critical|off
#pragma once
#ifndef CS_LOG_H
#define CS_LOG_H

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace cs {

/// The process-wide default logger.
inline spdlog::logger& log() { return *spdlog::default_logger_raw(); }

/// Sets the level of the default logger from its name; throws std::invalid_argument on an
/// unknown name.
inline void set_log_level(const std::string& level)
{
    const spdlog::level::level_enum lvl = spdlog::level::from_str(level);
    if (lvl == spdlog::level::off && level != "off") {
        throw std::invalid_argument("cs::set_log_level: unknown level '" + level +
                                    "' (expected trace|debug|info|warn|error|critical|off)");
    }
    spdlog::set_level(lvl);
}

}  // namespace cs

#endif  // CS_LOG_H
