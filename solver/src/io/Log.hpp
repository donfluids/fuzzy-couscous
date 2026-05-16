#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace blast {

void init_log(const std::string& run_name = "blast", const std::string& log_dir = "");

std::shared_ptr<spdlog::logger> log();

#define BLAST_INFO(...)  ::blast::log()->info(__VA_ARGS__)
#define BLAST_WARN(...)  ::blast::log()->warn(__VA_ARGS__)
#define BLAST_ERROR(...) ::blast::log()->error(__VA_ARGS__)
#define BLAST_DEBUG(...) ::blast::log()->debug(__VA_ARGS__)

}  // namespace blast
