#include "io/Log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <filesystem>
#include <mutex>

namespace blast {

namespace {
std::shared_ptr<spdlog::logger> g_log;
std::mutex                      g_log_mu;
}  // namespace

void init_log(const std::string& run_name, const std::string& log_dir) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_log) return;

    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    std::vector<spdlog::sink_ptr> sinks{console};

    if (!log_dir.empty()) {
        std::filesystem::create_directories(log_dir);
        auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            log_dir + "/" + run_name + ".log", /*truncate=*/false);
        file->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
        sinks.push_back(file);
    }

    g_log = std::make_shared<spdlog::logger>(run_name, sinks.begin(), sinks.end());
    g_log->set_level(spdlog::level::info);
    g_log->flush_on(spdlog::level::warn);
    spdlog::register_logger(g_log);
}

std::shared_ptr<spdlog::logger> log() {
    if (!g_log) init_log();
    return g_log;
}

}  // namespace blast
