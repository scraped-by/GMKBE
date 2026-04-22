#ifndef SEARCH_LOGGER_HPP
#define SEARCH_LOGGER_HPP

#include <string>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <utility>
#include <iomanip>
#include <filesystem>

class SearchLogger {
private:
    inline static SearchLogger* instance = nullptr;

    std::filesystem::path jsonPath;
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    uint64_t nextThreshold;
    std::vector<std::pair<uint64_t, double>> records;
    bool saved;

    static void atexitHandler() {
        if (instance) instance->save();
    }

public:
    void save() {
        if (saved || records.empty()) return;
        std::ofstream ofs(jsonPath);
        ofs << std::fixed << std::setprecision(3);
        ofs << "{\n";
        for (size_t i = 0; i < records.size(); ++i) {
            ofs << "    \"" << records[i].first << "\": " << records[i].second;
            if (i + 1 < records.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "}\n";
        saved = true;
    }

    SearchLogger() : nextThreshold(0), saved(false) {}

    SearchLogger(const std::string& datasetPath, int k) : nextThreshold(1), saved(false) {
        std::filesystem::path p(datasetPath);

        std::filesystem::path dir = p.parent_path();
        if (dir.empty()) {
            dir = ".";
        }

        std::filesystem::path targetDir = dir / "MaximalBiPlex";
        if (!std::filesystem::exists(targetDir)) {
            std::filesystem::create_directories(targetDir);
        }

        std::string newFileName = p.stem().string() + "-k" + std::to_string(k) + ".json";
        jsonPath = targetDir / newFileName;

        instance = this;
        std::atexit(atexitHandler);
    }

    ~SearchLogger() {
        save();
        if (instance == this) instance = nullptr;
    }

    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    void check(uint64_t count) {
        if (nextThreshold == 0 || count != nextThreshold) return;

        double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - startTime
        ).count();

        records.push_back({nextThreshold, ms});

        if (nextThreshold <= UINT64_MAX / 10)
            nextThreshold *= 10;
        else
            nextThreshold = 0;
    }
};

#endif