//
// Created by fxl on 2026/1/20.
//

#ifndef RECODER_H
#define RECODER_H

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include<cstring>
#include<set>
#include <fmt/core.h>
#include <fmt/chrono.h> // 需要包含此头文件以支持时间格式化
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace gpu {
	class TimeRecorder {
		public:
			TimeRecorder(
				const std::string& file_path, unsigned int k, unsigned int num,
				std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
				const std::string& dot_name)
				: k_(k), num_(num), start_time_(start_time) {
				// 1. 制作json文件路径
				fs::path dir_path = fs::path(file_path).parent_path();

				// 1.1 在file_path的所在目录中搜索.bin文件并获取其绝对路径
				fs::path bin_file_path;
				for (const auto& entry : fs::directory_iterator(dir_path)) {
					if (entry.is_regular_file() && entry.path().extension() == ".bin") {
						bin_file_path = fs::absolute(entry.path());
						break;
					}
				}

				if (bin_file_path.empty()) {
					std::cerr << "Error: No .bin file found in directory: " << dir_path
						<< std::endl;
					std::exit(1);
				}

				// 1.2 json文件路径 = bin文件去掉扩展名 + dot_name + ".json"
				json_file_path_ = bin_file_path.parent_path() /
				                  (bin_file_path.stem().string() + dot_name + ".json");

				// 2. 初始化json数据
				k_str_ = std::to_string(k_);
				num_str_ = std::to_string(num_);

				// 5小时对应的毫秒数: 5 * 60 * 60 * 1000 = 18,000,000 ms
				constexpr long long FIVE_HOURS_MS = 5LL * 60 * 60 * 1000;

				if (fs::exists(json_file_path_)) {
					// 打开并读取现有文件
					std::ifstream ifs(json_file_path_);
					if (ifs.is_open()) {
						try {
							ifs >> json_data_;
						}
						catch (const json::parse_error& e) {
							std::cerr << "JSON parse error: " << e.what() << std::endl;
							json_data_ = json::object();
						}
						ifs.close();
					}

					// 检查k层级是否存在
					if (json_data_.contains(k_str_)) {
						// k存在，检查num是否存在
						if (json_data_[k_str_].contains(num_str_)) {
							std::cout << "Record for k=" << k_ << ", num=" << num_
								<< " already exists. Exiting program." << std::endl;
							std::exit(0);
						}
						// num不存在，初始化为5小时的毫秒数
						json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
					}
					else {
						// k不存在，创建该层级并初始化num
						json_data_[k_str_] = json::object();
						json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
					}
				}
				else {
					// 文件不存在，创建新的json结构
					json_data_[k_str_] = json::object();
					json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
				}

				// 立即保存，确保5小时默认值被写入（防止程序崩溃丢失记录）
				save();
			}

			// 记录实际耗时（单位：毫秒）
			void record(double cost_time_ms) {
				json_data_[k_str_][num_str_] = cost_time_ms;
			}

			// 使用start_time自动计算并记录耗时
			void record() {
				auto end_time = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
						end_time - start_time_)
					.count();
				json_data_[k_str_][num_str_] = static_cast<double>(duration);
			}

			// 获取json文件路径（用于调试）
			std::string get_json_path() const { return json_file_path_.string(); }

			~TimeRecorder() { save(); }

		private:
			void save() {
				// 确保父目录存在
				fs::path parent_dir = json_file_path_.parent_path();
				if (!fs::exists(parent_dir)) {
					fs::create_directories(parent_dir);
				}

				std::ofstream ofs(json_file_path_);
				if (ofs.is_open()) {
					ofs << json_data_.dump(4); // 格式化输出，缩进4个空格
					ofs.close();
				}
				else {
					std::cerr << "Error: Cannot write to file: " << json_file_path_
						<< std::endl;
				}
			}

			unsigned int k_;
			unsigned int num_;
			std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
			fs::path json_file_path_;
			json json_data_;
			std::string k_str_;
			std::string num_str_;
	};
} // namespace gpu
namespace cpu {
	class TimeRecorder {
		public:
			TimeRecorder()= default;
			TimeRecorder(
				const std::string& file_path, unsigned int k, unsigned int num, std::set<int> checkpoints,
				const std::string& dot_name){
				init(file_path, k, num, checkpoints, dot_name);
			}
			void init(const std::string& file_path, unsigned int k, unsigned int num, std::set<int> checkpoints,
				const std::string& dot_name) {
				k_ = k;
				num_ = num;
				// 1. 制作json文件路径
				fs::path dir_path = fs::path(file_path).parent_path();

				// 1.1 在file_path的所在目录中搜索.bin文件并获取其绝对路径
				fs::path bin_file_path;
				for (const auto& entry : fs::directory_iterator(dir_path)) {
					if (entry.is_regular_file() && entry.path().extension() == ".bin") {
						bin_file_path = fs::absolute(entry.path());
						break;
					}
				}

				if (bin_file_path.empty()) {
					std::cerr << "Error: No .bin file found in directory: " << dir_path
						<< std::endl;
					std::exit(1);
				}

				// 1.2 json文件路径 = bin文件去掉扩展名 + dot_name + ".json"
				json_file_path_ = bin_file_path.parent_path() /
				                  (bin_file_path.stem().string() + dot_name + ".json");

				// 2. 初始化json数据
				k_str_ = std::to_string(k_);
				num_str_ = std::to_string(num_);

				// 5小时对应的毫秒数: 5 * 60 * 60 * 1000 = 18,000,000 ms
				constexpr long long FIVE_HOURS_MS = 5LL * 60 * 60 * 1000;

				if (fs::exists(json_file_path_)) {
					// 打开并读取现有文件
					std::ifstream ifs(json_file_path_);
					if (ifs.is_open()) {
						try {
							ifs >> json_data_;
						}
						catch (const json::parse_error& e) {
							std::cerr << "JSON parse error: " << e.what() << std::endl;
							json_data_ = json::object();
						}
						ifs.close();
					}

					// 检查k层级是否存在
					if (json_data_.contains(k_str_)) {
						// k存在，检查num是否存在

						if (json_data_[k_str_].contains(num_str_)) {
							std::cout << "Record for k=" << k_ << ", num=" << num_
								<< " already exists. Exiting program." << std::endl;
							std::exit(0);
						}
						for (auto num_: checkpoints) {
							std::string str = std::to_string(num_);
							if (not json_data_[k_str_].contains(num_str_))
								json_data_[k_str_][str] = FIVE_HOURS_MS;
						}
						// num不存在，初始化为5小时的毫秒数
						// json_data_[k_str_][num_str_] = FIVE_HOURS_MS; checkpoints里面包含了
					}
					else {
						// k不存在，创建该层级并初始化num
						json_data_[k_str_] = json::object();
						for (auto num_: checkpoints) {
							std::string str = std::to_string(num_);
							json_data_[k_str_][str] = FIVE_HOURS_MS;
						}
					}
				}
				else {
					// 文件不存在，创建新的json结构
					json_data_[k_str_] = json::object();
					for (auto num_: checkpoints) {
						std::string str = std::to_string(num_);
						json_data_[k_str_][str] = FIVE_HOURS_MS;
					}
				}
				// 立即保存，确保5小时默认值被写入（防止程序崩溃丢失记录）
				save();
			}

			// 记录实际耗时（单位：毫秒）
			void record(const std::string &num_str, double cost_time_ms) {
				std::cout << num_str << ", " << cost_time_ms << std::endl;
				json_data_[k_str_][num_str] = cost_time_ms;
				save();
			}
			void record(double cost_time_ms) {
				json_data_[k_str_][num_str_] = cost_time_ms;
			}
			void record(std::vector<std::pair<int, double> >& time_records) {
				for (const auto& record : time_records) {
					std::string tmp_num_ = std::to_string(record.first);
					json_data_[k_str_][tmp_num_] = record.second;
				}
			}

			// 获取json文件路径（用于调试）
			std::string get_json_path() const { return json_file_path_.string(); }

			~TimeRecorder() { save(); }
			void save() {
				if (json_file_path_.empty()) {
					return;
				}
				// 确保父目录存在
				fs::path parent_dir = json_file_path_.parent_path();
				if (!fs::exists(parent_dir)) {
					fs::create_directories(parent_dir);
				}

				std::ofstream ofs(json_file_path_);
				if (ofs.is_open()) {
					ofs << json_data_.dump(4); // 格式化输出，缩进4个空格
					ofs.close();
				}
				else {
					std::cerr << "Error: Cannot write to file: " << json_file_path_
						<< std::endl;
				}
			}

		private:


			unsigned int k_{};
			unsigned int num_{};
			fs::path json_file_path_;
			json json_data_;
			std::string k_str_;
			std::string num_str_;
	};
	inline TimeRecorder recorder;

} // namespace cpu
class CountLogger {
public:
    CountLogger() = default;
    CountLogger(std::string name, std::string filepath, uint k,
                size_t flush_threshold = 4096) {
        init(name, filepath, k, flush_threshold);
    }

    void init(std::string name, std::string filepath, uint k,
              size_t flush_threshold = 4096) {
        name_ = std::move(name);
        filepath_ = std::move(filepath);
        flush_threshold_ = flush_threshold;
        buffer_.reserve(flush_threshold_);

        fs::path dir_path = fs::path(filepath_).parent_path();
        fs::path bin_file_path;
        if (fs::exists(dir_path)) {
            for (const auto& entry : fs::directory_iterator(dir_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                    bin_file_path = fs::absolute(entry.path());
                    break;
                }
            }
        }
        if (bin_file_path.empty()) {
            std::cerr << __FILE__ ":" << __LINE__ << " Error: No .bin file found in directory: " << dir_path << std::endl;
            std::exit(1);
        }
        filepath_ = bin_file_path.parent_path() / (bin_file_path.stem().string() + "." + std::to_string(k) + "_" + name + "_throughput");

        // 首次打开时清空文件（如果需要追加模式，改为 std::ios::app）
        first_write_ = true;
    }

    void count(uint counter_) {
        auto now = std::chrono::system_clock::now();
        fmt::format_to(std::back_inserter(buffer_),
                       "{}: {:%Y-%m-%d %H:%M:%S} result {}\n",
                       name_, now, counter_);

        // 超过阈值时自动刷新到文件
        if (buffer_.size() >= flush_threshold_) {
            flush();
        }
    }

    /**
     * @brief 手动刷新缓冲区到文件
     */
    void flush() {
        if (buffer_.size() == 0 || filepath_.empty()) {
            return;
        }

        try {
            // 首次写入用 trunc 模式清空，后续用 app 模式追加
            auto mode = first_write_
                        ? (std::ios::trunc | std::ios::binary)
                        : (std::ios::app | std::ios::binary);

            std::ofstream ofs(filepath_, mode);
            if (ofs.is_open()) {
                ofs.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
                ofs.close();
                buffer_.clear();  // 清空缓冲区
                first_write_ = false;
            } else {
                std::cerr << "[CountLogger Error] Failed to open file for writing: " << filepath_ << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[CountLogger Error] Exception in flush: " << e.what() << std::endl;
        }
    }

    ~CountLogger() {
        flush();  // 析构时刷新剩余数据
    }

    CountLogger(const CountLogger&) = delete;
    CountLogger& operator=(const CountLogger&) = delete;

    // 支持移动语义
    CountLogger(CountLogger&&) = default;
    CountLogger& operator=(CountLogger&&) = default;

private:
    std::string name_;
    std::string filepath_;
    fmt::memory_buffer buffer_;
    size_t flush_threshold_ = 4096;  // 刷新阈值
    bool first_write_ = true;        // 标记是否首次写入
};
inline CountLogger count_logger;
#endif // RECODER_H
