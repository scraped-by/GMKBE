



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
#include <fmt/chrono.h>
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

				fs::path dir_path = fs::path(file_path).parent_path();


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


				json_file_path_ = bin_file_path.parent_path() /
				                  (bin_file_path.stem().string() + dot_name + ".json");


				k_str_ = std::to_string(k_);
				num_str_ = std::to_string(num_);


				constexpr long long FIVE_HOURS_MS = 5LL * 60 * 60 * 1000;

				if (fs::exists(json_file_path_)) {

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


					if (json_data_.contains(k_str_)) {

						if (json_data_[k_str_].contains(num_str_)) {
							std::cout << "Record for k=" << k_ << ", num=" << num_
								<< " already exists. Exiting program." << std::endl;
							std::exit(0);
						}

						json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
					}
					else {

						json_data_[k_str_] = json::object();
						json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
					}
				}
				else {

					json_data_[k_str_] = json::object();
					json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
				}


				save();
			}


			void record(double cost_time_ms) {
				json_data_[k_str_][num_str_] = cost_time_ms;
			}


			void record() {
				auto end_time = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
						end_time - start_time_)
					.count();
				json_data_[k_str_][num_str_] = static_cast<double>(duration);
			}


			std::string get_json_path() const { return json_file_path_.string(); }

			~TimeRecorder() { save(); }

		private:
			void save() {

				fs::path parent_dir = json_file_path_.parent_path();
				if (!fs::exists(parent_dir)) {
					fs::create_directories(parent_dir);
				}

				std::ofstream ofs(json_file_path_);
				if (ofs.is_open()) {
					ofs << json_data_.dump(4);
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
}
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

				fs::path dir_path = fs::path(file_path).parent_path();


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


				json_file_path_ = bin_file_path.parent_path() /
				                  (bin_file_path.stem().string() + dot_name + ".json");


				k_str_ = std::to_string(k_);
				num_str_ = std::to_string(num_);


				constexpr long long FIVE_HOURS_MS = 5LL * 60 * 60 * 1000;

				if (fs::exists(json_file_path_)) {

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


					if (json_data_.contains(k_str_)) {


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


					}
					else {

						json_data_[k_str_] = json::object();
						for (auto num_: checkpoints) {
							std::string str = std::to_string(num_);
							json_data_[k_str_][str] = FIVE_HOURS_MS;
						}
					}
				}
				else {

					json_data_[k_str_] = json::object();
					for (auto num_: checkpoints) {
						std::string str = std::to_string(num_);
						json_data_[k_str_][str] = FIVE_HOURS_MS;
					}
				}

				save();
			}


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


			std::string get_json_path() const { return json_file_path_.string(); }

			~TimeRecorder() { save(); }
			void save() {
				if (json_file_path_.empty()) {
					return;
				}

				fs::path parent_dir = json_file_path_.parent_path();
				if (!fs::exists(parent_dir)) {
					fs::create_directories(parent_dir);
				}

				std::ofstream ofs(json_file_path_);
				if (ofs.is_open()) {
					ofs << json_data_.dump(4);
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

}
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


        first_write_ = true;
    }

    void count(uint counter_) {
        auto now = std::chrono::system_clock::now();
        fmt::format_to(std::back_inserter(buffer_),
                       "{}: {:%Y-%m-%d %H:%M:%S} result {}\n",
                       name_, now, counter_);


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

            auto mode = first_write_
                        ? (std::ios::trunc | std::ios::binary)
                        : (std::ios::app | std::ios::binary);

            std::ofstream ofs(filepath_, mode);
            if (ofs.is_open()) {
                ofs.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
                ofs.close();
                buffer_.clear();
                first_write_ = false;
            } else {
                std::cerr << "[CountLogger Error] Failed to open file for writing: " << filepath_ << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[CountLogger Error] Exception in flush: " << e.what() << std::endl;
        }
    }

    ~CountLogger() {
        flush();
    }

    CountLogger(const CountLogger&) = delete;
    CountLogger& operator=(const CountLogger&) = delete;


    CountLogger(CountLogger&&) = default;
    CountLogger& operator=(CountLogger&&) = default;

private:
    std::string name_;
    std::string filepath_;
    fmt::memory_buffer buffer_;
    size_t flush_threshold_ = 4096;
    bool first_write_ = true;
};
inline CountLogger count_logger;
#endif
