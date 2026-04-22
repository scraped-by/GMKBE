



#ifndef TIMER_CUH
#define TIMER_CUH
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <gpu_utils.cuh>

namespace fs = std::filesystem;
using json = nlohmann::json;






class ScopedCPUTimer {
	public:
		ScopedCPUTimer(double& result_ms)
			: result_ms_(result_ms), start_time_(std::chrono::high_resolution_clock::now()) {
		}

		~ScopedCPUTimer() {
			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
			result_ms_ = duration / 1000.0; 
		}

	private:
		double& result_ms_; 
		std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};







class ScopedGPUTimer {
	public:
		ScopedGPUTimer(double& result_ms, cudaStream_t stream = 0)
			: result_ms_(result_ms), stream_(stream) {
			CUDA_ERROR_CHECK(cudaEventCreate(&start_));
			CUDA_ERROR_CHECK(cudaEventCreate(&stop_));
			CUDA_ERROR_CHECK(cudaEventRecord(start_, stream_));
		}

		~ScopedGPUTimer() {
			CUDA_ERROR_CHECK(cudaEventRecord(stop_, stream_));
			CUDA_ERROR_CHECK(cudaEventSynchronize(stop_));

			float milliseconds = 0;
			CUDA_ERROR_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
			result_ms_ = milliseconds;

			CUDA_ERROR_CHECK(cudaEventDestroy(start_));
			CUDA_ERROR_CHECK(cudaEventDestroy(stop_));
		}

	private:
		double& result_ms_; 
		cudaEvent_t start_, stop_;
		cudaStream_t stream_;
};

class TimeRecorder {
	public:
		
		unsigned int k_ = 0;

		
		TimeRecorder() : k_(0), num_(0) {
		}

		
		TimeRecorder(const std::string& file_path, unsigned int k, unsigned int num, const std::string& dot_name){
			init(file_path, k, num, dot_name);
		}

		
		~TimeRecorder() {
			if (!json_file_path_.empty()) {
				save();
			}
		}

		
		void record(double cost_time_ms) {
			
			if (!k_str_.empty() && !num_str_.empty()) {
				json_data_[k_str_][num_str_] = cost_time_ms;
			}
		}
		void init(const std::string& file_path, unsigned int k, unsigned int num, const std::string& dot_name) {
			k_ = k;
			num_ = num;
			
			fs::path dir_path = fs::path(file_path).parent_path();
			
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
				std::cerr << __FILE__ ":" << __LINE__ << "Error: No .bin file found in directory: " << dir_path << std::endl;
				std::exit(1);
			}

			
			json_file_path_ = bin_file_path.parent_path() / (bin_file_path.stem().string() + dot_name + ".json");

			
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
					
					
					
					
					
					json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
				}
				else {
					json_data_[k_str_] = json::object();
					json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
				}
			}
			else {
				
				json_data_ = json::object(); 
				json_data_[k_str_] = json::object();
				json_data_[k_str_][num_str_] = FIVE_HOURS_MS;
			}

			save();
		}

		std::string get_json_path() const { return json_file_path_.string(); }
		void save() {
			if (json_file_path_.empty()) return;

			
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
				std::cerr << "Error: Cannot write to file: " << json_file_path_ << std::endl;
			}
		}

	private:
		unsigned int num_ = 0;
		fs::path json_file_path_;
		json json_data_;
		std::string k_str_;
		std::string num_str_;


};



#ifdef __CUDACC__


inline TimeRecorder time_recorder;
#else
    TimeRecorder time_recorder;
#endif
#endif 
