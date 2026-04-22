



#ifndef OUTPUT_CUH
#define OUTPUT_CUH
#include <string>
#include "timer.cuh"

#include <fmt/core.h>
#include <fmt/chrono.h> 

class CountLogger {
public:
    




    CountLogger() = default;
    CountLogger(std::string name, std::string filepath, uint k){
        init(name, filepath, k);
    }

    void init(std::string name, std::string filepath, uint k) {
        name_ = std::move(name);
        filepath_ = std::move(filepath);
        buffer_.reserve(4096);
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
        filepath_= bin_file_path.parent_path() / (bin_file_path.stem().string() + "." + to_string(k) + "_gpu_throughput");
    }

    void count(uint counter_) {
        auto now = std::chrono::system_clock::now();
        fmt::format_to(std::back_inserter(buffer_),
                       "{}: {:%Y-%m-%d %H:%M:%S} result {}\n",
                       name_, now, counter_);
    }

    



    ~CountLogger() {
        if (buffer_.size() == 0 or filepath_.empty()) {
            return;
        }

        try {
            
            
            std::ofstream ofs(filepath_, std::ios::trunc | std::ios::binary);

            if (ofs.is_open()) {
                
                ofs.write(buffer_.data(), buffer_.size());
                ofs.close();
            } else {
                
                std::cerr << "[CountLogger Error] Failed to open file for writing: " << filepath_ << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[CountLogger Error] Exception in destructor: " << e.what() << std::endl;
        }
    }

    
    CountLogger(const CountLogger&) = delete;
    CountLogger& operator=(const CountLogger&) = delete;


private:
    std::string name_;
    std::string filepath_;
    fmt::memory_buffer buffer_; 
};
inline CountLogger count_logger;


class OutputFormat_class {

public:
    std::string static_part;
    OutputFormat_class() = default;
    OutputFormat_class(uint* X, uint X_length, uint* con_Y, uint conY_length, uint* base_curY, uint curY_length) {
        assign(X, X_length, con_Y, conY_length, base_curY, curY_length);
    }

    void OutputResult(const bool* selected, uint* varY, uint varY_length, uint count) {
        if (!OutputResults) return;
        fmt::print("Result {}: {}", count, static_part);
        
        for (uint i = 0; i < varY_length; i++) {
            if (selected[i]) {
                fmt::print("{} ", varY[i]);
            }
        }
        fmt::print("\n");
    }
    void OutputResult(uint count) {
        if (!OutputResults) return;
        fmt::println("Result {}: {}", count, static_part);
    }

    void OutputResult(const vector<uint>combinations, uint useful_id, uint k, uint* varY, uint varY_length, uint count) {
        if (!OutputResults) return;
        fmt::print("Result {}: {}", count, static_part);
        
        for (uint i = 0; i < k; i ++) {
            uint varY_id = combinations[useful_id * k + i];
            assert(varY_id < varY_length);
            fmt::print("{} ", varY[varY_id]);
        }
        fmt::print("\n");
    }

    void assign(uint* X, uint X_length, uint* con_Y, uint conY_length, uint* base_curY, uint curY_length) {

        
        vector<uint> x_vec(X, X + X_length);
        vector<uint> con_y_vec;
        vector<uint> base_cur_y_vec;
        static_part = fmt::format("X: {} | Y: ", fmt::join(x_vec, " "));
        if (base_curY != nullptr && curY_length != 0) {
            base_cur_y_vec.assign(base_curY, base_curY + curY_length);
            static_part += fmt::format("{} ", fmt::join(base_cur_y_vec, " "));
        }
        if (con_Y != nullptr && conY_length != 0) {
            con_y_vec.assign(con_Y, con_Y + conY_length);
            static_part += fmt::format("{} ", fmt::join(con_y_vec, " "));
        }
    }
    bool empty() const {
        return static_part.empty();
    }
    ~OutputFormat_class() = default;
};




void FormatResultToBuffer(fmt::memory_buffer& out_buf,
                          const std::vector<uint>& combinations,
                          uint useful_id,
                          uint k,
                          const uint* varY,
                          uint varY_length,
                          uint count,
                          const std::string& static_part) {

    
    fmt::format_to(std::back_inserter(out_buf), "Result {}: {}", count, static_part);

    
    for (uint i = 0; i < k; i++) {
        uint varY_id = combinations[useful_id * k + i];
        
        fmt::format_to(std::back_inserter(out_buf), "{} ", varY[varY_id]);
    }

    
    fmt::format_to(std::back_inserter(out_buf), "\n");
}

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>







class AsyncDoubleBufferWriter {
public:
    static constexpr size_t FLUSH_THRESHOLD = 32ULL * 1024 * 1024; 

    explicit AsyncDoubleBufferWriter(FILE* out = stdout) : out_(out) {
        
        buffers_[0].reserve(FLUSH_THRESHOLD + 2 * 1024 * 1024);
        buffers_[1].reserve(FLUSH_THRESHOLD + 2 * 1024 * 1024);
        writer_thread_ = std::thread(&AsyncDoubleBufferWriter::writer_loop, this);
    }

    ~AsyncDoubleBufferWriter() {
        flush_sync();                               
        {
            std::lock_guard<std::mutex> lock(mtx_);
            shutdown_ = true;
        }
        cv_task_.notify_one();
        if (writer_thread_.joinable())
            writer_thread_.join();
    }

    
    AsyncDoubleBufferWriter(const AsyncDoubleBufferWriter&)            = delete;
    AsyncDoubleBufferWriter& operator=(const AsyncDoubleBufferWriter&) = delete;

    

    
    fmt::memory_buffer& active_buffer() {
        return buffers_[active_];
    }

    
    void maybe_flush() {
        if (buffers_[active_].size() >= FLUSH_THRESHOLD) {
            swap_and_write_async();
        }
    }

    
    
    void flush_sync() {
        wait_pending();
        auto& buf = buffers_[active_];
        if (buf.size() > 0) {
            fwrite(buf.data(), 1, buf.size(), out_);
            buf.clear();
        }
    }

    size_t active_size() const { return buffers_[active_].size(); }

private:
    
    void wait_pending() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_done_.wait(lock, [this] { return !has_task_; });
    }

    
    void swap_and_write_async() {
        wait_pending();                       

        int to_write = active_;
        active_ = 1 - active_;               

        {
            std::lock_guard<std::mutex> lock(mtx_);
            task_idx_ = to_write;
            has_task_ = true;
        }
        cv_task_.notify_one();                
    }

    
    void writer_loop() {
        while (true) {
            int idx;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_task_.wait(lock, [this] { return has_task_ || shutdown_; });
                if (shutdown_ && !has_task_) return;   
                idx = task_idx_;
            }

            
            fwrite(buffers_[idx].data(), 1, buffers_[idx].size(), out_);
            buffers_[idx].clear();

            {
                std::lock_guard<std::mutex> lock(mtx_);
                has_task_ = false;
            }
            cv_done_.notify_one();            
        }
    }

    
    FILE* out_;
    fmt::memory_buffer buffers_[2];
    int active_ = 0;                          

    std::thread             writer_thread_;
    std::mutex              mtx_;
    std::condition_variable cv_task_;          
    std::condition_variable cv_done_;          
    bool has_task_  = false;
    bool shutdown_  = false;
    int  task_idx_  = -1;
};
inline AsyncDoubleBufferWriter doubleBufferWriter;


void ProcessBatchOptimized(
    const uint* buffer_h_isMax, 
    uint useful_batch,
    uint num, 
    
    const std::vector<uint>& one_batch,
    uint k,
    const uint* host_varY,
    uint varY_totalSize,
    std::string static_part,
    std::vector<std::pair<int, double>>& time_records,
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    uint& d_res_count 
) {
    using Clock = std::chrono::high_resolution_clock;

    
    
    std::vector<uint> valid_indices;
    valid_indices.reserve(num + 100); 

    for (uint i = 0; i < useful_batch; i++) {
        if (buffer_h_isMax[i] != 0) {
            valid_indices.push_back(i);
            
            
            if (valid_indices.size() >= num) break;
        }
    }

    
    
    fmt::memory_buffer batch_buffer;
    
    

    for (uint idx : valid_indices) {
        d_res_count++;

        
        if (OutputResults) {
            FormatResultToBuffer(batch_buffer, one_batch, idx, k, host_varY, varY_totalSize, d_res_count, static_part);
        }

        
        bool is_checkpoint = checkpoints.count(d_res_count);

        
        if (batch_buffer.size() > 1024 * 1024 || (is_checkpoint && batch_buffer.size() > 0)) {
            
            fwrite(batch_buffer.data(), 1, batch_buffer.size(), stdout);
            batch_buffer.clear();
        }

        
        if (is_checkpoint) {
            auto current_time = Clock::now();
            std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
            double cost_time = elapsed.count();
            time_records.push_back({d_res_count, cost_time});
            if (d_res_count == num)
                time_recorder.record(cost_time);

            
            fmt::println(stderr, "Output {}, cost Time: {} ms", d_res_count, cost_time);
        }

        if (d_res_count >= num) break;
    }
    if (num == 1000000)
        count_logger.count(d_res_count);
    
    if (batch_buffer.size() > 0) {
        fwrite(batch_buffer.data(), 1, batch_buffer.size(), stdout);
    }
}


void ProcessBatchOptimized2(
        const uint* buffer_h_isMax,
        uint useful_batch,
        uint num,
        const std::vector<uint>& one_batch,
        uint k,
        const uint* host_varY,
        uint varY_totalSize,
        std::string static_part,
        std::vector<std::pair<int, double>>& time_records,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
        uint& d_res_count
) {
    using Clock = std::chrono::high_resolution_clock;

    
    std::vector<uint> valid_indices;
    valid_indices.reserve(num + 100);

    for (uint i = 0; i < useful_batch; i++) {
        if (buffer_h_isMax[i] != 0) {
            valid_indices.push_back(i);
            if (valid_indices.size() >= num) break;
        }
    }

    
    for (uint idx : valid_indices) {
        d_res_count++;

        
        if (OutputResults) {
            FormatResultToBuffer(doubleBufferWriter.active_buffer(), one_batch, idx, k,
                                 host_varY, varY_totalSize, d_res_count, static_part);
        }

        bool is_checkpoint = checkpoints.count(d_res_count);

        if (is_checkpoint) {
            
            doubleBufferWriter.flush_sync();

            auto current_time = Clock::now();
            std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
            double cost_time = elapsed.count();
            time_records.push_back({d_res_count, cost_time});
            if (d_res_count == num)
                time_recorder.record(cost_time);
            fmt::println(stderr, "Output {}, cost Time: {} ms", d_res_count, cost_time);
        } else {
            
            doubleBufferWriter.maybe_flush();
        }

        if (d_res_count >= num) break;
    }

    if (num == 1000000)
        count_logger.count(d_res_count);

    
    
}
#endif 
