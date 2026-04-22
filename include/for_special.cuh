



#ifndef FOR_SPECIAL_H
#define FOR_SPECIAL_H
#include <output.cuh>
#include <partition.cuh>
#include <hashTable.cuh>
#include <for_checkmaximality.cuh>
#include <timer.cuh>
#define LOG_WARN(msg) std::cerr << "[WARNING] " << msg << std::endl;



bool check_memory_availability(
    const uint64_t _batch_size,
    const uint _k,
    const uint _ext_q_size,
    const uint _ext_p_size,
    Partition_v3& var_Y, Partition_v3& con_Y,
    bool verbose = false,
    bool check_uint32_overflow = false 
) {
    
    
    if (check_uint32_overflow) {
        const uint64_t LIMIT_32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

        
        uint64_t q_total = static_cast<uint64_t>(_ext_q_size) * _batch_size;
        uint64_t p_total = static_cast<uint64_t>(_ext_p_size) * _batch_size;
        uint64_t max_val = (q_total > p_total) ? q_total : p_total;

        
        
        if (max_val >= LIMIT_32) {
            if (verbose) {
                std::cout << "--- Constraint Check Failed ---" << std::endl;
                std::cout << "Error: 32-bit index overflow constraint violated." << std::endl;
                std::cout << "Max(ext_size * batch) must be < UINT32_MAX." << std::endl;
                std::cout << "Calculated Max: " << max_val << " (Limit: " << LIMIT_32 << ")" << std::endl;
                std::cout << "-------------------------------" << std::endl;
            }
            return false;
        }
    }

    

    
    const size_t MAX_SIZE = std::numeric_limits<size_t>::max();
    size_t required_device_memory = 0;
    bool overflow_detected = false;

    
    auto safe_add = [&](size_t& current_val, size_t add_val) {
        if (MAX_SIZE - current_val < add_val) {
            overflow_detected = true;
        } else {
            current_val += add_val;
        }
    };

    
    auto safe_mul = [&](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > MAX_SIZE / b) {
            overflow_detected = true;
            return 0;
        }
        return a * b;
    };

    

    
    size_t part_a = sizeof(uint);
    part_a = safe_mul(part_a, static_cast<size_t>(_k));
    part_a = safe_mul(part_a, static_cast<size_t>(_batch_size));
    safe_add(required_device_memory, part_a);

    
    double total_size_d = static_cast<double>(con_Y.totalSize) + static_cast<double>(var_Y.totalSize);
    double part_b_d = static_cast<double>(sizeof(uint)) * total_size_d / 0.4;

    if (part_b_d > static_cast<double>(MAX_SIZE)) {
        overflow_detected = true;
    } else {
        safe_add(required_device_memory, static_cast<size_t>(part_b_d));
    }

    
    size_t part_c = 2;
    part_c = safe_mul(part_c, static_cast<size_t>(_batch_size));
    part_c = safe_mul(part_c, sizeof(uint));
    safe_add(required_device_memory, part_c);

    
    size_t part_d = MaximalityCheckBuffer::totalUsedMemory(_batch_size, _k, _ext_q_size, _ext_p_size, verbose);
    safe_add(required_device_memory, part_d);

    
    if (overflow_detected) {
        if (verbose) {
            std::cout << "--- GPU Memory Check ---" << std::endl;
            std::cout << "Error: Integer overflow detected during memory calculation." << std::endl;
            std::cout << "Memory is INSUFFICIENT." << std::endl;
            std::cout << "------------------------" << std::endl;
        }
        return false;
    }

    
    size_t free_memory = 0;
    size_t total_memory = 0;

    cudaError_t cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
    if (cuda_status != cudaSuccess) {
        if (verbose) {
            std::cout << "CUDA Error: " << cudaGetErrorString(cuda_status) << std::endl;
        }
        return false;
    }

    
    bool is_enough = required_device_memory <= free_memory;

    if (verbose) {
        double required_mb = static_cast<double>(required_device_memory) / (1024.0 * 1024.0);
        double free_mb = static_cast<double>(free_memory) / (1024.0 * 1024.0);
        double total_mb = static_cast<double>(total_memory) / (1024.0 * 1024.0);

        std::cout << "--- GPU Memory Check ---" << std::endl;
        std::cout << "Required: " << required_mb << " MB (" << required_device_memory << " bytes)" << std::endl;
        std::cout << "Available: " << free_mb << " MB (" << free_memory << " bytes)" << std::endl;
        std::cout << "Total: " << total_mb << " MB (" << total_memory << " bytes)" << std::endl;
        std::cout << "Memory is " << (is_enough ? "SUFFICIENT" : "INSUFFICIENT") << "." << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    return is_enough;
}

bool check_memory_availability_optimized(
    const uint64_t _batch_size,
    const uint _k,
    const uint _ext_q_size,
    const uint _ext_p_size,
    Partition_v3& var_Y, Partition_v3& con_Y,
    bool verbose = false,
    bool check_uint32_overflow = false 
) {
    
    
    if (check_uint32_overflow) {
        const uint64_t LIMIT_32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

        
        uint64_t q_total = static_cast<uint64_t>(_ext_q_size) * _batch_size;
        uint64_t p_total = static_cast<uint64_t>(_ext_p_size) * _batch_size;
        uint64_t max_val = (q_total > p_total) ? q_total : p_total;

        
        
        if (max_val >= LIMIT_32) {
            if (verbose) {
                std::cout << "--- Constraint Check Failed ---" << std::endl;
                std::cout << "Error: 32-bit index overflow constraint violated." << std::endl;
                std::cout << "Max(ext_size * batch) must be < UINT32_MAX." << std::endl;
                std::cout << "Calculated Max: " << max_val << " (Limit: " << LIMIT_32 << ")" << std::endl;
                std::cout << "-------------------------------" << std::endl;
            }
            return false;
        }
    }

    

    
    const size_t MAX_SIZE = std::numeric_limits<size_t>::max();
    size_t required_device_memory = 0;
    bool overflow_detected = false;

    
    auto safe_add = [&](size_t& current_val, size_t add_val) {
        if (MAX_SIZE - current_val < add_val) {
            overflow_detected = true;
        } else {
            current_val += add_val;
        }
    };

    
    auto safe_mul = [&](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > MAX_SIZE / b) {
            overflow_detected = true;
            return 0;
        }
        return a * b;
    };

    

    
    size_t part_a = sizeof(uint);
    part_a = safe_mul(part_a, static_cast<size_t>(_k));
    part_a = safe_mul(part_a, static_cast<size_t>(_batch_size));
    safe_add(required_device_memory, part_a);

    
    double total_size_d = static_cast<double>(con_Y.totalSize) + static_cast<double>(var_Y.totalSize);
    double part_b_d = static_cast<double>(sizeof(uint)) * total_size_d / 0.4;

    if (part_b_d > static_cast<double>(MAX_SIZE)) {
        overflow_detected = true;
    } else {
        safe_add(required_device_memory, static_cast<size_t>(part_b_d));
    }

    
    size_t part_c = 2;
    part_c = safe_mul(part_c, static_cast<size_t>(_batch_size));
    part_c = safe_mul(part_c, sizeof(uint));
    safe_add(required_device_memory, part_c);

    
    size_t part_d = MaximalityCheckBuffer_optmized::totalUsedMemory(_batch_size, _ext_q_size, _ext_p_size);
    safe_add(required_device_memory, part_d);

    
    if (overflow_detected) {
        if (verbose) {
            std::cout << "--- GPU Memory Check ---" << std::endl;
            std::cout << "Error: Integer overflow detected during memory calculation." << std::endl;
            std::cout << "Memory is INSUFFICIENT." << std::endl;
            std::cout << "------------------------" << std::endl;
        }
        return false;
    }

    
    size_t free_memory = 0;
    size_t total_memory = 0;

    cudaError_t cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
    if (cuda_status != cudaSuccess) {
        if (verbose) {
            std::cout << "CUDA Error: " << cudaGetErrorString(cuda_status) << std::endl;
        }
        return false;
    }

    
    bool is_enough = required_device_memory <= free_memory;

    if (verbose) {
        double required_mb = static_cast<double>(required_device_memory) / (1024.0 * 1024.0);
        double free_mb = static_cast<double>(free_memory) / (1024.0 * 1024.0);
        double total_mb = static_cast<double>(total_memory) / (1024.0 * 1024.0);

        std::cout << "--- GPU Memory Check ---" << std::endl;
        std::cout << "Required: " << required_mb << " MB (" << required_device_memory << " bytes)" << std::endl;
        std::cout << "Available: " << free_mb << " MB (" << free_memory << " bytes)" << std::endl;
        std::cout << "Total: " << total_mb << " MB (" << total_memory << " bytes)" << std::endl;
        std::cout << "Memory is " << (is_enough ? "SUFFICIENT" : "INSUFFICIENT") << "." << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    return is_enough;
}

void Special_GPU(Partition_v3& X, Partition_v3& var_Y, Partition_v3& con_Y, Partition_v3& ext_p, Partition_v3& ext_q,
    arrMapTable& g_state_viewer, bool* device_G_prune, uint*& device_degrees,
    uint*& device_neighbors, uint*& device_neighbors_offset, const uint component_length, const uint theta, const uint k, bool OutputResults, cudaStream_t& stream) {
    OutputFormat_class outputFormat;
    uint* host_varY = nullptr;

    KLoopGenerator_v2 generator(k, var_Y.totalSize);
    uint* combinations;
    auto const [totalCombinationNum, flag] = generator.combination_total_num(k, var_Y.totalSize);
    if (flag == false) {
        cerr << "The number of all combinations might be too large" << endl;
    }
    uint64_t batch_size;
    const uint64_t MAX_POW2_IN_UINT64 = 1ULL << 63;
    
    
    if (totalCombinationNum > MAX_POW2_IN_UINT64) {
        
        LOG_WARN("totalCombinationNum (" << totalCombinationNum
                 << ") is too large to align to the next power of 2 within uint64_t.");
        LOG_WARN("Falling back to using totalCombinationNum or UINT64_MAX as batch_size.");
        batch_size = totalCombinationNum;

    } else {
        
        batch_size = next_power_of_2(totalCombinationNum);
    }
    while (batch_size) {
        if (check_memory_availability(batch_size, k, ext_q.totalSize, ext_p.totalSize, var_Y, con_Y)) {
            if (batch_size > totalCombinationNum && check_memory_availability(totalCombinationNum, k, ext_q.totalSize, ext_p.totalSize, var_Y, con_Y))
                batch_size = totalCombinationNum;
            uint64_t tmp_batch_size = max(batch_size * ext_p.totalSize, batch_size * ext_q.totalSize);
            if (tmp_batch_size > UINT32_MAX) {
                
                batch_size /= 1.05;
                continue;
            }
            break;
        }
        batch_size /= 1.05;
    }
    batch_size = min(static_cast<uint64_t>(num) * 4, batch_size);
    if (batch_size == 0) {
        ERROR_CALL("设备空间不足");
    }
    CUDA_ERROR_CHECK(cudaMallocAsync(&combinations, sizeof(uint) * k * batch_size, stream));
    Utils::HashTable::cudaHashTable3 hash_table(con_Y.totalSize + var_Y.totalSize, 0.4, stream);
    if (con_Y.totalSize) {
        hash_table.insert(con_Y.partition, con_Y.totalSize, var_Y.totalSize + 1, stream);
    }
    if (var_Y.totalSize) {
        hash_table.insert(var_Y.partition, var_Y.totalSize, stream);
    }
    MaximalityCheckBuffer buffer(batch_size, k, ext_q.totalSize, ext_p.totalSize, stream);

    int currentIteration = 0;
    fmt::println("not optimized version");
    fmt::print("batch_size = {}", batch_size);
    cout << endl;
    const uint64_t batch_nums = (totalCombinationNum + batch_size - 1) / batch_size;

    uint grid_num = CALC_GRID_DIM(var_Y.totalSize, THREADS_PER_BLOCK);
    changeGPrune<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(var_Y.partition, var_Y.totalSize, device_G_prune);
    while (generator.hasMoreCombinations()) {
        auto one_batch    = generator.getNextBatch_continuousMemory(batch_size);
        currentIteration++;
        fmt::print("batch_nums: {}, currentIteration: {}", batch_nums, currentIteration);
        cout << endl;
        size_t useful_batch = one_batch.size() / k;
        CUDA_ERROR_CHECK(cudaMemcpy(
            combinations, one_batch.data(), sizeof(uint) * k * useful_batch, cudaMemcpyHostToDevice));
        CheckMaximality_for_special(device_degrees, device_neighbors, device_neighbors_offset, device_G_prune,
            combinations, useful_batch, g_state_viewer, hash_table, k, X, con_Y,
            var_Y, ext_p, ext_q, buffer, component_length, theta, k, stream);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        if (outputFormat.empty()) {
            uint *host_X = new uint[X.totalSize], *host_conY = new uint[con_Y.totalSize];
            host_varY       = new uint[var_Y.totalSize];
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                host_X, X.partition, sizeof(uint) * X.totalSize, cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(host_conY, con_Y.partition, sizeof(uint) * con_Y.totalSize,
                cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(host_varY, var_Y.partition, sizeof(uint) * var_Y.totalSize,
                cudaMemcpyDeviceToHost, stream));
            outputFormat.assign(host_X, X.totalSize, host_conY, con_Y.totalSize, nullptr, 0);
            delete[] host_X, host_conY;
        }
        ProcessBatchOptimized(buffer.h_isMax, useful_batch,
            num, one_batch, k, host_varY, var_Y.totalSize,
            outputFormat.static_part, time_records, start_time, d_res_count);
        if (d_res_count >= num) return;
    }
    if (host_varY != nullptr) {
        delete[] host_varY;
    }
    CUDA_ERROR_CHECK(cudaFreeAsync(combinations, stream));
}

__global__ void insert_conY(int*__restrict__ G_temp, const uint*__restrict__ conY, const uint length_conY) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint grid_size = blockDim.x * blockDim.y * blockDim.z;
    for (uint idx = tid; idx < length_conY; idx += grid_size) {
        G_temp[conY[idx]] = 1;
    }
}
__global__ void insert_varY(int*__restrict__ G_temp, const uint*__restrict__ varY, const uint length_varY) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint grid_size = blockDim.x * blockDim.y * blockDim.z;
    for (uint idx = tid; idx < length_varY; idx += grid_size) {
        G_temp[varY[idx]] = idx;
    }
}















































































bool processLockFree(const MaximalityCheckBuffer_optmized& buffer, size_t &useful_batch, OutputFormat_class& outputFormat,
    vector<uint> &one_batch, int k, uint* host_varY, Partition_v3& var_Y) {

    std::atomic<uint> d_res_count_{0};
    std::atomic<bool> done{false};
    std::atomic<uint> next_checkpoint_idx{0};

    std::vector<uint> checkpoint_list(checkpoints.begin(), checkpoints.end());
    std::sort(checkpoint_list.begin(), checkpoint_list.end());

    
    std::thread monitor([&]() {
        while (!done) {
            uint current = d_res_count_.load(std::memory_order_relaxed);
            uint cp_idx = next_checkpoint_idx.load();

            while (cp_idx < checkpoint_list.size() &&
                   checkpoint_list[cp_idx] <= current) {
                auto elapsed = Clock::now() - start_time;
                fmt::println("Output {}, Time: {} ms",
                           checkpoint_list[cp_idx],
                           std::chrono::duration<double, std::milli>(elapsed).count());
                next_checkpoint_idx.fetch_add(1);
                cp_idx++;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    
    const uint num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    std::mutex output_mutex;  

    for (uint t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t]() {
            for (uint i = t; i < useful_batch && !done; i += num_threads) {
                if (buffer.h_isMax[i] != 0) {
                    uint my_count = d_res_count_.fetch_add(1,
                                        std::memory_order_relaxed) + 1;

                    if (OutputResults) {
                        std::lock_guard<std::mutex> lock(output_mutex);
                        outputFormat.OutputResult(one_batch, i, k,
                                                 host_varY, var_Y.totalSize, my_count);
                    }

                    if (my_count >= num) {
                        done = true;
                    }
                }
            }
        });
    }

    for (auto& w : workers) w.join();
    done = true;
    monitor.join();
    d_res_count = d_res_count_.load(std::memory_order_relaxed);
    return done;
}

void Special_GPU_optimized(Partition_v3& X, Partition_v3& var_Y, Partition_v3& con_Y, Partition_v3& ext_p, Partition_v3& ext_q,
    arrMapTable& g_state_viewer, bool* device_G_prune, uint*& device_degrees,
    uint*& device_neighbors, uint*& device_neighbors_offset, const uint right_length, const uint theta, const uint k, bool OutputResults, cudaStream_t& stream) {
    OutputFormat_class outputFormat;
    uint* host_varY = nullptr;

    KLoopGenerator_v2 generator(k, var_Y.totalSize);
    uint* combinations, *curYs;
    auto const [totalCombinationNum, flag] = generator.combination_total_num(k, var_Y.totalSize);
    if (flag == false) {
        cerr << "The number of all combinations might be too large" << endl;
    }
    uint64_t batch_size;
    const uint64_t MAX_POW2_IN_UINT64 = 1ULL << 63;
    
    
    if (totalCombinationNum > MAX_POW2_IN_UINT64) {
        
        LOG_WARN("totalCombinationNum (" << totalCombinationNum
                 << ") is too large to align to the next power of 2 within uint64_t.");
        LOG_WARN("Falling back to using totalCombinationNum or UINT64_MAX as batch_size.");
        batch_size = totalCombinationNum;

    } else {
        
        batch_size = next_power_of_2(totalCombinationNum / 4);
    }
    while (batch_size) {
        if (check_memory_availability_optimized(batch_size, k, ext_q.totalSize, ext_p.totalSize, var_Y, con_Y, false, true)) {
            if (batch_size > totalCombinationNum && check_memory_availability_optimized(totalCombinationNum, k, ext_q.totalSize, ext_p.totalSize, var_Y, con_Y, false, true))
                batch_size = totalCombinationNum;
            break;
        }
        batch_size /= 1.05;
    }
    if (batch_size == 0) {
        ERROR_CALL("设备空间不足");
    }
    batch_size = min(static_cast<uint64_t>(num) * 4, batch_size);
    
        
    CUDA_ERROR_CHECK(cudaMallocAsync(&combinations, sizeof(uint) * k * batch_size, stream));
    CUDA_ERROR_CHECK(cudaMallocAsync(&curYs, sizeof(uint) * k * batch_size, stream));


    Utils::HashTable::cudaHashTable3 conY_curY_hashTable(con_Y.totalSize + var_Y.totalSize, 0.4, stream);
    if (con_Y.totalSize) {
        conY_curY_hashTable.insert(con_Y.partition, con_Y.totalSize, var_Y.totalSize + 1, stream);
    }
    if (var_Y.totalSize) {
        conY_curY_hashTable.insert(var_Y.partition, var_Y.totalSize, stream);
    }
    
    
    MaximalityCheckBuffer_optmized buffer(batch_size, k, ext_q.totalSize, ext_p.totalSize, stream);

    int currentIteration = 0;
    fmt::print("batch_size = {}", batch_size);
    cout << endl;
    const uint64_t batch_nums = (totalCombinationNum + batch_size - 1) / batch_size;

    uint grid_num = CALC_GRID_DIM(var_Y.totalSize, THREADS_PER_BLOCK);
    changeGPrune<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(var_Y.partition, var_Y.totalSize, device_G_prune);
    while (generator.hasMoreCombinations()) {
        auto one_batch    = generator.getNextBatch_continuousMemory(batch_size);
        currentIteration++;
        fmt::print("batch_nums: {}, currentIteration: {}", batch_nums, currentIteration);
        cout << endl;
        size_t useful_batch = one_batch.size() / k;
        CUDA_ERROR_CHECK(cudaMemcpy(
            combinations, one_batch.data(), sizeof(uint) * k * useful_batch, cudaMemcpyHostToDevice));
        CheckMaximality_for_special_optimized(device_degrees, device_neighbors, device_neighbors_offset, device_G_prune,
            combinations, useful_batch, g_state_viewer, conY_curY_hashTable, k, X, con_Y,
            var_Y, ext_p, ext_q, buffer, right_length, theta, k, stream);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        if (outputFormat.empty()) {
            uint *host_X = new uint[X.totalSize], *host_conY = new uint[con_Y.totalSize];
            host_varY       = new uint[var_Y.totalSize];
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                host_X, X.partition, sizeof(uint) * X.totalSize, cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(host_conY, con_Y.partition, sizeof(uint) * con_Y.totalSize,
                cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(host_varY, var_Y.partition, sizeof(uint) * var_Y.totalSize,
                cudaMemcpyDeviceToHost, stream));
            outputFormat.assign(host_X, X.totalSize, host_conY, con_Y.totalSize, nullptr, 0);
            delete[] host_X, host_conY;
        }
        ProcessBatchOptimized(buffer.h_isMax, useful_batch,
            num, one_batch, k, host_varY, var_Y.totalSize,
            outputFormat.static_part, time_records, start_time, d_res_count);
        if (d_res_count >= num) return;
    }
    if (host_varY != nullptr) {
        delete[] host_varY;
    }
    CUDA_ERROR_CHECK(cudaFreeAsync(combinations, stream));
}

#endif 
