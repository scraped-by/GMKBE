














#ifndef IMB_GPU_INCLUDE_UTILS_H
#define IMB_GPU_INCLUDE_UTILS_H
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <iostream>
using namespace std;

uint d_res_count   = 0;
uint num           = 100000; 
bool OutputResults = true;
int gpu_id;


using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
std::set<int> checkpoints;
std::vector<std::pair<int, double>> time_records;
TimePoint start_time;

#define _1GB_intNum  1073741824
#define _1MB_intNum  262144
#define MAX_GRID_DIM_X INT32_MAX
#define MAX_GRID_DIM_Y 65535
#define MAX_GRID_DIM_Z 65535
#define CUDA_ERROR_CHECK(call)                                                                                      \
    {                                                                                                               \
        const cudaError_t error = call;                                                                             \
        if (error != cudaSuccess) {                                                                                 \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                                                                \
        }                                                                                                           \
    }

#define CUDA_ERROR_CHECK_DRV_API(call)                                               \
    {                                                                                \
        const CUresult error = call;                                                 \
        if (error != CUDA_SUCCESS) {                                                 \
            char* error_str = new char[1024];                                        \
            cuGetErrorString(error, (const char**) &error_str);                      \
            printf("[%s:%d] %s Error :%s!\n", __FILE__, __LINE__, #call, error_str); \
            exit(1);                                                                 \
        }                                                                            \
    }

#define ERROR_CALL(msg)                                                                   \
    {                                                                                     \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl; \
        exit(1);                                                                          \
    }


template <typename T>
void inline moveDataToDevice(T* hostData, T*& deviceData, size_t size) {
    CUDA_ERROR_CHECK(cudaMalloc(&deviceData, size * sizeof(T)));
    CUDA_ERROR_CHECK(cudaMemcpy(deviceData, hostData, size * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
}
void inline moveDataToDevice2(uint32_t* hostData, uint32_t*& deviceData, size_t size) {
    CUDA_ERROR_CHECK(cudaMalloc(&deviceData, size * sizeof(uint32_t)));
    CUDA_ERROR_CHECK(cudaMemcpy(deviceData, hostData, size * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
}

template<typename T>
inline T next_power_of_2(T n) { 
    if (n == 0 || n == 1) {
        return 1;
    }
    if ((n & (n - 1)) == 0) {
        return n;
    }
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

class KLoopGenerator {
private:
    int k;
    int length;
    std::vector<int> current;
    bool hasNext;

public:
    
    KLoopGenerator(int k, int length) : k(k), length(length), current(k), hasNext(true) {
        if (k <= 0 || length <= 0 || k > length) {
            hasNext = false;
            return;
        }

        
        for (int i = 0; i < k; i++) {
            current[i] = i;
        }
    }

    
    std::vector<std::vector<int>> getNextBatch(int n) {
        std::vector<std::vector<int>> result;

        if (!hasNext || n <= 0) {
            return result;
        }

        for (int count = 0; count < n && hasNext; count++) {
            
            result.push_back(current);

            
            if (!generateNext()) {
                hasNext = false;
                break;
            }
        }

        return result;
    }
    std::vector<uint> getNextBatch_continuousMemory(uint64_t n) {
        std::vector<uint> result;
        result.reserve(static_cast<uint64_t>(k * n));
        if (!hasNext || n <= 0) {
            return result;
        }

        for (uint64_t count = 0; count < n && hasNext; count++) {
            fmt::print("count: {} , n: {}", count, n);
            cout << endl;
            
            result.insert(result.end(), current.begin(), current.end());

            
            if (!generateNext()) {
                hasNext = false;
                break;
            }
        }

        return result;
    }

    
    bool hasMoreCombinations() const {
        return hasNext;
    }

    
    void reset() {
        if (k <= 0 || length <= 0 || k > length) {
            hasNext = false;
            return;
        }

        hasNext = true;
        for (int i = 0; i < k; i++) {
            current[i] = i;
        }
    }

    
    std::vector<int> getCurrentState() const {
        return current;
    }

    static tuple<uint64_t, bool> combination_total_num(uint k, uint length) {
        if (k > length || k < 0) {
            return {0, false};
        }
        if (k == 0 || k == length) {
            return {1, true};
        }

        
        if (k > length - k) {
            k = length - k;
        }

        uint64_t result = 1;

        for (int i = 0; i < k; ++i) {
            
            if (result > UINT64_MAX / (length - i)) {
                std::cerr << "Warning: Combination result may overflow!" << std::endl;
                return {-1, false};
            }

            result = result * (length - i) / (i + 1);
        }

        return {result, true};
    }

private:
    
    bool generateNext() {
        
        int pos = k - 1;

        
        while (pos >= 0 && current[pos] == length - k + pos) {
            pos--;
        }

        
        if (pos < 0) {
            return false;
        }

        
        current[pos]++;

        
        for (int i = pos + 1; i < k; i++) {
            current[i] = current[i - 1] + 1;
        }

        return true;
    }
};



#include <vector>
#include <algorithm>
#include <omp.h>

typedef uint64_t uint64;

class KLoopGenerator_v2 {
private:
    int k;
    int n_elements;
    uint64_t current_global_idx;
    uint64_t total_combinations;
    bool hasNext;

    
    const uint64_t OMP_THRESHOLD = 50000;

    
    std::vector<std::vector<uint64_t>> nCr_table;

public:
    KLoopGenerator_v2(int k, int length)
        : k(k), n_elements(length), current_global_idx(0), hasNext(true) {

        
        initNCRTable();

        
        if (k > length || k < 0) total_combinations = 0;
        else total_combinations = getNCR(length, k);

        if (total_combinations == 0) hasNext = false;
    }

    static tuple<uint64_t, bool> combination_total_num(uint k, uint length) {
        if (k > length || k < 0) {
            return {0, false};
        }
        if (k == 0 || k == length) {
            return {1, true};
        }

        
        if (k > length - k) {
            k = length - k;
        }

        uint64_t result = 1;

        for (int i = 0; i < k; ++i) {
            
            if (result > UINT64_MAX / (length - i)) {
                std::cerr << "Warning: Combination result may overflow!" << std::endl;
                return {-1, false};
            }

            result = result * (length - i) / (i + 1);
        }

        return {result, true};
    }

    
    
    
    std::vector<uint> getNextBatch_continuousMemory(uint64_t batch_size) {
        std::vector<uint> result;

        if (!hasNext || batch_size == 0) return result;

        
        uint64_t count = batch_size;
        if (current_global_idx + count > total_combinations) {
            count = total_combinations - current_global_idx;
        }

        
        
        result.resize(count * k);

        
        if (count < OMP_THRESHOLD) {
            
            generateSerial(result.data(), count, current_global_idx);
        } else {
            
            generateParallel(result.data(), count, current_global_idx);
        }

        
        current_global_idx += count;
        if (current_global_idx >= total_combinations) {
            hasNext = false;
        }

        return result;
    }

    bool hasMoreCombinations() const { return hasNext; }

private:
    
    void generateSerial(uint* buffer, uint64_t count, uint64_t start_rank) {
        std::vector<int> temp_current(k);
        
        getCombinationAtRank(start_rank, temp_current);

        uint* ptr = buffer;
        for (uint64_t i = 0; i < count; ++i) {
            
            for (int j = 0; j < k; ++j) {
                ptr[j] = (uint)temp_current[j];
            }
            ptr += k;
            
            if (i < count - 1) generateNextLocal(temp_current);
        }
    }

    
    void generateParallel(uint* buffer, uint64_t count, uint64_t start_rank) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();

            
            uint64_t items_per_thread = count / nthreads;
            uint64_t remainder = count % nthreads;

            
            uint64_t local_start = tid * items_per_thread + std::min((uint64_t)tid, remainder);
            uint64_t local_count = items_per_thread + (tid < remainder ? 1 : 0);
            uint64_t local_end = local_start + local_count;

            if (local_count > 0) {
                
                std::vector<int> local_comb(k);
                getCombinationAtRank(start_rank + local_start, local_comb);

                
                uint* ptr = buffer + local_start * k;

                
                for (uint64_t i = 0; i < local_count; ++i) {
                    for (int j = 0; j < k; ++j) {
                        ptr[j] = (uint)local_comb[j];
                    }
                    ptr += k;
                    
                    if (i < local_count - 1) {
                        generateNextLocal(local_comb);
                    }
                }
            }
        }
    }

    
    void generateNextLocal(std::vector<int>& current) {
        int pos = k - 1;
        while (pos >= 0 && current[pos] == n_elements - k + pos) {
            pos--;
        }
        if (pos >= 0) {
            current[pos]++;
            for (int i = pos + 1; i < k; i++) {
                current[i] = current[i - 1] + 1;
            }
        }
    }

    
    
    void getCombinationAtRank(uint64_t rank, std::vector<int>& out_comb) {
        int current_val = -1;
        
        

        for (int i = 0; i < k; ++i) {
            
            
            int start_val = current_val + 1;

            
            for (int v = start_val; v <= n_elements - (k - i); ++v) {
                
                
                uint64_t count = getNCR(n_elements - 1 - v, k - 1 - i);

                if (rank < count) {
                    
                    out_comb[i] = v;
                    current_val = v;
                    break; 
                } else {
                    
                    rank -= count;
                }
            }
        }
    }

    
    void initNCRTable() {
        
        int max_n = n_elements;
        if (max_n > 2000) max_n = 2000; 

        nCr_table.resize(max_n + 1, std::vector<uint64_t>(k + 1, 0));
        for (int i = 0; i <= max_n; i++) {
            nCr_table[i][0] = 1;
            if (i <= k) nCr_table[i][i] = 1;
            for (int j = 1; j < i && j <= k; j++) {
                nCr_table[i][j] = nCr_table[i - 1][j - 1] + nCr_table[i - 1][j];
            }
        }
    }

    
    uint64_t getNCR(int n, int r) const {
        if (r < 0 || r > n) return 0;
        if (n < nCr_table.size()) return nCr_table[n][r];
        
        
        uint64_t res = 1;
        for(int i=1; i<=r; ++i) {
            res = res * (n - i + 1) / i;
        }
        return res;
    }
};

inline void getPartition(vector<uint>& partition_, vector<uint>& originalData, uint* degree) {
    for (uint i = 0;;) {
        while (i < originalData.size() && degree[originalData[i]] < 32) {
            i++;
        }
        partition_[0] = i;
        if (i == originalData.size()) {
            break;
        }
        while (i < originalData.size() && degree[originalData[i]] < 1024) {
            i++;
        }
        partition_[1] = i - partition_[0];
        if (i == originalData.size()) {
            break;
        }
        while (i < originalData.size()) {
            assert(degree[originalData[i++]] < 131072);
        }
        partition_[2] = originalData.size() - partition_[1] - partition_[0];
        break;
    }
}











inline void getPartition_v3(std::vector<uint>& partition_sizes, const std::vector<uint>& sorted_data,
    const uint* degree, uint max_partition_num) {

    if (sorted_data.empty() || max_partition_num == 0) {
        return; 
    }

    uint data_idx         = 0; 
    uint last_split_point = 0; 

    
    for (uint j = 0; j < max_partition_num - 1; ++j) {
        
        
        uint64_t threshold = 1;
        for (int k = 0; k < j + 1; ++k) {
            threshold *= 32;
        }

        
        
        while (data_idx < sorted_data.size() && (uint64_t) degree[sorted_data[data_idx]] < threshold) {
            data_idx++;
        }

        
        partition_sizes[j] = data_idx - last_split_point;

        
        last_split_point = data_idx;

        
        if (data_idx == sorted_data.size()) {
            
            break;
        }
    }

    
    
    if (last_split_point < sorted_data.size()) {
        partition_sizes[max_partition_num - 1] = sorted_data.size() - last_split_point;
    }
}

#define CALC_GRID_DIM(total_threads, block_dim) \
    max(static_cast<unsigned int>(std::min( \
        (static_cast<uint64_t>(total_threads) + (block_dim) - 1) / (block_dim), \
        static_cast<uint64_t>(MAX_GRID_DIM_X) \
    )), 1)



#endif 
