













#ifndef IMB_GPU2_INCLUDE_GPU_UTILS_CUH
#define IMB_GPU2_INCLUDE_GPU_UTILS_CUH

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <utils.h>
namespace cg = cooperative_groups;
#define MAX_BLOCK_SIZE 1024
#define WARP_SIZE      32
#define FULL_MASK      0xffffffff
#define MAX_UINT       0xFFFFFFFFU
#define MIN_UINT       0x00000000U
#define THREADS_PER_BLOCK 256
#define DEVICE_        __device__ __forceinline__


struct edgeList {
    int degree;
    int* neighbors;
};
enum class CompareOp { Greater, GreaterEqual, Less, LessEqual, Equal, NotEqual, COUNT };






namespace Utils {
    __host__ DEVICE_ bool get_judge(int val, int threshold, CompareOp op) {
        switch (op) {
        case CompareOp::Greater:
            return (val > threshold);
        case CompareOp::GreaterEqual:
            return (val >= threshold);
        case CompareOp::Less:
            return (val < threshold);
        case CompareOp::LessEqual:
            return (val <= threshold);
        case CompareOp::Equal:
            return (val == threshold);
        case CompareOp::NotEqual:
            return (val != threshold);
        }
    }
    template <typename T>
    __global__ void touch_elements_kernel(T* ptr, size_t count) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            
            
            volatile T temp = ptr[idx];
        }
    }

    class L2PersistenceManager {
private:
    struct MemoryRegion {
        void* ptr;
        size_t size;
    };

    std::vector<MemoryRegion> regions;
    size_t totalSizeInBytes = 0;
    int deviceId = 0;
    cudaDeviceProp prop;
    bool isApplied = false; 

public:
    L2PersistenceManager(int deviceId): deviceId(deviceId) {
        CUDA_ERROR_CHECK(cudaGetDevice(&deviceId));
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    }

    
    
    ~L2PersistenceManager() {
        
        if (isApplied) {
            
            
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
        }
    }

    
    template <typename T>
    void addPointer(T* d_ptr, size_t length) {
        size_t size = length * sizeof(T);
        regions.push_back({reinterpret_cast<void*>(d_ptr), size});
        totalSizeInBytes += size;
    }

    
    bool apply(cudaStream_t stream = nullptr, bool prefetch = true) {
        if (prop.major < 8) {
            
            return true;
        }

        size_t l2CacheSize = prop.l2CacheSize;
        size_t threshold = static_cast<size_t>(l2CacheSize * 0.70);

        if (totalSizeInBytes > threshold) {
            std::cerr << "[Error] Total requested size exceeds 70% of L2 Cache!" << std::endl;
            return false;
        }

        int percent = static_cast<int>((totalSizeInBytes * 100) / l2CacheSize) + 2;
        if (percent > 75) percent = 75;
        if (percent < 0) percent = 0;

        CUDA_ERROR_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2CacheSize * percent / 100));

        if (regions.empty()) return true;

        uintptr_t minAddr = reinterpret_cast<uintptr_t>(regions[0].ptr);
        uintptr_t maxAddr = minAddr + regions[0].size;

        for (const auto& region : regions) {
            uintptr_t start = reinterpret_cast<uintptr_t>(region.ptr);
            uintptr_t end = start + region.size;
            if (start < minAddr) minAddr = start;
            if (end > maxAddr) maxAddr = end;
        }

        size_t windowSize = maxAddr - minAddr;

        cudaAccessPolicyWindow policyWindow;
        policyWindow.base_ptr = reinterpret_cast<void*>(minAddr);
        policyWindow.num_bytes = windowSize;
        policyWindow.hitRatio = 1.0f;
        policyWindow.hitProp = cudaAccessPropertyPersisting;
        policyWindow.missProp = cudaAccessPropertyStreaming;
        
        cudaStreamAttrValue attr;
        attr.accessPolicyWindow = policyWindow;

        CUDA_ERROR_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
        
        if (prefetch) {
            for (const auto& region : regions) {
                
                
                int blockSize = 256;
                int numBlocks = (region.size + blockSize - 1) / blockSize;
                
                touch_elements_kernel<char><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<char*>(region.ptr), region.size
                );
            }
        }

        isApplied = true;
        return true;
    }

    
    
    void reset(cudaStream_t stream = 0) {
        if (prop.major < 8) return;

        
        
        CUDA_ERROR_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));

        
        
        cudaAccessPolicyWindow policyWindow;
        policyWindow.base_ptr = reinterpret_cast<void*>(0); 
        policyWindow.num_bytes = 0;                         
        policyWindow.hitRatio = 0.0f;                       
        policyWindow.hitProp = cudaAccessPropertyNormal;    
        policyWindow.missProp = cudaAccessPropertyNormal;   
        cudaStreamAttrValue attr;
        attr.accessPolicyWindow = policyWindow;
        CUDA_ERROR_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));

        
        regions.clear();
        totalSizeInBytes = 0;
        isApplied = false;

        std::cout << "L2 Persistence Released (Reset to Normal)." << std::endl;
    }
};

    template <typename T>
    __device__ __forceinline__ T atomicAggInc(T* ctr) {
        auto g = cg::coalesced_threads();
        T warp_res;
        if (g.thread_rank() == 0) {
            warp_res = atomicAdd(ctr, g.size());
        }
        return g.shfl(warp_res, 0) + g.thread_rank();
    }
    template <typename T>
    __device__ __forceinline__ T atomicAggInc(T* ctr, T value) {
        auto g = cg::coalesced_threads();
        T warp_res;

        if (g.thread_rank() == 0) {
            warp_res = atomicAdd(ctr, g.size() * value);
        }
        T base_offset = g.shfl(warp_res, 0);
        return base_offset + g.thread_rank() * value;
    }

    
    template <typename T>
    __device__ __forceinline__ bool atomicWarpReduceCAS(T* target, T compare, T val) {
        auto g            = cg::coalesced_threads();
        bool warp_success = false;

        if (g.thread_rank() == 0) {
            warp_success = (atomicCAS(target, compare, val) == compare);
        }
        
        return g.shfl(warp_success, 0);
    }
    template <typename T>
    __device__ __forceinline__ T max_warpPrimitive(T val, const uint laneID, T* maxVal) {
        const T warpMax = __reduce_max_sync(FULL_MASK, val);
        if (laneID == 0) {
            atomicMax(maxVal, warpMax);
        }
        return warpMax;
    }
    __device__ __forceinline__ uint32_t and_warpPrimitive0(uint32_t predicate, const uint laneID, uint32_t* andResult) {
        uint32_t ans = __reduce_and_sync(FULL_MASK, predicate);
        if (laneID == 0) {
            atomicAnd(andResult, ans);
        }
        return ans;
    }

    template <typename T>
    __device__ __forceinline__ int binarySearch_returnIdx(
        T target, const T* sortedArray, const uint arrayLength) { 
        if (arrayLength == 0) {
            return -1; 
        }
        int left  = 0;
        int right = arrayLength - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2; 
            if (sortedArray[mid] == target) {
                return mid; 
            } else if (sortedArray[mid] < target) {
                left = mid + 1; 
            } else {
                right = mid - 1; 
            }
        }
        return -1; 
    }
    template <typename T>
    __device__ __forceinline__ bool binarySearch(
        T target, const T* sortedArray, const uint arrayLength) { 
        return binarySearch_returnIdx(target, sortedArray, arrayLength) != -1;
    }
    template <typename T>
    __device__ __forceinline__ int sequentialSearch_unrolled_returnIdx(const T target, const T* Unique_Array, const uint arrayLength) {
        int idx = -1;
        #pragma unroll
        for (int i = 0; i < arrayLength; ++i) {
            if (Unique_Array[i] == target) {
                idx = i;
                
                
                
            }
        }
        return idx;
    }
    template <typename T>
    __device__ __forceinline__ bool sequentialSearch_unrolled(const T target, const T* Unique_Array, const uint arrayLength) {
        return sequentialSearch_unrolled_returnIdx(target, Unique_Array, arrayLength) != -1;
    }


    template <typename T>
    __device__ __forceinline__ T min_warpPrimitive(T val, const uint laneID, T* minVal) {
        const T warpMin = __reduce_min_sync(FULL_MASK, val);
        if (laneID == 0) {
            atomicMin(minVal, warpMin);
        }
        return warpMin;
    }
    template <typename T>
    __device__ __forceinline__ T __reduce_warp_cub_sum(T val) {
        typedef cub::WarpReduce<T, 32> WarpReduce;
        __shared__ typename WarpReduce::TempStorage temp_storage;
        return WarpReduce(temp_storage).Sum(val);
    }
    template <typename T>
    __device__ __forceinline__ void sum_warpPrimitive(T val, const uint laneID, T* sumVal) {
        const T warpSum = __reduce_warp_cub_sum(val);
        if (laneID == 0) {
            atomicAdd(sumVal, warpSum);
        }
    }

    template <typename T>
    __device__ __forceinline__ void warpLevelCount(T* ctr) {
        const auto g = cg::coalesced_threads();
        if (g.thread_rank() == 0) {
            atomicAdd(ctr, g.size());
        }
    }
    template <typename T>
    __device__ __forceinline__ T atomicAggDec(T* ctr) {
        auto g = cg::coalesced_threads();
        T warp_res;
        if (g.thread_rank() == 0) {
            warp_res = atomicSub(ctr, g.size()); 
        }
        return g.shfl(warp_res, 0) - g.thread_rank();
    }

    __global__ void any_satisfy_v6_coop_uint(const uint* data, const uint size, const int threshold, const CompareOp op,
        uint* result, volatile int* found_flag) {
        const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
        const uint stride = blockDim.x * gridDim.x;

        for (uint i = tid; i < size && *found_flag == 0; i += stride) {
            if (Utils::get_judge(static_cast<int>(data[i]), threshold, op)) {
                atomicExch((int*) found_flag, 1);
                *result = 1;
                break;
            }
        }
    }

    __global__ void any_satisfy_v6_coop_int(const int* data, const uint size, const int threshold, const CompareOp op,
        uint* result, volatile int* found_flag) {
        const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
        const uint stride = blockDim.x * gridDim.x;
        for (uint i = tid; i < size && *found_flag == 0; i += stride) {
            if (Utils::get_judge(data[i], threshold, op)) {
                atomicExch((int*) found_flag, 1);
                *result = 1;
                break;
            }
        }
    }

    template <typename T>
    inline bool any_satisfied(T* d_data, uint size, uint threshold, CompareOp op, cudaStream_t stream = nullptr) {
        static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned int>,
                  "any_satisfied function only accepts 'int' or 'unsigned int' as template argument T.");
        uint* d_result;
        int* d_found_flag;
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_result, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_found_flag, sizeof(int), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_result, 0, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_found_flag, 0, sizeof(int), stream));
        
        
        bool any_satisfied_flag;
        uint gridDim = CALC_GRID_DIM(size, THREADS_PER_BLOCK);
        if constexpr (std::is_same_v<T, unsigned int>)
            any_satisfy_v6_coop_uint<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(d_data, size, threshold, op, d_result, d_found_flag);
        else
            any_satisfy_v6_coop_int<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(d_data, size, threshold, op, d_result, d_found_flag);
        
        
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&any_satisfied_flag, d_result, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_result, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_found_flag, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        return any_satisfied_flag;
    }

    __global__ void multi_any_satisfied(const uint* d_data, uint* d_result, int L, int threshold, CompareOp op) {
        const uint tid        = threadIdx.x;
        uint group_id   = blockIdx.x;
        uint block_size = blockDim.x;
        __shared__ uint local_value;
        if (tid == 0) {
            local_value = 0;
        }
        __syncthreads();
        for (uint i = tid; i < L; i += block_size) {
            if (Utils::get_judge(static_cast<int>(d_data[group_id * L + i]), threshold, op)) {
                local_value = 1;
            }
        }
        __syncthreads();
        if (tid == 0) {
            d_result[group_id] = local_value;
        }
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    __global__ void multi_any_satisfied_vectorized(const uint* d_data, uint* d_result, uint64_t n, int L, int threshold, CompareOp op) {
        int group_id = blockIdx.x;
        if (group_id >= n) return;

        int tid = threadIdx.x;

        
        
        long long group_offset = (long long)group_id * L;

        __shared__ int shared_any;
        if (tid == 0) shared_any = 0;
        __syncthreads();

        
        
        
        long long vec_L = L / 4;

        
        
        for (long long i = tid; i < vec_L; i += blockDim.x) {
            
            long long base_idx = group_offset + i * 4;

            
            
            uint4 data4;
            data4.x = d_data[base_idx + 0];
            data4.y = d_data[base_idx + 1];
            data4.z = d_data[base_idx + 2];
            data4.w = d_data[base_idx + 3];

            if (atomicOr(&shared_any, 0) == 1) break; 

            if (Utils::get_judge(static_cast<int>(data4.x), threshold, op) ||
                Utils::get_judge(static_cast<int>(data4.y), threshold, op) ||
                Utils::get_judge(static_cast<int>(data4.z), threshold, op) ||
                Utils::get_judge(static_cast<int>(data4.w), threshold, op)) {
                atomicOr(&shared_any, 1);
                
                
            }
        }

        
        long long remaining_start = vec_L * 4;
        for (long long i = remaining_start + tid; i < L; i += blockDim.x) {
            if (atomicOr(&shared_any, 0) == 1) break; 

            
            if (Utils::get_judge(d_data[group_offset + i], threshold, op)) {
                atomicOr(&shared_any, 1);
            }
        }

        __syncthreads();

        if (tid == 0) {
            d_result[group_id] = shared_any;
        }
    }

    inline void cpu_any_reference(const uint *data, uint* &result, const int n, const int L, const int threshold, const CompareOp op) {
        for (int i = 0; i < n; i++) {
            result[i] = 0;
            for (int j = 0; j < L; j++) {
                if (get_judge(static_cast<int>(data[i * L + j]), threshold, op)) {
                    result[i] = 1;
                    break;
                }
            }
        }
    }
    inline tuple<uint*, bool> multi_group_any_satisified( 
        uint* d_data, uint64_t n, uint L, int threshold, CompareOp op, cudaStream_t stream = nullptr) {
        uint64_t total_size = n * L;
        if (total_size < 32768) {
            uint* h_result = new uint[n];
            uint* h_data = new uint[total_size];
            CUDA_ERROR_CHECK(cudaMemcpyAsync(h_data, d_data, total_size * sizeof(uint), cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            cpu_any_reference(h_data, h_result, n, L, threshold, op);
            return {h_result, true};
        }
        uint *d_result;
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_result, n * sizeof(uint), stream));
        uint blockSize = min(next_power_of_2(L), 512);
        uint64_t grid = min(n, static_cast<uint64_t>(MAX_GRID_DIM_X));
        if (total_size < 2097152) {
            multi_any_satisfied <<<grid, blockSize, 0, stream>>>(d_data, d_result, L, threshold, op);
        }
        else
            multi_any_satisfied_vectorized <<<grid, blockSize, 0, stream>>>(d_data, d_result, n, L, threshold, op);
        return {d_result, false};
    }

    __global__ void initialize_with_anyValue(uint* __restrict__ datas, uint length, uint value) {
        const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
        uint stride = blockDim.x * gridDim.x;
        for (uint i = tid; i < length; i += stride) {
            datas[i] = value;
        }
    }

    __global__ void initialize_with_anyValue(int* __restrict__ datas, uint length, int value) {
        const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
        uint stride = blockDim.x * gridDim.x;
        for (uint i = tid; i < length; i += stride) {
            datas[i] = value;
        }
    }

    __global__ void device_sum(const uint* data, uint* result, const uint size) {
        const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
        const uint laneID = tid % 32;
        auto _sum         = __reduce_warp_cub_sum(tid < size ? data[tid] : 0);
        if (laneID == 0) {
            atomicAdd(result, _sum);
        }
    }
    __global__ void device_sum(const uint* data, unsigned long long int* result, const size_t size) {
        const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
        const uint laneID = tid % 32;
        auto _sum         = __reduce_warp_cub_sum(tid < size ? data[tid] : 0);
        if (laneID == 0) {
            atomicAdd(result, _sum);
        }
    }
    template <typename KernelFunc>
    void calculateOptimalBlockSize(KernelFunc kernel, int dynamicSMemSize = 0) {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynamicSMemSize, 0);
        printf("Suggested block size: %d\n", blockSize);
        printf("Minimum grid size for max occupancy: %d\n", minGridSize);
    }


    __global__ void count_if_std_atomicAdd(int* data, size_t data_length, int if_target, uint* count, CompareOp op) {
        const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
        const uint grid_size = blockDim.x * gridDim.x;
        uint local_count = 0;
        for (size_t idx = tid; idx < data_length; idx += grid_size) {
            if (get_judge(data[idx], if_target, op)) {
                local_count ++;
            }
        }
        atomicAdd(count, local_count);
    }

    __global__ void count_if_std_atomicAgg(int* data, size_t data_length, int if_target, uint* count, CompareOp op) {
        const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
        const uint grid_size = blockDim.x * gridDim.x;
        for (size_t idx = tid; idx < data_length; idx += grid_size) {
            if (get_judge(data[idx], if_target, op)) {
                atomicAggInc(count);
            }
        }
    }
    __inline__ __device__ uint warpReduceSum(uint val) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        return val;
    }

    __device__ void blockReduceAndAtomicAdd(uint local_val, uint* global_counter) {
        uint sum = warpReduceSum(local_val);
        static __shared__ uint shared[32];

        int lane = threadIdx.x % warpSize; 
        int wid = threadIdx.x / warpSize;  
        if (lane == 0) {
            shared[wid] = sum;
        }
        __syncthreads();
        int num_warps = (blockDim.x + 31) / 32;

        sum = 0;
        if (wid == 0) {
            if (lane < num_warps) {
                sum = shared[lane];
            }
            sum = warpReduceSum(sum);
        }

        if (threadIdx.x == 0) {
            atomicAdd(global_counter, sum);
        }
    }
    __global__ void count_if_ILP_blockReduce(int* data, size_t data_length, int if_target, uint* count, CompareOp op) {
        const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
        const uint grid_size = blockDim.x * gridDim.x;
        const int ILP = 4;

        uint local_count = 0;
        size_t idx = tid;
        for (; idx + grid_size * (ILP - 1) < data_length; idx += grid_size * ILP) {
#pragma unroll
            for (int i = 0; i < ILP; i++) {
                if (get_judge(data[idx + i * grid_size], if_target, op)) {
                    local_count++;
                }
            }
        }
        for (; idx < data_length; idx += grid_size) {
            if (get_judge(data[idx], if_target, op)) {
                local_count++;
            }
        }
        blockReduceAndAtomicAdd(local_count, count);
    }

    __global__ void count_if_ILP_sum_warpPrimitive(int* data, size_t data_length, int if_target, uint* count, CompareOp op) {
        const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
        const uint grid_size = blockDim.x * gridDim.x;
        const uint laneID = threadIdx.x % 32;
        const int ILP = 4;

        uint local_count = 0;
        size_t idx = tid;

        for (; idx + grid_size * (ILP - 1) < data_length; idx += grid_size * ILP) {
#pragma unroll
            for (int i = 0; i < ILP; i++) {
                if (get_judge(data[idx + i * grid_size], if_target, op)) {
                    local_count++;
                }
            }
        }
        for (; idx < data_length; idx += grid_size) {
            if (get_judge(data[idx], if_target, op)) {
                local_count++;
            }
        }
        sum_warpPrimitive(local_count, laneID, count);
    }

    __host__ void count_if_smart(int* data, size_t data_length, int if_target, uint* count, CompareOp op) {
        
        
        const size_t THRESHOLD_USE_AGG = 5000;
        const size_t THRESHOLD_USE_blockReduce = 50000000;
        uint blockSize = 256;
        uint gridSize = (data_length + blockSize - 1) / blockSize;
        if (gridSize > 2048) gridSize = 2048;
        if (data_length < THRESHOLD_USE_AGG) {
            count_if_std_atomicAdd<<<gridSize, blockSize>>>(data, data_length, if_target, count, op);
        } else if (data_length < THRESHOLD_USE_blockReduce) {
            count_if_ILP_sum_warpPrimitive<<<gridSize, blockSize>>>(data, data_length, if_target, count, op);
        }
        else {
            count_if_ILP_blockReduce<<<gridSize, blockSize>>>(data, data_length, if_target, count, op);
        }
    }

    enum class ElementwiseOp : int {
        AND,
        OR,
        NOT,
        XOR,
        NAND,
        NOR
    };

    template<typename T, ElementwiseOp Op>
    __device__ __forceinline__ T elementwise_binary_op_compile_time(T a, T b) {
        if constexpr (Op == ElementwiseOp::AND) {
            return a & b;
        } else if constexpr (Op == ElementwiseOp::OR) {
            return a | b;
        } else if constexpr (Op == ElementwiseOp::NOT) {
            return static_cast<T>(a == 0 ? 1 : 0);
        } else if constexpr (Op == ElementwiseOp::XOR) {
            return a ^ b;
        } else if constexpr (Op == ElementwiseOp::NAND) {
            return static_cast<T>((a & b) == 0 ? 1 : 0);
        } else if constexpr (Op == ElementwiseOp::NOR) {
            return static_cast<T>((a | b) == 0 ? 1 : 0);
        }
        return a;
    }

    template<typename T, ElementwiseOp Op>
    __global__ void transform_kernel_compile_time(
        const T* __restrict__ input1,
        const T* __restrict__ input2,
        T* __restrict__ output,
        size_t size
    ) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            T b = (Op == ElementwiseOp::NOT) ? T(0) : input2[idx];
            output[idx] = elementwise_binary_op_compile_time<T, Op>(input1[idx], b);
        }
    }

    template<typename T>
    void transform_compile_time(
        const T* d_input1,
        const T* d_input2,
        T* d_output,
        size_t size,
        ElementwiseOp op,
        cudaStream_t stream = 0
    ) {
        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        switch (op) {
        case ElementwiseOp::AND:
            transform_kernel_compile_time<T, ElementwiseOp::AND>
                <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input1, d_input2, d_output, size);
            break;
        case ElementwiseOp::OR:
            transform_kernel_compile_time<T, ElementwiseOp::OR>
                <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input1, d_input2, d_output, size);
            break;
        case ElementwiseOp::NOT:
            transform_kernel_compile_time<T, ElementwiseOp::NOT>
                <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input1, d_input2, d_output, size);
            break;
        case ElementwiseOp::XOR:
            transform_kernel_compile_time<T, ElementwiseOp::XOR>
                <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input1, d_input2, d_output, size);
            break;
        case ElementwiseOp::NAND:
            transform_kernel_compile_time<T, ElementwiseOp::NAND>
                <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input1, d_input2, d_output, size);
            break;
        case ElementwiseOp::NOR:
            transform_kernel_compile_time<T, ElementwiseOp::NOR>
                <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input1, d_input2, d_output, size);
            break;
        }
    }
} 

namespace global_graph_parameter {
    int *G_temp = nullptr;
}



#endif 
