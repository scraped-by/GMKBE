



#ifndef FOR_CHECKMAXIMALITY_HASHTABLEHASHTABLE_H
#define FOR_CHECKMAXIMALITY_HASHTABLEHASHTABLE_H
#include "for_test.h"
#include "gpu_utils.cuh"
#include <hashTable.cuh>
#include <output.cuh>
#include <partition.cuh>
#include <utils.h>
#include <algorithm>
#include <functional>

__global__ void enumerate_Y_get_earlyStop2(const uint* X, const uint X_length, const uint* y, const uint y_length,
    const uint* degrees, uint* neighbors, const uint* neighbors_offset, const uint* selected_count, const int threshold,
    const uint state_id, uint* result,
    int* G_label,
    HashTable_viewer g_temp_viewer,
    Utils::MultiHash::cudaMultiHashTable_gpu_viewer delta_G_temps, const bool* selected_allStates) {
    if (result[0] == 1)return;
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < X_length * y_length; idx += grid_size) {
        uint x_id = idx % X_length;
        uint y_id = idx / X_length;
        if (selected_allStates != nullptr && selected_allStates[state_id * y_length + y_id]) {
            continue;
        }
        uint y_value       = y[y_id];
        uint* y_neighbours = neighbors + neighbors_offset[y_value];
        uint y_degree      = degrees[y_value];
        uint temp_num      = threshold + selected_count[state_id];
        uint x_value = X[x_id];
        uint value   = 0;
        if (G_label[x_value] && Utils::binarySearch(x_value, y_neighbours, y_degree)) {
            value += 1;
        }
        auto delta        = delta_G_temps.get_value(x_value, state_id);
        uint g_temp_value = delta + g_temp_viewer.get_value(x_value);
        if (result[0] == 0 && g_temp_value + value < temp_num) {
            atomicCAS(result, 0, 1);
        }
    }
}



__global__ void any_satisfied_intersection_size_double_results(uint* intersection, uint length, int _threshold,
    uint state_id, uint* selected_count, CompareOp op, uint* result, int* found_flag, int _threshold2,
    uint* result2, CompareOp op2) {
    const uint tid       = threadIdx.x + blockIdx.x * blockDim.x;
    const uint stride    = blockDim.x * gridDim.x;
    const int threshold  = _threshold + static_cast<int>(selected_count[state_id]);
    const int threshold2 = _threshold2 + static_cast<int>(selected_count[state_id]);
    for (uint i = tid; i < length && *found_flag == 0; i += stride) {
        if (Utils::get_judge(static_cast<int>(intersection[i]), threshold, op)) {
            atomicExch((int*) found_flag, 1);
            *result = 1;
            continue;
        }
        if (Utils::get_judge(static_cast<int>(intersection[i]), threshold2, op2)) {
            *result2 = 1;
        }
    }
}


__global__ void enumerateY_to_get_judge_difference_set(const uint* degrees, uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint Y_length, const uint* ext_partitions,
    const uint ext_length, arrMapTable g_state_viewer, const int threshold,
    CompareOp op, uint* iterations, bool* selected_allStates, uint state_id, uint* selected_count, uint *intersection, const int threshold2, CompareOp op2) {
    
    
    
    

    
    
    
    
    
    
    
    
    
    

    if (ext_length == 0 || Y_length == 0) {
        return;
    }
    uint totalLength         = Y_length * ext_length;
    const uint tid                 = threadIdx.x + blockIdx.x * blockDim.x;
    uint grid_size           = blockDim.x * gridDim.x;
    const int count_ = static_cast<int>(selected_count[state_id]);
    const bool* selected_bar = selected_allStates == nullptr ? nullptr : selected_allStates + state_id * Y_length;
    for (uint idx = tid; idx < totalLength; idx += grid_size) {
        uint Y_idx = idx / ext_length; 
        if (selected_bar != nullptr && selected_bar[Y_idx] == false) {
            continue;
        }
        auto y_data       = Y_partitions[Y_idx];
        uint ext_iner_idx = idx % ext_length;
        if (Utils::get_judge(static_cast<int>(intersection[ext_iner_idx]), threshold2 + count_, op2)) {
            atomicAnd(&iterations[ext_iner_idx], 0);
            continue;
        }
        if (iterations[ext_iner_idx] == 0) {
            continue;
        }
        uint ext_node            = ext_partitions[ext_iner_idx];
        uint* search_range       = neighbors + neighbors_offset[ext_node];
        uint search_range_length = degrees[ext_node];
        int search_idx           = Utils::binarySearch_returnIdx(y_data, search_range, search_range_length);
        if (search_idx == -1) {
            
            int value = static_cast<int>(g_state_viewer.get_value(y_data));
            if (Utils::get_judge(value, threshold, op)) {
                atomicAnd(&iterations[ext_iner_idx], 0);
            }
        }
    }
}

__global__ void enumerateY_to_get_judge_difference_set(const uint* degrees, uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint Y_length, const uint* ext_partitions,
    const uint ext_length, arrMapTable g_state_viewer, const int threshold,
    CompareOp op, uint* iterations, uint *intersection, const int threshold2, CompareOp op2) {
    
    
    
    

    
    
    
    
    
    
    
    
    
    

    if (ext_length == 0 || Y_length == 0) {
        return;
    }
    uint totalLength         = Y_length * ext_length;
    const uint tid                 = threadIdx.x + blockIdx.x * blockDim.x;
    uint grid_size           = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < totalLength; idx += grid_size) {
        uint Y_idx = idx / ext_length; 
        auto y_data       = Y_partitions[Y_idx];
        uint ext_iner_idx = idx % ext_length;
        if (Utils::get_judge(static_cast<int>(intersection[ext_iner_idx]), threshold2, op2)) {
            atomicAnd(&iterations[ext_iner_idx], 0);
            continue;
        }
        if (iterations[ext_iner_idx] == 0) {
            continue;
        }
        uint ext_node            = ext_partitions[ext_iner_idx];
        uint* search_range       = neighbors + neighbors_offset[ext_node];
        uint search_range_length = degrees[ext_node];
        int search_idx           = Utils::binarySearch_returnIdx(y_data, search_range, search_range_length);
        if (search_idx != -1) {
            continue; 
        }
        int value = static_cast<int>(g_state_viewer.get_value(y_data));
        if (Utils::get_judge(value, threshold, op)) {
            atomicAnd(&iterations[ext_iner_idx], 0);
        }
    }
}

__global__ void enumerateY_to_get_judge_difference_set2(const uint* degrees, const uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint Y_length, const uint* ext_partitions,
    const uint ext_length, arrMapTable g_state_viewer, const int threshold,
    const CompareOp op, uint* iterations, const uint *intersection, const int threshold2, const CompareOp op2) {
    cg::thread_block block             = cg::this_thread_block();
    cg::thread_block_tile<32> group = cg::tiled_partition<32>(block);
    constexpr uint group_size              = group.size();
    const uint tid                     = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size               = blockDim.x * gridDim.x;
    const uint group_id                = tid / group_size;
    const uint inner_id                = group.thread_rank();
    const uint group_num               = grid_size / group_size;
    for (uint ext_id = group_id; ext_id < ext_length; ext_id += group_num) {
        if (Utils::get_judge(static_cast<int>(intersection[ext_id]), threshold2, op2)) {
            if (group_id == 0) iterations[ext_id] = 0;
            continue;
        }
        for (uint y_idx = inner_id; y_idx < Y_length; y_idx += group_size) {
            const uint y_data              = Y_partitions[y_idx];
            const uint ext_node            = ext_partitions[ext_id];
            const uint* search_range       = neighbors + neighbors_offset[ext_node];
            const uint search_range_length = degrees[ext_node];
            bool flag = !Utils::binarySearch(y_data, search_range, search_range_length);
            int value = static_cast<int>(g_state_viewer.get_value(y_data));
            if (flag && Utils::get_judge(value, threshold, op)) {
                atomicExch(&iterations[ext_node], 0);
            }
        }
    }
}

__global__ void changeGPrune(
    uint *ext_p, uint ext_p_length, uint* intersection, uint64_t intersection_length, int _threshold, bool* device_G_prune, CompareOp op, const uint* is_Max = nullptr) {
    const uint64_t tid      = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t stride   = blockDim.x * gridDim.x;
    for (uint64_t i = tid; i < intersection_length; i += stride) {
        if (is_Max != nullptr) {
            const uint state_id = i / ext_p_length;
            if (is_Max[state_id] == 0) continue;
        }
        const uint ext_p_idx = i % ext_p_length;
        const uint node      = ext_p[ext_p_idx];
        if (device_G_prune[node]) continue;
        if (Utils::get_judge(static_cast<int>(intersection[i]), _threshold, op)) {
            device_G_prune[node] = true;
            
        }
    }
}

__global__ void changeGPrune_ILP(
    uint *ext_p,
    uint ext_p_length,
    uint* intersection,
    uint64_t intersection_length,
    int _threshold,
    bool* device_G_prune,
    CompareOp op,
    const uint* is_Max = nullptr)
{
    
    
    const uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t stride = blockDim.x * gridDim.x;

    
    
    
    const uint64_t vec_limit = intersection_length / 4;

    
    for (uint64_t i = tid; i < vec_limit; i += stride) {
        uint64_t base_idx = i * 4; 

        
        uint4 vals = reinterpret_cast<uint4*>(intersection)[i];

        
        
        uint ext_p_idx = base_idx % ext_p_length;
        uint state_id  = base_idx / ext_p_length;

        
        
        
        

        
        {
            bool process = true;
            if (is_Max != nullptr && is_Max[state_id] == 0) process = false;

            if (process) {
                const uint node = ext_p[ext_p_idx];
                
                if (!device_G_prune[node]) {
                    if (Utils::get_judge(static_cast<int>(vals.x), _threshold, op)) {
                        device_G_prune[node] = true;
                    }
                }
            }
        }

        
        
        ext_p_idx++;
        if (ext_p_idx == ext_p_length) { ext_p_idx = 0; state_id++; }

        {
            bool process = true;
            if (is_Max != nullptr && is_Max[state_id] == 0) process = false;

            if (process) {
                const uint node = ext_p[ext_p_idx];
                if (!device_G_prune[node]) {
                    if (Utils::get_judge(static_cast<int>(vals.y), _threshold, op)) {
                        device_G_prune[node] = true;
                    }
                }
            }
        }

        
        ext_p_idx++;
        if (ext_p_idx == ext_p_length) { ext_p_idx = 0; state_id++; }

        {
            bool process = true;
            if (is_Max != nullptr && is_Max[state_id] == 0) process = false;

            if (process) {
                const uint node = ext_p[ext_p_idx];
                if (!device_G_prune[node]) {
                    if (Utils::get_judge(static_cast<int>(vals.z), _threshold, op)) {
                        device_G_prune[node] = true;
                    }
                }
            }
        }

        
        ext_p_idx++;
        if (ext_p_idx == ext_p_length) { ext_p_idx = 0; state_id++; }

        {
            bool process = true;
            if (is_Max != nullptr && is_Max[state_id] == 0) process = false;

            if (process) {
                const uint node = ext_p[ext_p_idx];
                if (!device_G_prune[node]) {
                    if (Utils::get_judge(static_cast<int>(vals.w), _threshold, op)) {
                        device_G_prune[node] = true;
                    }
                }
            }
        }
    }

    
    
    for (uint64_t i = vec_limit * 4 + tid; i < intersection_length; i += stride) {
        if (is_Max != nullptr) {
            const uint state_id = i / ext_p_length;
            if (is_Max[state_id] == 0) continue;
        }
        const uint ext_p_idx = i % ext_p_length;
        const uint node = ext_p[ext_p_idx];
        if (device_G_prune[node]) continue;
        if (Utils::get_judge(static_cast<int>(intersection[i]), _threshold, op)) {
            device_G_prune[node] = true;
        }
    }
}









bool is_ILP_suitable(uint64_t total_works, int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    
    int sm_count = prop.multiProcessorCount;

    
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

    
    
    
    
    
    double estimated_occupancy_factor = 0.6;

    
    
    

    
    long long target_active_threads = (long long)sm_count * max_threads_per_sm;

    
    
    
    

    
    
    
    

    long long min_ilp_threads_needed = target_active_threads * estimated_occupancy_factor;

    
    
    long long actual_ilp_threads = total_works / 4;

    

    
    






    if (actual_ilp_threads >= min_ilp_threads_needed) {
        
        
        return true;
    }
    
    
    
    return false;
}

__global__ void changeGPrune(
    uint *ext_p, uint ext_p_length, uint* intersection, uint intersection_length, int _threshold, int* device_G_prune, CompareOp op) {
    const uint tid      = threadIdx.x + blockIdx.x * blockDim.x;
    const uint stride   = blockDim.x * gridDim.x;
    for (uint i = tid; i < intersection_length; i += stride) {
        if (Utils::get_judge(static_cast<int>(intersection[i]), _threshold, op)) {
            device_G_prune[ext_p[i % ext_p_length]] = true;
            
        }
    }
}


__global__ void any_satisfied_intersection_size(uint* intersection, uint64_t ext_length,
    uint64_t intersection_length, int _threshold, CompareOp op, uint* checks, int* found_flag) {
    const uint64_t tid    = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t stride = blockDim.x * gridDim.x;
    for (uint64_t idx = tid; idx < intersection_length; idx += stride) {
        uint i        = idx % ext_length;
        uint state_id = idx / ext_length;
        if (found_flag[state_id] == 1) {
            continue;
        }
        if (Utils::get_judge(static_cast<int>(intersection[state_id * ext_length + i]), _threshold, op)) {
            atomicExch(&found_flag[state_id], 1);
            checks[state_id] = 0;
        }
    }
}

__global__ void enumerateY_to_get_judge_difference_set_for_special(const uint* degrees, uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint Y_length, const uint* ext_partitions,
    const uint ext_length, arrMapTable g_state_viewer, const int threshold,
    const uint k, CompareOp op, uint* iterations, uint64_t batch_size, uint* checks,
    uint *intersection, const int threshold2, const CompareOp op2) {
    
    
    
    

    
    
    
    
    
    
    
    
    
    

    

    if (ext_length == 0 || Y_length == 0) {
        return;
    }
    const uint64_t totalLength = (uint64_t)Y_length * (uint64_t)ext_length * batch_size;
    const uint tid         = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size   = blockDim.x * gridDim.x;
    for (uint64_t idx = tid; idx < totalLength; idx += grid_size) {
        const uint64_t state_id = idx / (Y_length * ext_length);
        if (checks[state_id] == 0) {
            continue;
        }
        uint64_t state_inner_id     = idx % (Y_length * ext_length);
        uint Y_idx              = state_inner_id / ext_length; 
        const uint ext_inner_idx = state_inner_id % ext_length;
        if (iterations[state_id * ext_length + ext_inner_idx] != 0 && Utils::get_judge(static_cast<int>(intersection[state_id * ext_length + ext_inner_idx]), threshold2, op2)) {
            
            atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
            continue;
        }
        if (iterations[state_id * ext_length + ext_inner_idx] == 0) {
            continue;
        }
        const uint y_data              = __ldg(&Y_partitions[Y_idx]);
        const uint ext_node            = __ldg(&ext_partitions[ext_inner_idx]);
        const uint search_range_length = __ldg(&degrees[ext_node]);
        uint* search_range       = neighbors + neighbors_offset[ext_node];

        if (Utils::binarySearch(y_data, search_range, search_range_length)) {
            continue; 
        }
        if (iterations[state_id * ext_length + ext_inner_idx] == 0) {
            continue;
        }
        int value = static_cast<int>(g_state_viewer.get_value(y_data));
        if (Utils::get_judge(value, threshold, op)) {
            atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
        }
    }
}

#define FLAG_HAS_NON_NEIGHBOR 1 
#define FLAG_HAS_KILLER_Y     2 





__global__ void precompute_ext_flags_kernel(
    const uint* __restrict__ degrees,
    const uint* __restrict__ neighbors,
    const uint* __restrict__ neighbors_offset,
    const uint* __restrict__ Y_partitions,
    const uint Y_length,
    const uint* __restrict__ ext_partitions,
    const uint ext_length,
    arrMapTable g_state_viewer,
    const int threshold,
    CompareOp op,
    uint8_t* __restrict__ ext_flags 
) {
    
    
    extern __shared__ uint shared_Y[];

    const uint tid = threadIdx.x;
    const uint block_size = blockDim.x;

    
    for (uint i = tid; i < Y_length; i += block_size) {
        shared_Y[i] = Y_partitions[i];
    }
    __syncthreads();

    
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ext_length) return;

    const uint ext_node = __ldg(&ext_partitions[idx]);
    const uint deg = __ldg(&degrees[ext_node]);
    const uint* search_range = neighbors + neighbors_offset[ext_node];

    uint8_t flags = 0;

    
    for (uint i = 0; i < Y_length; ++i) {
        uint y_data = shared_Y[i];

        
        bool is_neighbor = false;

        
        
        if (deg < 32)
            is_neighbor = Utils::sequentialSearch_unrolled(y_data, search_range, deg);
        else
            is_neighbor = Utils::binarySearch(y_data, search_range, deg);
        
        
        
        
        
        

        if (!is_neighbor) {
            
            flags |= FLAG_HAS_NON_NEIGHBOR;

            
            int value = static_cast<int>(g_state_viewer.get_value(y_data));
            if (Utils::get_judge(value, threshold, op)) {
                flags |= FLAG_HAS_KILLER_Y;
                
                
                break;
            }
        }
    }

    ext_flags[idx] = flags;
}

__global__ void precompute_ext_flags_kernel_dynamic(
    const uint* __restrict__ degrees,
    const uint* __restrict__ neighbors,
    const uint* __restrict__ neighbors_offset,
    const uint* __restrict__ Y_partitions,
    const uint Y_length,
    const uint* __restrict__ ext_partitions,
    const uint ext_length,
    arrMapTable g_state_viewer,
    const int threshold,
    CompareOp op,
    uint8_t* __restrict__ ext_flags,
    const uint tile_size_limit 
) {
    
    extern __shared__ uint shared_Y[];

    const uint tid = threadIdx.x;
    const uint block_size = blockDim.x;
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    uint ext_node = 0;
    uint deg = 0;
    const uint* search_range = nullptr;
    bool is_valid_thread = (idx < ext_length);

    if (is_valid_thread) {
        ext_node = __ldg(&ext_partitions[idx]);
        deg = __ldg(&degrees[ext_node]);
        search_range = neighbors + neighbors_offset[ext_node];
    }

    uint8_t flags = 0;
    bool found_killer = false;

    
    
    
    for (uint tile_start = 0; tile_start < Y_length; tile_start += tile_size_limit) {

        
        uint current_tile_count = tile_size_limit;
        if (tile_start + tile_size_limit > Y_length) {
            current_tile_count = Y_length - tile_start;
        }

        
        for (uint i = tid; i < current_tile_count; i += block_size) {
            shared_Y[i] = Y_partitions[tile_start + i];
        }
        __syncthreads();

        
        if (is_valid_thread && !found_killer) {
            for (uint i = 0; i < current_tile_count; ++i) {
                uint y_data = shared_Y[i];

                bool is_neighbor = false;
                if (deg < 32)
                    is_neighbor = Utils::sequentialSearch_unrolled(y_data, search_range, deg);
                else
                    is_neighbor = Utils::binarySearch(y_data, search_range, deg);

                if (!is_neighbor) {
                    flags |= FLAG_HAS_NON_NEIGHBOR;

                    int value = static_cast<int>(g_state_viewer.get_value(y_data));
                    if (Utils::get_judge(value, threshold, op)) {
                        flags |= FLAG_HAS_KILLER_Y;
                        found_killer = true;
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        ext_flags[idx] = flags;
    }
}


__global__ void update_states_batched_kernel_varY(
    const uint* degrees,
    const uint* neighbors,
    const uint* neighbors_offset,
    const uint* Y_partitions, 
    const uint ext_length,
    const uint* ext_partitions,
    arrMapTable g_state_viewer,
    const int threshold,
    const uint k,
    CompareOp op,
    uint* iterations,
    uint* combinations,
    uint64_t batch_size,
    const uint* checks,
    uint *intersection,
    const int threshold2,
    const CompareOp op2,
    const uint8_t* __restrict__ ext_flags 
) {
    const uint64_t totalLength = (uint64_t)ext_length * batch_size;
    const uint tid         = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size   = blockDim.x * gridDim.x;

    for (uint64_t idx = tid; idx < totalLength; idx += grid_size) {
        const uint64_t state_id = idx / ext_length;

        
        if (checks[state_id] == 0) {
            continue;
        }

        const uint ext_inner_idx = idx % ext_length;

        
        if (Utils::get_judge(static_cast<int>(intersection[state_id * ext_length + ext_inner_idx]), threshold2, op2)) {
            atomicAnd(&iterations[idx], 0);
            continue;
        }

        
        
        
        
        uint8_t flag = __ldg(&ext_flags[ext_inner_idx]);
        if (flag == 0) {
            continue; 
        }

        
        
        
        const uint ext_node = __ldg(&ext_partitions[ext_inner_idx]);
        const uint search_range_length = __ldg(&degrees[ext_node]);
        const uint* combination = combinations + state_id * k;
        const uint* search_range = neighbors + neighbors_offset[ext_node];

        for (uint combination_id = 0; combination_id < k; combination_id++) {
            const uint Y_idx  = __ldg(&combination[combination_id]);
            const uint y_data = __ldg(&Y_partitions[Y_idx]);

            
            if (Utils::binarySearch(y_data, search_range, search_range_length)) {
                continue; 
            }

            int value = static_cast<int>(g_state_viewer.get_value(y_data));
            if (Utils::get_judge(value, threshold, op)) {
                
                atomicAnd(&iterations[idx], 0);
                break; 
            }
        }
    }
}






__global__ void update_states_batched_kernel(
    const uint ext_length,
    const uint8_t* __restrict__ ext_flags, 
    uint* iterations,
    uint64_t batch_size,
    const uint* checks,
    const uint* intersection,
    const int threshold2,
    const CompareOp op2
) {
    const uint64_t total_items = (uint64_t)ext_length * batch_size;
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_items) return;

    const uint64_t state_id = idx / ext_length;

    
    if (checks[state_id] == 0) return;

    const uint ext_inner_idx = idx % ext_length;
    const uint iter_idx = idx; 

    
    if (iterations[iter_idx] == 0) return;

    
    const uint8_t flags = ext_flags[ext_inner_idx];

    
    if (flags & FLAG_HAS_KILLER_Y) {
        iterations[iter_idx] = 0; 
        return;
    }

    
    if (flags & FLAG_HAS_NON_NEIGHBOR) {
        
        
        if (Utils::get_judge(static_cast<int>(intersection[iter_idx]), threshold2, op2)) {
            iterations[iter_idx] = 0;
        }
    }

    
}

__host__ void enumerateY_to_get_judge_difference_set_for_special_v4(
    const uint* degrees,
    const uint* neighbors,
    const uint* neighbors_offset,
    const uint* Y_partitions,
    const uint Y_length,
    const uint* ext_partitions,
    const uint ext_length,
    arrMapTable g_state_viewer,
    const int threshold,
    CompareOp op,
    uint* iterations,
    uint64_t batch_size,
    const uint* checks,
    uint *intersection,
    const int threshold2,
    const CompareOp op2
) {
    
    
    
    
    
    uint8_t* d_ext_flags = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_ext_flags, ext_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for ext_flags: %s\n", cudaGetErrorString(err));
        return;
    }

    
    
    
    
    {
        int blockSize = 256; 
        int gridSize = (ext_length + blockSize - 1) / blockSize;

        
        size_t sharedMemSize = Y_length * sizeof(uint);

        
        

        precompute_ext_flags_kernel<<<gridSize, blockSize, sharedMemSize>>>(
            degrees,
            neighbors,
            neighbors_offset,
            Y_partitions,
            Y_length,
            ext_partitions,
            ext_length,
            g_state_viewer,
            threshold,
            op,
            d_ext_flags 
        );

        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 1 launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ext_flags);
            return;
        }
    }

    
    
    
    
    {
        
        uint64_t total_items = batch_size * (uint64_t)ext_length;

        int blockSize = 256;
        
        
        
        int gridSize = (int)((total_items + blockSize - 1) / blockSize);

        update_states_batched_kernel<<<gridSize, blockSize>>>(
            ext_length,
            d_ext_flags, 
            iterations,
            batch_size,
            checks,
            intersection,
            threshold2,
            op2
        );

        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 2 launch failed: %s\n", cudaGetErrorString(err));
            
        }
    }

    
    
    
    cudaFree(d_ext_flags);
}

__host__ void enumerateY_to_get_judge_difference_set_for_special_v5(
    const uint* degrees,
    const uint* neighbors,
    const uint* neighbors_offset,
    const uint* Y_partitions,
    const uint Y_length,
    const uint* ext_partitions,
    const uint ext_length,
    arrMapTable g_state_viewer,
    const int threshold,
    CompareOp op,
    uint* iterations,
    uint64_t batch_size,
    const uint* checks,
    uint *intersection,
    const int threshold2,
    const CompareOp op2
) {
    
    
    
    uint8_t* d_ext_flags = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_ext_flags, ext_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for ext_flags: %s\n", cudaGetErrorString(err));
        return;
    }

    
    
    
    {

        int device_id = gpu_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        
        int max_shmem_per_block = prop.sharedMemPerBlock;

        
        
        
        if (prop.major >= 7) {
            int max_bytes_supported = 0;
            
            cudaDeviceGetAttribute(&max_bytes_supported, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

            
            
            cudaError_t set_attr_err = cudaFuncSetAttribute(
                precompute_ext_flags_kernel_dynamic,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                max_bytes_supported
            );

            if (set_attr_err == cudaSuccess) {
                
                max_shmem_per_block = max_bytes_supported;
            }
        }

        
        
        
        
        
        size_t target_shmem_bytes = max_shmem_per_block / 2;

        
        size_t needed_bytes_full = Y_length * sizeof(uint);
        if (needed_bytes_full < target_shmem_bytes) {
            target_shmem_bytes = needed_bytes_full;
        }

        
        uint calculated_tile_size = (uint)(target_shmem_bytes / sizeof(uint));

        
        if (calculated_tile_size > Y_length) {
            calculated_tile_size = Y_length;
        }
        if (calculated_tile_size == 0) calculated_tile_size = 256; 

        
        size_t dynamic_shmem_size = calculated_tile_size * sizeof(uint);

        
        
        
        int blockSize = 256;
        int gridSize = (ext_length + blockSize - 1) / blockSize;

        
        precompute_ext_flags_kernel_dynamic<<<gridSize, blockSize, dynamic_shmem_size>>>(
            degrees,
            neighbors,
            neighbors_offset,
            Y_partitions,
            Y_length,
            ext_partitions,
            ext_length,
            g_state_viewer,
            threshold,
            op,
            d_ext_flags,
            calculated_tile_size 
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 1 launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ext_flags);
            return;
        }
    }

    
    
    
    {
        
        uint64_t total_items = batch_size * (uint64_t)ext_length;

        int blockSize = 256;
        int gridSize = (int)((total_items + blockSize - 1) / blockSize);

        update_states_batched_kernel<<<gridSize, blockSize>>>(
            ext_length,
            d_ext_flags,
            iterations,
            batch_size,
            checks,
            intersection,
            threshold2,
            op2
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 2 launch failed: %s\n", cudaGetErrorString(err));
        }
    }

    
    
    
    cudaFree(d_ext_flags);
}

__host__ void enumerateY_to_get_judge_difference_set_for_special_v6(
    const uint* degrees,
    const uint* neighbors,
    const uint* neighbors_offset,
    const uint* Y_partitions,
    const uint Y_length,
    const uint* ext_partitions,
    const uint ext_length,
    arrMapTable g_state_viewer,
    const int threshold,
    CompareOp op,
    uint* iterations,
    uint64_t batch_size,
    const uint* checks,
    uint *intersection,
    const int threshold2,
    const CompareOp op2
) {
    
    
    
    uint8_t* d_ext_flags = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_ext_flags, ext_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for ext_flags: %s\n", cudaGetErrorString(err));
        return;
    }

    
    
    
    {
        
        
        
        static int s_max_shmem_per_block = 0;
        static bool s_is_initialized = false;

        if (!s_is_initialized) {
            int device_id = 0;
            cudaGetDevice(&device_id);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device_id);

            
            s_max_shmem_per_block = prop.sharedMemPerBlock;

            
            if (prop.major >= 7) {
                int max_bytes_supported = 0;
                
                cudaDeviceGetAttribute(&max_bytes_supported, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

                
                
                
                

                
                cudaError_t set_attr_err = cudaFuncSetAttribute(
                    precompute_ext_flags_kernel_dynamic,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    max_bytes_supported
                );

                if (set_attr_err == cudaSuccess) {
                    s_max_shmem_per_block = max_bytes_supported;
                } else {
                    
                    fprintf(stderr, "Warning: Failed to set dynamic shared memory attribute. Using default.\n");
                }
            }

            
            s_is_initialized = true;
            
        }

        
        
        
        
        
        size_t target_shmem_bytes = s_max_shmem_per_block / 2;

        
        size_t needed_bytes_full = Y_length * sizeof(uint);
        if (needed_bytes_full < target_shmem_bytes) {
            target_shmem_bytes = needed_bytes_full;
        }

        
        uint calculated_tile_size = (uint)(target_shmem_bytes / sizeof(uint));

        
        if (calculated_tile_size > Y_length) {
            calculated_tile_size = Y_length;
        }
        if (calculated_tile_size == 0) calculated_tile_size = 256; 

        
        size_t dynamic_shmem_size = calculated_tile_size * sizeof(uint);

        
        
        
        int blockSize = 256;
        int gridSize = (ext_length + blockSize - 1) / blockSize;

        precompute_ext_flags_kernel_dynamic<<<gridSize, blockSize, dynamic_shmem_size>>>(
            degrees,
            neighbors,
            neighbors_offset,
            Y_partitions,
            Y_length,
            ext_partitions,
            ext_length,
            g_state_viewer,
            threshold,
            op,
            d_ext_flags,
            calculated_tile_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 1 launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ext_flags);
            return;
        }
    }

    
    
    
    {
        
        uint64_t total_items = batch_size * (uint64_t)ext_length;

        int blockSize = 256;
        int gridSize = (int)((total_items + blockSize - 1) / blockSize);

        update_states_batched_kernel<<<gridSize, blockSize>>>(
            ext_length,
            d_ext_flags,
            iterations,
            batch_size,
            checks,
            intersection,
            threshold2,
            op2
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 2 launch failed: %s\n", cudaGetErrorString(err));
        }
    }

    
    
    
    cudaFree(d_ext_flags);
}


__host__ void enumerate_varY_to_get_judge_difference_set_for_special_v6(
    const uint* degrees,
    const uint* neighbors,
    const uint* neighbors_offset,
    const uint* Y_partitions,
    const uint Y_length,
    const uint* ext_partitions,
    const uint ext_length,
    arrMapTable g_state_viewer,
    const int threshold,
    const uint k,
    CompareOp op,
    uint* iterations,
    uint* combinations,
    uint64_t batch_size,
    const uint* checks,
    uint *intersection,
    const int threshold2,
    const CompareOp op2
) {
    
    
    
    uint8_t* d_ext_flags = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_ext_flags, ext_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for ext_flags: %s\n", cudaGetErrorString(err));
        return;
    }

    
    
    
    {
        
        
        
        static int s_max_shmem_per_block = 0;
        static bool s_is_initialized = false;

        if (!s_is_initialized) {
            int device_id = 0;
            cudaGetDevice(&device_id);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device_id);

            
            s_max_shmem_per_block = prop.sharedMemPerBlock;

            
            if (prop.major >= 7) {
                int max_bytes_supported = 0;
                
                cudaDeviceGetAttribute(&max_bytes_supported, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

                
                
                
                

                
                cudaError_t set_attr_err = cudaFuncSetAttribute(
                    precompute_ext_flags_kernel_dynamic,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    max_bytes_supported
                );

                if (set_attr_err == cudaSuccess) {
                    s_max_shmem_per_block = max_bytes_supported;
                } else {
                    
                    fprintf(stderr, "Warning: Failed to set dynamic shared memory attribute. Using default.\n");
                }
            }

            
            s_is_initialized = true;
            
        }

        
        
        
        
        
        size_t target_shmem_bytes = s_max_shmem_per_block / 2;

        
        size_t needed_bytes_full = Y_length * sizeof(uint);
        if (needed_bytes_full < target_shmem_bytes) {
            target_shmem_bytes = needed_bytes_full;
        }

        
        uint calculated_tile_size = (uint)(target_shmem_bytes / sizeof(uint));

        
        if (calculated_tile_size > Y_length) {
            calculated_tile_size = Y_length;
        }
        if (calculated_tile_size == 0) calculated_tile_size = 256; 

        
        size_t dynamic_shmem_size = calculated_tile_size * sizeof(uint);

        
        
        
        int blockSize = 256;
        int gridSize = (ext_length + blockSize - 1) / blockSize;

        precompute_ext_flags_kernel_dynamic<<<gridSize, blockSize, dynamic_shmem_size>>>(
            degrees,
            neighbors,
            neighbors_offset,
            Y_partitions,
            Y_length,
            ext_partitions,
            ext_length,
            g_state_viewer,
            threshold,
            op,
            d_ext_flags,
            calculated_tile_size
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 1 launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ext_flags);
            return;
        }
    }

    
    
    
    {
        
        uint64_t total_items = batch_size * (uint64_t)ext_length;

        int blockSize = 256;
        int gridSize = (int)((total_items + blockSize - 1) / blockSize);
        update_states_batched_kernel_varY<<<gridSize, blockSize>>>(
            degrees, neighbors, neighbors_offset,
            Y_partitions, 
            ext_length, ext_partitions,
            g_state_viewer, threshold, k, op,
            iterations, combinations, batch_size,
            checks, intersection, threshold2, op2,
            d_ext_flags 
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel 2 launch failed: %s\n", cudaGetErrorString(err));
        }
    }

    
    
    
    cudaFree(d_ext_flags);
}


__global__ void enumerateY_to_get_judge_difference_set_for_special_optimized_varY(const uint* degrees, uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint64_t Y_length, const uint* ext_partitions,
    const uint64_t ext_length, arrMapTable g_state_viewer, const int threshold,
    const uint k, CompareOp op, uint* iterations, uint* combinations, uint64_t batch_size, uint* checks,
    uint *intersection, const int threshold2, const CompareOp op2) {
    if (ext_length == 0 || Y_length == 0) {
        return;
    }
    const uint64_t totalLength = (uint64_t)ext_length * batch_size;
    const uint64_t tid         = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t grid_size   = blockDim.x * gridDim.x;
    for (uint64_t idx = tid; idx < totalLength; idx += grid_size) {
        const uint64_t state_id = idx / ext_length;
        if (checks[state_id] == 0) {
            continue;
        }
        const uint ext_inner_idx = idx % ext_length;
        if (Utils::get_judge(static_cast<int>(intersection[state_id * ext_length + ext_inner_idx]), threshold2, op2)) {
            
            atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
            continue;
        }
        const uint ext_node = __ldg(&ext_partitions[ext_inner_idx]);
        const uint search_range_length = __ldg(&degrees[ext_node]);
        const uint* combination = combinations + state_id * k;
        for (uint combination_id = 0; combination_id < k; combination_id++) {
            const uint Y_idx  = __ldg(&combination[combination_id]);
            const uint y_data = __ldg(&Y_partitions[Y_idx]);
            uint* search_range = neighbors + neighbors_offset[ext_node];
            if (Utils::binarySearch(y_data, search_range, search_range_length)) {
                continue;
            }
            int value = static_cast<int>(g_state_viewer.get_value(y_data));
            if (Utils::get_judge(value, threshold, op)) {
                atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
            }
        }
    }
}


__global__ void enumerateY_to_get_judge_difference_set_for_special_optimized_varY_sharedMemory(const uint* degrees, uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint64_t Y_length, const uint* ext_partitions,
    const uint64_t ext_length, arrMapTable g_state_viewer, const int threshold,
    const uint k, CompareOp op, uint* iterations, uint* combinations, uint64_t batch_size, uint* checks,
    uint *intersection, const int threshold2, const CompareOp op2) {
    if (ext_length == 0 || Y_length == 0) {
        return;
    }
    extern __shared__ uint checks_sharedMem[];
    for (uint i = threadIdx.x; i < batch_size; i += blockDim.x)
        checks_sharedMem[i] = checks[i];
    __syncthreads();
    const uint64_t totalLength = (uint64_t)ext_length * batch_size;
    const uint64_t tid         = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t grid_size   = blockDim.x * gridDim.x;
    for (uint64_t idx = tid; idx < totalLength; idx += grid_size) {
        const uint64_t state_id = idx / ext_length;
        if (checks_sharedMem[state_id] == 0) {
            continue;
        }
        const uint64_t ext_inner_idx = idx % ext_length;
        if (Utils::get_judge(static_cast<int>(intersection[state_id * ext_length + ext_inner_idx]), threshold2, op2)) {
            
            atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
            
            continue;
        }
        const uint ext_node = __ldg(&ext_partitions[ext_inner_idx]);
        const uint search_range_length = __ldg(&degrees[ext_node]);
        const uint* combination = combinations + state_id * k;
        for (uint combination_id = 0; combination_id < k; combination_id++) {
            const uint Y_idx  = __ldg(&combination[combination_id]);
            const uint y_data = __ldg(&Y_partitions[Y_idx]);
            uint* search_range = neighbors + neighbors_offset[ext_node];
            if (Utils::binarySearch(y_data, search_range, search_range_length)) {
                continue;
            }
            int value = static_cast<int>(g_state_viewer.get_value(y_data));
            if (Utils::get_judge(value, threshold, op)) {
                atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
                
            }
        }
    }
}


__global__ void enumerateY_to_get_judge_difference_set_for_special_sharedMemory(const uint* degrees, uint* neighbors,
    const uint* neighbors_offset, const uint* Y_partitions, const uint Y_length, const uint* ext_partitions,
    const uint ext_length, arrMapTable g_state_viewer, const int threshold,
    const uint k, const CompareOp op, uint* iterations, uint64_t batch_size, uint* checks,
    uint *intersection, const int threshold2, const CompareOp op2) {
    
    
    
    

    
    
    
    
    
    
    
    
    
    

    if (ext_length == 0 || Y_length == 0) {
        return;
    }
    extern __shared__ uint checks_sharedMem[];
    for (uint i = threadIdx.x; i < batch_size; i += blockDim.x)
        checks_sharedMem[i] = checks[i];
    __syncthreads();
    const uint64_t totalLength = (uint64_t)Y_length * (uint64_t)ext_length * batch_size;
    const uint tid         = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size   = blockDim.x * gridDim.x;
    for (uint64_t idx = tid; idx < totalLength; idx += grid_size) {
        const uint state_id = idx / (Y_length * ext_length);
        if (checks_sharedMem[state_id] == 0) {
            continue;
        }
        uint64_t state_inner_id     = idx % (Y_length * ext_length);
        uint Y_idx              = state_inner_id / ext_length; 
        auto y_data        = Y_partitions[Y_idx];
        uint ext_inner_idx = state_inner_id % ext_length;
        if (iterations[state_id * ext_length + ext_inner_idx] != 0 && Utils::get_judge(static_cast<int>(intersection[state_id * ext_length + ext_inner_idx]), threshold2, op2)) {
            
            atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
            continue;
        }
        if (iterations[state_id * ext_length + ext_inner_idx] == 0) {
            continue;
        }
        uint ext_node            = ext_partitions[ext_inner_idx];
        uint* search_range       = neighbors + neighbors_offset[ext_node];
        uint search_range_length = degrees[ext_node];
        if (Utils::binarySearch(y_data, search_range, search_range_length)) {
            continue; 
        }
        int value = static_cast<int>(g_state_viewer.get_value(y_data));
        if (iterations[state_id * ext_length + ext_inner_idx] != 0 && Utils::get_judge(value, threshold, op)) {
            atomicAnd(&iterations[state_id * ext_length + ext_inner_idx], 0);
        }
    }
}

__global__ void changeGPrune(uint* baseCurY, uint baseCurY_length, bool* device_G_prune) {
    const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
    const uint stride = blockDim.x * gridDim.x;
    for (uint i = tid; i < baseCurY_length; i += stride) {
        device_G_prune[baseCurY[i]] = true;
        
    }
}

__global__ void changeGPrune(const bool* selected_allStates, const uint* var_Y, const uint state_id, const uint length_varY, bool* device_G_prune) {
    const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
    const uint stride = blockDim.x * gridDim.x;
    for (uint i = tid; i < length_varY; i += stride) {
        if (!selected_allStates[state_id * length_varY + i]) continue;
        const uint node = var_Y[tid];
        device_G_prune[node] = true;
        
    }
}

__global__ void changeGPrune(
    uint* varY, uint varY_length, uint* combinations, uint combinations_length, bool* device_G_prune, uint k, uint* d_isMax) {
    const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
    const uint stride = blockDim.x * gridDim.x;
    for (uint i = tid; i < combinations_length; i += stride) {
        uint state_id = i / k;
        if (d_isMax[state_id] == 0)
            continue;
        if (combinations[i] >= varY_length) {
            continue;
        }
        device_G_prune[varY[combinations[i]]] = true;
        
    }
}

struct MaximalityCheckBuffer {
    
    uint64_t batch_size;
    uint k;
    uint ext_q_size;
    uint ext_p_size;

    
    uint* d_isMax;
    uint* h_isMax; 

    
    uint* d_intersection_q;
    uint* d_result2_q;
    int* d_found_flag_q;
    uint* d_iterations_q;

    
    uint* d_intersection_p;
    uint* d_iterations_p;

    
    uint* h_temp_result1;
    uint* h_temp_result2;

    cudaStream_t stream;

    
    MaximalityCheckBuffer(uint64_t _batch_size, uint _k, uint _ext_q_size, uint _ext_p_size, cudaStream_t _stream)
        : batch_size(_batch_size), k(_k), ext_q_size(_ext_q_size), ext_p_size(_ext_p_size), stream(_stream) {

        
        
        
        
        CUDA_ERROR_CHECK(cudaMallocHost((void**)&h_isMax, batch_size * sizeof(unsigned int)));
        CUDA_ERROR_CHECK(cudaMallocHost((void**)&h_temp_result1, batch_size * sizeof(unsigned int)));
        CUDA_ERROR_CHECK(cudaMallocHost((void**)&h_temp_result2, batch_size * sizeof(unsigned int)));


        
        CUDA_ERROR_CHECK(cudaMalloc(&d_isMax, sizeof(uint) * batch_size));

        
        if (ext_q_size > 0) {
            CUDA_ERROR_CHECK(cudaMalloc(&d_result2_q, sizeof(uint) * batch_size));
            CUDA_ERROR_CHECK(cudaMalloc(&d_found_flag_q, sizeof(int) * batch_size));
            CUDA_ERROR_CHECK(cudaMalloc(&d_intersection_q, sizeof(uint) * ext_q_size * batch_size));
            CUDA_ERROR_CHECK(cudaMalloc(&d_iterations_q, sizeof(uint) * ext_q_size * batch_size));
        } else {
            d_result2_q      = nullptr;
            d_found_flag_q   = nullptr;
            d_intersection_q = nullptr;
            d_iterations_q   = nullptr;
        }

        
        if (ext_p_size > 0) {
            CUDA_ERROR_CHECK(cudaMalloc(&d_intersection_p, sizeof(uint) * ext_p_size * batch_size));
            CUDA_ERROR_CHECK(cudaMalloc(&d_iterations_p, sizeof(uint) * ext_p_size * batch_size));
        } else {
            d_intersection_p = nullptr;
            d_iterations_p = nullptr;
        }
    }

    
    ~MaximalityCheckBuffer() {
        
        
        
        if (h_isMax)        cudaFreeHost(h_isMax);
        if (h_temp_result1) cudaFreeHost(h_temp_result1);
        if (h_temp_result2) cudaFreeHost(h_temp_result2);
        CUDA_ERROR_CHECK(cudaFree(d_isMax));
        if (d_result2_q) {
            CUDA_ERROR_CHECK(cudaFree(d_result2_q));
        }
        if (d_found_flag_q) {
            CUDA_ERROR_CHECK(cudaFree(d_found_flag_q));
        }
        if (d_intersection_q) {
            CUDA_ERROR_CHECK(cudaFree(d_intersection_q));
        }
        if (d_iterations_q) {
            CUDA_ERROR_CHECK(cudaFree(d_iterations_q));
        }
        if (d_intersection_p) {
            CUDA_ERROR_CHECK(cudaFree(d_intersection_p));
        }
        if (d_iterations_p)
            CUDA_ERROR_CHECK(cudaFree(d_iterations_p))
    }

    
    void reset() {
        memset(h_isMax, -1, sizeof(uint) * batch_size);
        uint64_t grid_num = CALC_GRID_DIM(batch_size, static_cast<uint64_t>(THREADS_PER_BLOCK));
        Utils::initialize_with_anyValue<<<grid_num, THREADS_PER_BLOCK, 0, stream>>> (d_isMax, batch_size, 1);

        if (ext_q_size > 0) {
            CUDA_ERROR_CHECK(cudaMemsetAsync(d_result2_q, 0, sizeof(uint) * batch_size, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(d_found_flag_q, 0, sizeof(int) * batch_size, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(d_intersection_q, 0, sizeof(uint) * ext_q_size * batch_size, stream));
            grid_num = CALC_GRID_DIM(ext_q_size * batch_size, THREADS_PER_BLOCK);
            Utils::initialize_with_anyValue<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(d_iterations_q, ext_q_size * batch_size, 1);
        }

        if (ext_p_size > 0) {
            CUDA_ERROR_CHECK(cudaMemsetAsync(d_intersection_p, 0, sizeof(uint) * ext_p_size * batch_size, stream));
            grid_num = CALC_GRID_DIM(ext_p_size * batch_size, THREADS_PER_BLOCK);
            Utils::initialize_with_anyValue<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(d_iterations_p, ext_p_size * batch_size, 1);
        }
    }

    
    









    static size_t totalUsedMemory(
        const uint64_t _batch_size,
        const uint _k,
        const uint _ext_q_size,
        const uint _ext_p_size,
        bool verbose = false)
    {
        
        
        size_t required_device_memory = 0;

        
        required_device_memory += sizeof(uint) * _batch_size;

        
        if (_ext_q_size > 0) {
            required_device_memory += sizeof(uint) * _batch_size;                 
            required_device_memory += sizeof(int)  * _batch_size;                 
            required_device_memory += sizeof(uint) * _ext_q_size * _batch_size;   
            required_device_memory += sizeof(uint) * _ext_q_size * _batch_size;   
        }

        
        if (_ext_p_size > 0) {
            required_device_memory += sizeof(uint) * _ext_p_size * _batch_size;   
            required_device_memory += sizeof(uint) * _ext_p_size * _batch_size;   
        }
        return required_device_memory;
    }
};

struct MaximalityCheckBuffer_optmized {
    
    
    uint64_t batch_size;
    uint k;
    uint ext_q_size;
    uint ext_p_size;

    
    uint* d_isMax;
    uint* h_isMax; 
    
    uint* d_intersection_q;

    
    uint* d_intersection_p;

    cudaStream_t stream;

    
    MaximalityCheckBuffer_optmized(uint64_t _batch_size, uint _k, uint _ext_q_size, uint _ext_p_size, cudaStream_t _stream)
        : batch_size(_batch_size), k(_k), ext_q_size(_ext_q_size), ext_p_size(_ext_p_size), stream(_stream) {
        
        CUDA_ERROR_CHECK(cudaMallocHost((void**)&h_isMax, batch_size * sizeof(unsigned int)));
        
        CUDA_ERROR_CHECK(cudaMalloc(&d_isMax, sizeof(uint) * batch_size));
        
        if (ext_q_size > 0) {
            CUDA_ERROR_CHECK(cudaMalloc(&d_intersection_q, sizeof(uint) * ext_q_size * batch_size));
        } else {
            d_intersection_q = nullptr;
        }

        
        if (ext_p_size > 0) {
            CUDA_ERROR_CHECK(cudaMalloc(&d_intersection_p, sizeof(uint) * ext_p_size * batch_size));
        } else {
            d_intersection_p = nullptr;
        }
    }

    
    ~MaximalityCheckBuffer_optmized() {
        if (h_isMax)        cudaFreeHost(h_isMax);
        CUDA_ERROR_CHECK(cudaFree(d_isMax));
        if (d_intersection_q) {
            CUDA_ERROR_CHECK(cudaFree(d_intersection_q));
        }
        if (d_intersection_p) {
            CUDA_ERROR_CHECK(cudaFree(d_intersection_p));
        }
    }

    
    void reset() {
        memset(h_isMax, -1, sizeof(uint) * batch_size);
        uint64_t grid_num = CALC_GRID_DIM(batch_size, static_cast<uint64_t>(THREADS_PER_BLOCK));
        Utils::initialize_with_anyValue<<<grid_num, THREADS_PER_BLOCK, 0, stream>>> (d_isMax, batch_size, 1);
        if (ext_q_size > 0) {
            CUDA_ERROR_CHECK(cudaMemsetAsync(d_intersection_q, 0, sizeof(uint) * ext_q_size * batch_size, stream));
        }
        if (ext_p_size > 0) {
            CUDA_ERROR_CHECK(cudaMemsetAsync(d_intersection_p, 0, sizeof(uint) * ext_p_size * batch_size, stream));
        }
    }

    
    








    static size_t totalUsedMemory(
    const uint64_t _batch_size,
    const uint _ext_q_size,
    const uint _ext_p_size)
    {
        
        
        size_t required_device_memory = 0;

        
        required_device_memory += sizeof(uint) * _batch_size;

        
        if (_ext_q_size > 0) {
            required_device_memory += sizeof(uint) * static_cast<size_t>(_ext_q_size) * _batch_size;
        }

        
        if (_ext_p_size > 0) {
            required_device_memory += sizeof(uint) * static_cast<size_t>(_ext_p_size) * _batch_size;
        }

        return required_device_memory;
    }
};

__global__ void get_conYcurY_intersection_sizesX_(Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, uint* combinations, uint64_t batch_size, uint k,
    uint varY_length, uint* counts, const int threshold, const uint start, const uint group_size) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid / group_size < partition_data.size[start] * batch_size) {
        const uint group_id      = tid / group_size;
        const uint inner_id      = tid % group_size;
        const uint group_num     = grid_size / group_size < 1 ? 1 : grid_size / group_size;
        const uint offset        = partition_data.get_global_offset(start);
        const uint end           = partition_data.size[start] * batch_size;
        for (uint idx = group_id; idx < end; idx += group_num) {
            uint nodeIdx     = idx % partition_data.size[start] + offset;
            uint state_id    = idx / partition_data.size[start];
            if (combinations[state_id * k] == -1) continue;
            const uint node  = partition_data.partition[nodeIdx];
            if (degrees[node] < threshold) continue;
            uint local_count = 0;
            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                auto y_idx          = con_cur_Y.get_value(neighbor);
                if (y_idx == varY_length + 1
                || (y_idx != -1 && Utils::sequentialSearch_unrolled(y_idx, combinations + state_id * k, k))) {
                    local_count++;
                }
            }
            
            
            atomicAdd(&counts[state_id * partition_length + nodeIdx], local_count);
        }
    }
}

__global__ void get_conYcurY_intersection_sizesX_optimized(
    Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y,
    uint* combinations, uint64_t batch_size, uint k,
    uint varY_length, uint* counts, const int threshold, const uint start, const uint group_size)
{
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;

    
    const uint end_idx = partition_data.size[start] * batch_size;
    const uint group_id = tid / group_size;
    const uint inner_id = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset = partition_data.get_global_offset(start);

    if (group_id >= end_idx) return;

    for (uint idx = group_id; idx < end_idx; idx += group_num) {
        uint nodeIdx = idx % partition_data.size[start] + offset;
        uint state_id = idx / partition_data.size[start];

        
        if (combinations[state_id * k] == -1) continue;

        const uint node = partition_data.partition[nodeIdx];
        const uint degree = degrees[node]; 

        if (degree < threshold) continue;

        uint local_count = 0;
        const uint row_start = neighbors_offset[node];

        
        
        
        uint i = inner_id;

        
        for (; i + group_size < degree; i += group_size * 2) {
            
            uint n1 = neighbors[row_start + i];
            uint n2 = neighbors[row_start + i + group_size];

            
            
            
            auto y_idx1 = con_cur_Y.get_value(n1);
            auto y_idx2 = con_cur_Y.get_value(n2);
            
            bool match1 = (y_idx1 == varY_length + 1) ||
                          (y_idx1 != -1 && Utils::sequentialSearch_unrolled(y_idx1, combinations + state_id * k, k));

            bool match2 = (y_idx2 == varY_length + 1) ||
                          (y_idx2 != -1 && Utils::sequentialSearch_unrolled(y_idx2, combinations + state_id * k, k));

            if (match1) local_count++;
            if (match2) local_count++;
        }

        
        for (; i < degree; i += group_size) {
            const uint neighbor = neighbors[row_start + i];
            auto y_idx = con_cur_Y.get_value(neighbor);
            if (y_idx == varY_length + 1 ||
               (y_idx != -1 && Utils::sequentialSearch_unrolled(y_idx, combinations + state_id * k, k))) {
                local_count++;
            }
        }

        atomicAdd(&counts[state_id * partition_length + nodeIdx], local_count);
    }
}

__global__ void get_conYcurY_intersection_sizesX_varY(Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    const uint *curYs, uint* combinations, uint64_t batch_size, uint k,
    uint* counts, const int threshold, const uint start, const uint group_size) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid / group_size < partition_data.size[start] * batch_size) {
        const uint group_id      = tid / group_size;
        const uint inner_id      = tid % group_size;
        const uint group_num     = grid_size / group_size < 1 ? 1 : grid_size / group_size;
        const uint offset        = partition_data.get_global_offset(start);
        const uint end           = partition_data.size[start] * batch_size;
        for (uint idx = group_id; idx < end; idx += group_num) {
            uint nodeIdx     = idx % partition_data.size[start] + offset;
            uint state_id    = idx / partition_data.size[start];
            if (combinations[state_id * k] == -1) continue;
            const uint node  = partition_data.partition[nodeIdx];
            if (degrees[node] < threshold) continue;
            uint local_count = 0;
            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                const uint comb_idx = Utils::sequentialSearch_unrolled_returnIdx(neighbor, curYs + state_id * k, k);
                if (comb_idx != -1)
                    local_count++;
                
                
                
            }
            
            Utils::sum_warpPrimitive(local_count, inner_id, &counts[state_id * partition_length + nodeIdx]);
            
        }
    }
}

__global__ void get_conYcurY_intersection_sizesX_conY(Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_Y_hashTable, uint* combinations, uint64_t batch_size, uint k,
    uint varY_length, uint* counts, const int threshold, const uint start, const uint group_size) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid / group_size < partition_data.size[start] * batch_size) {
        const uint group_id      = tid / group_size;
        const uint inner_id      = tid % group_size;
        const uint group_num     = grid_size / group_size < 1 ? 1 : grid_size / group_size;
        const uint offset        = partition_data.get_global_offset(start);
        const uint end           = partition_data.size[start] * batch_size;
        for (uint idx = group_id; idx < end; idx += group_num) {
            uint nodeIdx     = idx % partition_data.size[start] + offset;
            uint state_id    = idx / partition_data.size[start];
            if (combinations[state_id * k] == -1) continue;
            const uint node  = partition_data.partition[nodeIdx];
            if (degrees[node] < threshold) continue;
            uint local_count = 0;
            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                const bool found    = con_Y_hashTable.find_key(neighbor);
                if (found)
                    local_count++;
            }
            
            atomicAdd(&counts[state_id * partition_length + nodeIdx], local_count);
        }
    }
}

__global__ void get_conYcurY_intersection_sizesX(Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    int* con_cur_Y, uint* combinations, uint64_t batch_size, uint k,
    uint varY_length, uint* counts, const int threshold, const uint start, const uint group_size) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid / group_size < partition_data.size[start] * batch_size) {
        const uint group_id      = tid / group_size;
        const uint inner_id      = tid % group_size;
        const uint group_num     = grid_size / group_size < 1 ? 1 : grid_size / group_size;
        const uint offset        = partition_data.get_global_offset(start);
        const uint end           = partition_data.size[start] * batch_size;
        for (uint idx = group_id; idx < end; idx += group_num) {
            uint nodeIdx     = idx % partition_data.size[start] + offset;
            uint state_id    = idx / partition_data.size[start];
            if (combinations[state_id * k] == -1) continue;
            const uint node  = partition_data.partition[nodeIdx];
            if (degrees[node] < threshold) continue;
            uint local_count = 0;
            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor   = neighbors[neighbors_offset[node] + i];
                const uint y_idx      = con_cur_Y[neighbor];
                if (y_idx == varY_length + 1
                || (y_idx != -1 && Utils::sequentialSearch_unrolled(y_idx, combinations + state_id * k, k))) {
                    local_count++;
                }
            }
            
            atomicAdd(&counts[state_id * partition_length + nodeIdx], local_count);
        }
    }
}


__device__ __forceinline__ uint warpReduceSum(uint val) {
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void get_conYcurY_intersection_sizesX_optimized(
    Partition_Device_viewer_v3 partition_data,
    uint partition_length,
    const uint* __restrict__ degrees,          
    const uint* __restrict__ neighbors,
    const uint* __restrict__ neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y,
    const uint* __restrict__ combinations,
    uint64_t batch_size,
    uint k,
    uint varY_length,
    uint* counts,
    const int threshold,
    const uint start,
    const uint group_size)
{
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;

    
    const uint total_tasks = partition_data.size[start] * batch_size;

    
    if (tid / group_size < total_tasks) {
        const uint group_id      = tid / group_size;
        const uint inner_id      = tid % group_size; 
        const uint group_num     = grid_size / group_size < 1 ? 1 : grid_size / group_size;
        const uint offset        = partition_data.get_global_offset(start);

        
        for (uint idx = group_id; idx < total_tasks; idx += group_num) {
            uint nodeIdx     = idx % partition_data.size[start] + offset;
            uint state_id    = idx / partition_data.size[start];

            
            
            if (combinations[state_id * k] == -1) continue;

            const uint node  = partition_data.partition[nodeIdx];

            
            const uint node_degree = degrees[node];
            if (node_degree < threshold) continue;

            uint local_count = 0;
            const uint neighbor_start = neighbors_offset[node];

            
            
            
            #pragma unroll 4
            for (uint i = inner_id; i < node_degree; i += group_size) {
                const uint neighbor = neighbors[neighbor_start + i];

                
                auto y_idx = con_cur_Y.get_value(neighbor);

                
                if (y_idx == varY_length + 1 ||
                   (y_idx != -1 && Utils::sequentialSearch_unrolled(y_idx, combinations + state_id * k, k))) {
                    local_count++;
                }
            }

            
            

            
            if (group_size == 32) {
                local_count = warpReduceSum(local_count);
                if (inner_id == 0) {
                    atomicAdd(&counts[state_id * partition_length + nodeIdx], local_count);
                }
            }
            
            
            
            else {
                
                uint warp_sum = warpReduceSum(local_count);

                
                
                if ((inner_id & 31) == 0) {
                    atomicAdd(&counts[state_id * partition_length + nodeIdx], warp_sum);
                }
            }
            
            
            
        }
    }
}



























































__global__ void get_conYcurY_intersection_sizesX_64(
    Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y,
    uint* combinations,
    uint batch_size, 
    uint k,
    uint varY_length, uint* counts, const int threshold,
    const uint start, const uint group_size) {

    
    
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;

    
    const uint partition_size = partition_data.size[start];

    
    
    const uint64_t total_work = static_cast<uint64_t>(partition_size) * batch_size;

    
    const uint group_id = tid / group_size;

    
    if (group_id < total_work) {
        const uint inner_id = tid % group_size;

        
        const uint group_num = grid_size / group_size;
        const uint actual_group_num = (group_num < 1) ? 1 : group_num;
        const uint offset = partition_data.get_global_offset(start);

        
        for (uint64_t idx = group_id; idx < total_work; idx += actual_group_num) {

            
            
            
            uint state_id = static_cast<uint>(idx / partition_size);

            
            
            uint nodeIdx = static_cast<uint>(idx % partition_size) + offset;

            
            
            uint comb_offset = state_id * k;

            if (combinations[comb_offset] == static_cast<uint>(-1)) continue;

            const uint node = partition_data.partition[nodeIdx];
            if (degrees[node] < threshold) continue;

            uint local_count = 0;
            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                auto y_idx = con_cur_Y.get_value(neighbor);
                
                if (y_idx == varY_length + 1
                    || (y_idx != static_cast<uint>(-1)
                        && Utils::sequentialSearch_unrolled(y_idx, combinations + comb_offset, k))) {
                    local_count++;
                }
            }

            
            
            uint64_t count_idx = static_cast<uint64_t>(state_id) * partition_length + nodeIdx;
            atomicAdd(&counts[count_idx], local_count);
            
        }
    }
}

__global__ void get_conYcurY_intersection_sizesX_varY_64(
    Partition_Device_viewer_v3 partition_data, uint partition_length,
    uint* degrees, uint* neighbors, uint* neighbors_offset,
    uint* curYs, uint batch_size, uint k, uint* counts, const int threshold,
    const uint start, const uint group_size) {

    
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;

    const uint partition_size = partition_data.size[start];

    
    const uint64_t total_work = static_cast<uint64_t>(partition_size) * batch_size;

    const uint group_id = tid / group_size;

    
    if (group_id < total_work) {
        const uint inner_id = tid % group_size;
        const uint group_num = grid_size / group_size;
        const uint actual_group_num = (group_num < 1) ? 1 : group_num;
        const uint offset = partition_data.get_global_offset(start);

        
        for (uint64_t idx = group_id; idx < total_work; idx += actual_group_num) {

            
            
            const uint nodeIdx = static_cast<uint>(idx % partition_size) + offset;

            
            const uint state_id = static_cast<uint>(idx / partition_size);

            const uint node = partition_data.partition[nodeIdx];
            if (degrees[node] < threshold) continue;

            uint local_count = 0;

            
            
            const uint* current_curYs = curYs + static_cast<uint64_t>(state_id) * k;

            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                
                const uint comb_idx = Utils::sequentialSearch_unrolled_returnIdx(neighbor, current_curYs, k);
                if (comb_idx != -1)
                    local_count++;
            }

            
            
            
            uint64_t count_idx = static_cast<uint64_t>(state_id) * partition_length + nodeIdx;
            Utils::sum_warpPrimitive(local_count, inner_id, &counts[count_idx]);
        }
    }
}

__global__ void get_conYcurY_intersection_sizesX_sharedMem(const Partition_Device_viewer_v3 partition_data, const uint partition_length,
    const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, const uint* combinations, const uint64_t batch_size, const uint k,
    const uint varY_length, uint* counts,  const uint start, const uint group_size) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    extern __shared__ uint shared_combinations[];
    for (uint i = threadIdx.x; i < batch_size * k; i += blockDim.x) {
        shared_combinations[i] = combinations[i];
    }
    __syncthreads();
    const uint offset = partition_data.get_global_offset(start);
    if (tid / group_size < partition_data.size[start] * batch_size) {
        const uint group_id      = tid / group_size;
        const uint inner_id      = tid % group_size;
        const uint group_num     = grid_size / group_size;
        const uint end           = partition_data.size[start] * batch_size;
        for (uint idx = group_id; idx < end; idx += group_num) {
            const uint nodeIdx     = idx % partition_data.size[start] + offset;
            const uint state_id    = idx / partition_data.size[start];
            if (shared_combinations[state_id * k] == -1) continue;
            const uint node  = partition_data.partition[nodeIdx];
            uint local_count = 0;
            for (uint i = inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                auto y_idx          = con_cur_Y.get_value(neighbor);
                if (y_idx == varY_length + 1
                || (y_idx != -1 && (k < 32 ? Utils::sequentialSearch_unrolled(y_idx, shared_combinations + state_id * k, k) :
                    Utils::binarySearch(y_idx, shared_combinations + state_id * k, k)))) {
                    local_count++;
                    }
            }
            atomicAdd(&counts[state_id * partition_length + nodeIdx], local_count);
        }
    }
}


__global__ void get_conYcurY_intersection_sizes0(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, uint* counts) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (auto nodeIdx = tid; nodeIdx < partition_data.size[0]; nodeIdx += grid_size) {
        const uint node  = partition_data.partition[nodeIdx];
        uint local_count = 0;
        for (uint i = 0; i < degrees[node]; i++) {
            const uint neighbor = neighbors[neighbors_offset[node] + i];
            auto y_idx   = con_cur_Y.get_value(neighbor);
            if (y_idx != -1) {
                local_count++;
            }
        }
        counts[nodeIdx] = local_count;
    }
}

__global__ void get_conYcurY_intersection_sizes1(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, uint* counts) {
    auto block      = cg::this_thread_block();
    auto group2     = cg::tiled_partition<32>(block);
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    uint group2_num = grid_size / group2.size();
    uint group2_id  = tid / group2.size();
    if (group2_id < partition_data.size[1]) {
        const uint start = partition_data.size[0];
        const uint end   = start + partition_data.size[1];
        for (uint nodeIdx = start + group2_id; nodeIdx < end; nodeIdx += group2_num) {
            const uint node  = partition_data.partition[nodeIdx];
            uint offeset     = neighbors_offset[node];
            uint local_count = 0;
            for (uint i = group2.thread_rank(); i < degrees[node]; i += group2.size()) {
                const uint neighbor = neighbors[offeset + i];
                auto y_idx          = con_cur_Y.get_value(neighbor);
                if (y_idx != -1) {
                    local_count++;
                }
            }
            atomicAdd(&counts[nodeIdx], local_count);
        }
    }
}

__global__ void get_conYcurY_intersection_sizes2(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, uint* counts) {
    auto block           = cg::this_thread_block();
    auto group2          = cg::tiled_partition<32>(block);
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint tile_rank = tid / 1024;
    if (tile_rank < partition_data.size[2]) {
        const uint threadID      = tid % 1024;
        const uint totalTileSize = grid_size / 1024;
        const uint start         = partition_data.size[0] + partition_data.size[1];
        const uint end           = start + partition_data.size[2];
        for (uint nodeIdx = start + tile_rank; nodeIdx < end; nodeIdx += totalTileSize) {
            const uint node  = partition_data.partition[nodeIdx];
            uint local_count = 0;
            for (uint i = threadID; i < degrees[node]; i += 1024) {
                const uint neighbor = neighbors[neighbors_offset[node] + i];
                const uint y_idx    = con_cur_Y.get_value(neighbor);
                if (y_idx != -1) {
                    local_count++;
                }
            }
            atomicAdd(&counts[nodeIdx], local_count);
        }
    }
}


__global__ void get_conYcurY_intersection_sizesX(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, uint* counts,  const uint start, const uint group_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id  = tid / group_size;
    const uint inner_id  = tid % group_size;
    const uint group_num = grid_size / group_size;
    uint bias = partition_data.get_global_offset(start);
    
    
    
    
    
    for (uint idx = group_id; idx < partition_data.size[start]; idx += group_num) {
        const uint nodeIdx = idx + bias;
        const uint node  = partition_data.partition[nodeIdx];
        uint local_count = 0;
        for (uint i = inner_id; i < degrees[node]; i += group_size) {
            const uint neighbor = neighbors[neighbors_offset[node] + i];
            const uint y_idx    = con_cur_Y.get_value(neighbor);
            if (y_idx != -1) {
                local_count++;
            }
        }
        atomicAdd(&counts[nodeIdx], local_count);
    }
}











































































__global__ void any_satisfied_intersection_size_double_results(uint* intersection, uint length, int _threshold,
    CompareOp op, uint* result, int _threshold2, uint* result2, CompareOp op2) {
    const uint tid    = threadIdx.x + blockIdx.x * blockDim.x;
    const uint stride = blockDim.x * gridDim.x;
    for (uint i = tid; i < length; i += stride) {
        if (Utils::get_judge(static_cast<int>(intersection[i]), _threshold, op)) {
            *result = 1;
            break;
        }
        if (Utils::get_judge(static_cast<int>(intersection[i]), _threshold2, op2)) {
            *result2 = 1;
        }
    }
}

__global__ void compare(uint* a, uint* b, uint* result, uint length, CompareOp op) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint i = tid; i < length; i += grid_size) {
        if (!Utils::get_judge(static_cast<int>(a[i]), static_cast<int>(b[i]), op)) {
            atomicAdd(result, 1);
            break;
        }
        if (result[0] != 0) break;
    }
}


void CheckMaximality_for_special(uint* device_degrees, uint* device_neighbors, uint* device_neighbors_offset,
    bool*& device_G_prune, uint* combinations, uint64_t batch_size,
    arrMapTable g_state_viewer, Utils::HashTable::cudaHashTable3& hash_table,
    const uint final_curY_size, Partition_v3& X, Partition_v3& con_Y, Partition_v3& var_Y,
    Partition_v3& ext_p, Partition_v3& ext_q, MaximalityCheckBuffer& buffer, uint component_length, uint theta, uint k, cudaStream_t& stream) {

    const int temp_num = static_cast<int>(con_Y.totalSize + final_curY_size + 1);
    tuple<uint*, bool> result1(nullptr, false), result2(nullptr, false);


    
    buffer.reset();

    if (ext_q.totalSize > 0) {
        LAUNCH_PARTITION_KERNEL_V3(&ext_q, get_conYcurY_intersection_sizesX_, stream, ext_q.get_gpu_viewer(), ext_q.totalSize,
            device_degrees, device_neighbors, device_neighbors_offset, hash_table.get_viewer(), combinations,
            buffer.batch_size, buffer.k, var_Y.totalSize, buffer.d_intersection_q, temp_num - 1 - static_cast<int>(buffer.k));

        uint gridDim = min(CALC_GRID_DIM(ext_q.totalSize * batch_size, THREADS_PER_BLOCK), 1024 * 16);
        any_satisfied_intersection_size<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(buffer.d_intersection_q, ext_q.totalSize,
            ext_q.totalSize * batch_size, temp_num - 1, CompareOp::Equal, buffer.d_isMax, buffer.d_found_flag_q);
        if (con_Y.totalSize > 0) {
            uint totalThreads = con_Y.totalSize * ext_q.totalSize * batch_size;
            uint grid_num = CALC_GRID_DIM(totalThreads, THREADS_PER_BLOCK);
            enumerateY_to_get_judge_difference_set_for_special_v6(device_degrees, device_neighbors, device_neighbors_offset, con_Y.partition,
                con_Y.totalSize, ext_q.partition, ext_q.totalSize, g_state_viewer,
                (int) X.totalSize + 1 - (int) buffer.k, CompareOp::Less, buffer.d_iterations_q,
                batch_size, buffer.d_isMax, buffer.d_intersection_q, temp_num - 1 - static_cast<int>(buffer.k),
                CompareOp::Less);
        }
        if (component_length < temp_num - 1 - static_cast<int>(buffer.k) && var_Y.totalSize > 0) {
            uint64_t totalThreads = ext_q.totalSize * batch_size;
            uint grid_num = CALC_GRID_DIM(totalThreads, THREADS_PER_BLOCK);
            if (batch_size < 4096) {
                enumerateY_to_get_judge_difference_set_for_special_optimized_varY_sharedMemory<<<grid_num, THREADS_PER_BLOCK, batch_size * sizeof(uint),
                    stream>>>(device_degrees, device_neighbors, device_neighbors_offset, var_Y.partition,
                    var_Y.totalSize, ext_q.partition, ext_q.totalSize, g_state_viewer,
                    (int) X.totalSize + 1 - (int) buffer.k, buffer.k, CompareOp::Less, buffer.d_iterations_q,
                    combinations, batch_size, buffer.d_isMax, buffer.d_intersection_q,
                    temp_num - 1 - static_cast<int>(buffer.k), CompareOp::Less);
            }
            else {
                enumerateY_to_get_judge_difference_set_for_special_optimized_varY<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(device_degrees,
                    device_neighbors, device_neighbors_offset, var_Y.partition, var_Y.totalSize, ext_q.partition,
                    ext_q.totalSize, g_state_viewer, (int) X.totalSize + 1 - (int) buffer.k, buffer.k, CompareOp::Less,
                    buffer.d_iterations_q, combinations, batch_size, buffer.d_isMax, buffer.d_intersection_q,
                    temp_num - 1 - static_cast<int>(buffer.k), CompareOp::Less);
            }
            
            
            
            
            

        }
        result1 = Utils::multi_group_any_satisified(
            buffer.d_iterations_q, batch_size, ext_q.totalSize, 1, CompareOp::Equal, stream);
    }


    if (ext_p.totalSize > 0) {
        LAUNCH_PARTITION_KERNEL_V3(&ext_p, get_conYcurY_intersection_sizesX_, stream, ext_p.get_gpu_viewer(), ext_p.totalSize,
            device_degrees, device_neighbors, device_neighbors_offset, hash_table.get_viewer(), combinations,
            buffer.batch_size, buffer.k, var_Y.totalSize, buffer.d_intersection_p, temp_num - 1 - static_cast<int>(buffer.k));
        uint grid_num = min(CALC_GRID_DIM(ext_p.totalSize * batch_size, THREADS_PER_BLOCK), 1024 * 16);
        changeGPrune<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(ext_p.partition, ext_p.totalSize, buffer.d_intersection_p,
            ext_p.totalSize * batch_size, static_cast<int>(theta) - static_cast<int>(buffer.k), device_G_prune,
            CompareOp::GreaterEqual, buffer.d_isMax);
        if (con_Y.totalSize > 0) {
            enumerateY_to_get_judge_difference_set_for_special_v6(device_degrees,
                device_neighbors, device_neighbors_offset, con_Y.partition, con_Y.totalSize, ext_p.partition,
                ext_p.totalSize, g_state_viewer, (int) X.totalSize + 1 - (int) buffer.k, CompareOp::Less,
                buffer.d_iterations_p, batch_size, buffer.d_isMax, buffer.d_intersection_p,
                temp_num - 1 - static_cast<int>(buffer.k), CompareOp::Less);
        }
        if (component_length < temp_num - 1 - static_cast<int>(buffer.k) && var_Y.totalSize > 0) {
            uint64_t totalThreads = ext_p.totalSize * batch_size;
            uint grid_num = CALC_GRID_DIM(totalThreads, THREADS_PER_BLOCK);
            if (batch_size < 4096) {
                enumerateY_to_get_judge_difference_set_for_special_optimized_varY_sharedMemory<<<grid_num, THREADS_PER_BLOCK, batch_size * sizeof(uint),
                    stream>>>(device_degrees, device_neighbors, device_neighbors_offset, var_Y.partition,
                    var_Y.totalSize, ext_p.partition, ext_p.totalSize, g_state_viewer,
                    (int) X.totalSize + 1 - (int) buffer.k, buffer.k, CompareOp::Less, buffer.d_iterations_p,
                    combinations, batch_size, buffer.d_isMax, buffer.d_intersection_p,
                    temp_num - 1 - static_cast<int>(buffer.k), CompareOp::Less);
            }
            else {
                enumerateY_to_get_judge_difference_set_for_special_optimized_varY<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(device_degrees,
                    device_neighbors, device_neighbors_offset, var_Y.partition, var_Y.totalSize, ext_p.partition,
                    ext_p.totalSize, g_state_viewer, (int) X.totalSize + 1 - (int) buffer.k, buffer.k, CompareOp::Less,
                    buffer.d_iterations_p, combinations, batch_size, buffer.d_isMax, buffer.d_intersection_p,
                    temp_num - 1 - static_cast<int>(buffer.k), CompareOp::Less);
            }
        }

        result2 = Utils::multi_group_any_satisified(
            buffer.d_iterations_p, batch_size, ext_p.totalSize, 1, CompareOp::Equal, stream);
    }
    auto [any_result1, is_cpu1] = result1;
    auto [any_result2, is_cpu2] = result2;

    

    uint* h_result1_accessor = nullptr;
    uint* h_result2_accessor = nullptr;

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(buffer.h_isMax, buffer.d_isMax, sizeof(uint) * batch_size, cudaMemcpyDeviceToHost, stream));

    if (any_result1 != nullptr) {
        if (is_cpu1) {
            h_result1_accessor = any_result1;
        } else {
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                buffer.h_temp_result1, any_result1, sizeof(uint) * batch_size, cudaMemcpyDeviceToHost, stream));
            h_result1_accessor = buffer.h_temp_result1;
        }
    }

    if (any_result2 != nullptr) {
        if (is_cpu2) {
            h_result2_accessor = any_result2;
        } else {
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                buffer.h_temp_result2, any_result2, sizeof(uint) * batch_size, cudaMemcpyDeviceToHost, stream));
            h_result2_accessor = buffer.h_temp_result2;
        }
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
#pragma omp parallel for if(batch_size >= 4096) \
schedule(static) \
default(shared)
    for (uint i = 0; i < batch_size; i++) {
        
        bool check1 = (h_result1_accessor != nullptr && h_result1_accessor[i] == 1);
        bool check2 = (h_result2_accessor != nullptr && h_result2_accessor[i] == 1);

        if (buffer.h_isMax[i] != 0 && (check1 || check2)) {
            buffer.h_isMax[i] = 0;
        }
    }

    if (any_result1) {
        if (is_cpu1) {
            free(any_result1);
        } else {
            cudaFree(any_result1);
        }
    }
    if (any_result2) {
        if (is_cpu2) {
            free(any_result2);
        } else {
            cudaFree(any_result2);
        }
    }
}


void CheckMaximality_for_special_optimized(uint* device_degrees, uint* device_neighbors, uint* device_neighbors_offset,
    bool*& device_G_prune, uint* combinations, uint64_t batch_size,
    arrMapTable g_state_viewer, Utils::HashTable::cudaHashTable3& conY_curY_hashTable,
    const uint final_curY_size, Partition_v3& X, Partition_v3& con_Y, Partition_v3& var_Y,
    Partition_v3& ext_p, Partition_v3& ext_q, MaximalityCheckBuffer_optmized& buffer, const uint right_length, const uint theta, const uint k, cudaStream_t& stream) {

    const int temp_num = static_cast<int>(con_Y.totalSize + final_curY_size + 1);
    const int threshold = (int) X.totalSize + 1 - (int) buffer.k;
    
    buffer.reset();
    if (ext_q.totalSize > 0) {
        {
            Utils::L2PersistenceManager manager(gpu_id);
            manager.addPointer(conY_curY_hashTable.d_buffer, conY_curY_hashTable.totalBufferSize);
            manager.apply(stream);
            const uint* partition_sizes = ext_q.host_size_.get();
            const uint max_partitions = ext_q.max_partition_num_;
            for (uint start = 0; start < max_partitions; start++) {
                const uint current_partition_size = partition_sizes[start];
                if (current_partition_size == 0)
                    continue;
                const uint group_size = std::max(32u, safe_power_of_32(start));
                const uint64_t total_threads_needed = group_size * current_partition_size * batch_size;
                const uint grid_num = std::min(CALC_GRID_DIM(total_threads_needed, THREADS_PER_BLOCK), static_cast<uint>(MAX_GRID_DIM_X));
                if (grid_num == 0)
                    continue;
                if (total_threads_needed < UINT32_MAX) {
                    get_conYcurY_intersection_sizesX_<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(ext_q.get_gpu_viewer(), ext_q.totalSize,
                        device_degrees, device_neighbors, device_neighbors_offset, conY_curY_hashTable.get_viewer(), combinations,
                        batch_size, buffer.k, var_Y.totalSize, buffer.d_intersection_q, temp_num - 1 - static_cast<int>(buffer.k), start, group_size);
                }
                else {
                    get_conYcurY_intersection_sizesX_64<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(ext_q.get_gpu_viewer(), ext_q.totalSize,
                        device_degrees, device_neighbors, device_neighbors_offset, conY_curY_hashTable.get_viewer(), combinations,
                        batch_size, buffer.k, var_Y.totalSize, buffer.d_intersection_q, temp_num - 1 - static_cast<int>(buffer.k), start, group_size);
                }

            }
            manager.reset(stream);
        }
        
        
        
        if (threshold <= 0){
            



























            auto [multi_group_any, is_cpu] = Utils::multi_group_any_satisified(buffer.d_intersection_q, batch_size, ext_q.totalSize, temp_num - 1 - k, CompareOp::GreaterEqual, stream);
            
            auto [multi_group_any2, is_cpu2] = Utils::multi_group_any_satisified(buffer.d_intersection_q, batch_size, ext_q.totalSize, temp_num - 1, CompareOp::Equal, stream);
            
            if (is_cpu) {
                std::transform(multi_group_any, multi_group_any + batch_size,
                    multi_group_any2, multi_group_any, std::bit_or<uint>());
                
                bool any_Max = std::any_of(multi_group_any, multi_group_any + batch_size, [](uint element)->bool{return element == 0;});
                
                std::transform(multi_group_any, multi_group_any + batch_size, multi_group_any, [](uint element){ return element ^ 1;});
                
                if (any_Max) {
                    CUDA_ERROR_CHECK(cudaMemcpy(buffer.d_isMax, multi_group_any, batch_size * sizeof(uint), cudaMemcpyHostToDevice));
                }
                else {
                    memcpy(buffer.h_isMax, multi_group_any, batch_size * sizeof(uint));
                    free(multi_group_any);
                    free(multi_group_any2);
                    return;
                }
            }
            else {
                Utils::transform_compile_time(multi_group_any, multi_group_any2, multi_group_any,  batch_size, Utils::ElementwiseOp::OR, stream);
                bool any_Max = Utils::any_satisfied(multi_group_any, batch_size, 0, CompareOp::Equal, stream);
                Utils::transform_compile_time(multi_group_any, multi_group_any2, multi_group_any, batch_size, Utils::ElementwiseOp::NOT, stream);
                CUDA_ERROR_CHECK(cudaFreeAsync(multi_group_any2, stream));
                if (any_Max) {
                    CUDA_ERROR_CHECK(cudaMemcpyAsync(buffer.d_isMax, multi_group_any, batch_size * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
                    CUDA_ERROR_CHECK(cudaFreeAsync(multi_group_any, stream));
                }
                else {
                    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
                    CUDA_ERROR_CHECK(cudaMemcpy(buffer.h_isMax, multi_group_any, batch_size * sizeof(uint), cudaMemcpyDeviceToHost));
                    CUDA_ERROR_CHECK(cudaFree(multi_group_any));
                    return;
                }
            }
        }
    }
    if (ext_p.totalSize > 0) {
        {
            Utils::L2PersistenceManager manager(gpu_id);
            manager.addPointer(conY_curY_hashTable.d_buffer, conY_curY_hashTable.totalBufferSize);
            manager.apply(stream);
            const uint* partition_sizes = ext_p.host_size_.get();
            const uint max_partitions = ext_p.max_partition_num_;


            for (uint start = 0; start < max_partitions; start++) {
                const uint current_partition_size = partition_sizes[start];
                if (current_partition_size == 0)
                    continue;

                
                
                const uint group_size = std::max(32u, safe_power_of_32(start));
                
                const uint64_t total_threads_needed = group_size * current_partition_size * batch_size;
                uint grid_num = std::min(CALC_GRID_DIM(total_threads_needed, THREADS_PER_BLOCK), static_cast<uint>(MAX_GRID_DIM_X));

                if (grid_num == 0)
                    continue;

                if (total_threads_needed < UINT32_MAX) {
                    grid_num = min(grid_num, 1024 * 32);
                    get_conYcurY_intersection_sizesX_optimized<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                        ext_p.get_gpu_viewer(), ext_p.totalSize,
                        device_degrees, device_neighbors, device_neighbors_offset,
                        conY_curY_hashTable.get_viewer(), combinations,
                        buffer.batch_size, buffer.k, var_Y.totalSize, buffer.d_intersection_p,
                        temp_num - 1 - static_cast<int>(buffer.k), start, group_size);
                }
                
                
                
                
                
                
                else {
                    
                    const uint64_t max_threads = static_cast<uint64_t>(UINT32_MAX) - 1;
                    const uint64_t threads_per_state = static_cast<uint64_t>(group_size) * current_partition_size;
                    const uint64_t max_states_per_batch = std::max(1ULL, static_cast<unsigned long long>(max_threads / threads_per_state));

                    for (uint64_t processed_states = 0; processed_states < batch_size; ) {
                        const uint64_t states_this_batch = std::min(max_states_per_batch, batch_size - processed_states);
                        const uint64_t threads_this_batch = states_this_batch * threads_per_state;

                        uint grid_num_batch = std::min(
                            CALC_GRID_DIM(threads_this_batch, THREADS_PER_BLOCK),
                            static_cast<uint>(MAX_GRID_DIM_X)
                        );
                        grid_num_batch = std::min(grid_num_batch, 1024 * 32u);

                        if (grid_num_batch > 0) {
                            get_conYcurY_intersection_sizesX_optimized<<<grid_num_batch, THREADS_PER_BLOCK, 0, stream>>>(
                                ext_p.get_gpu_viewer(),
                                ext_p.totalSize,
                                device_degrees,
                                device_neighbors,
                                device_neighbors_offset,
                                conY_curY_hashTable.get_viewer(),
                                combinations + processed_states * buffer.k,
                                states_this_batch,
                                buffer.k,
                                var_Y.totalSize,
                                buffer.d_intersection_p + processed_states * ext_p.totalSize,
                                temp_num - 1 - static_cast<int>(buffer.k),
                                start,
                                group_size
                            );
                        }
                        processed_states += states_this_batch;
                    }
                }
            }

            manager.reset(stream);
        }

        
        {
            const uint64_t total_works = static_cast<uint64_t>(ext_p.totalSize) * static_cast<uint64_t>(batch_size);
            const bool suitable = is_ILP_suitable(total_works);
            if (suitable) {
                uint grid_num = CALC_GRID_DIM(ext_p.totalSize * batch_size, THREADS_PER_BLOCK * 4);
                changeGPrune_ILP<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(ext_p.partition, ext_p.totalSize, buffer.d_intersection_p,
                    ext_p.totalSize * batch_size, static_cast<int>(theta) - static_cast<int>(buffer.k), device_G_prune,
                    CompareOp::GreaterEqual, buffer.d_isMax);
            }
            else {
                uint grid_num = CALC_GRID_DIM(ext_p.totalSize * batch_size, THREADS_PER_BLOCK);
                if (grid_num > 16384)
                    grid_num = CALC_GRID_DIM(ext_p.totalSize * batch_size, THREADS_PER_BLOCK * 4);
                changeGPrune<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(ext_p.partition, ext_p.totalSize, buffer.d_intersection_p,
                    ext_p.totalSize * batch_size, static_cast<int>(theta) - static_cast<int>(buffer.k), device_G_prune,
                    CompareOp::GreaterEqual, buffer.d_isMax);
            }
        }

        if (threshold <= 0) {
            
            auto [multi_group_any, is_cpu] = Utils::multi_group_any_satisified(buffer.d_intersection_p, batch_size, ext_p.totalSize, temp_num - 1 - k, CompareOp::GreaterEqual, stream);
            if (not is_cpu) {
                uint* tmp = new uint[buffer.batch_size];
                CUDA_ERROR_CHECK(cudaMemcpy(tmp, multi_group_any, buffer.batch_size * sizeof(uint), cudaMemcpyDeviceToHost));
                CUDA_ERROR_CHECK(cudaFree(multi_group_any));
                multi_group_any = tmp;
            }
            bool any_Max = std::any_of(multi_group_any, multi_group_any + buffer.batch_size, [](uint element)->bool{return element == 0;});
            std::transform(multi_group_any, multi_group_any + buffer.batch_size, multi_group_any, [](uint element){ return element ^ 1;});
            if (any_Max) {
                uint* tmp_h_isMax = new uint[buffer.batch_size];
                CUDA_ERROR_CHECK(cudaMemcpy(tmp_h_isMax, buffer.d_isMax, buffer.batch_size * sizeof(uint), cudaMemcpyDeviceToHost));
                std::transform(multi_group_any, multi_group_any + buffer.batch_size,
                    tmp_h_isMax, multi_group_any, std::bit_or<uint>());
            }
            memcpy(buffer.h_isMax, multi_group_any, buffer.batch_size * sizeof(uint));
        }
    }
}


bool CheckMaximality_for_Gnerate(uint* device_degrees, uint* device_neighbors, uint* device_neighbors_offset,
    bool*& device_G_prune, arrMapTable& g_state_viewer,
    Utils::HashTable::cudaHashTable3& hash_table, Partition_v3& X, Partition_v3& con_Y, Partition_v3& cur_Y, Partition_v3& ext_p,
    Partition_v3& ext_q, const uint theta, const uint k, cudaStream_t& stream) {
    bool isMax = true;
    if (ext_q.totalSize) {
        uint* intersection;
        uint *d_result, *d_result2;
        int* d_found_flag;
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_result, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_result2, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_found_flag, sizeof(int), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&intersection, sizeof(uint) * ext_q.totalSize, stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_result, 0, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_result2, 0, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_found_flag, 0, sizeof(int), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(intersection, 0, sizeof(uint) * ext_q.totalSize, stream));
        LAUNCH_PARTITION_KERNEL_V3(&ext_q, get_conYcurY_intersection_sizesX, stream, ext_q.get_gpu_viewer(), device_degrees,
                device_neighbors, device_neighbors_offset, hash_table.get_viewer(), intersection);
        auto grid_size = CALC_GRID_DIM(ext_q.totalSize, THREADS_PER_BLOCK);
        any_satisfied_intersection_size_double_results<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(intersection, ext_q.totalSize,
            con_Y.totalSize + cur_Y.totalSize, CompareOp::Equal, d_result, static_cast<int>(con_Y.totalSize + cur_Y.totalSize) - static_cast<int>(k),
            d_result2, CompareOp::GreaterEqual);
        uint host_result, host_result2;
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_result, d_result, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_result2, d_result2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        if (host_result == 1) {
            CUDA_ERROR_CHECK(cudaFreeAsync(intersection, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(d_result, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(d_found_flag, stream));
            return false;
        }
        if (host_result2 == 1) {
            uint* iterations;
            CUDA_ERROR_CHECK(cudaMallocAsync(&iterations, sizeof(uint) * ext_q.totalSize, stream));
            
            uint gridNum = CALC_GRID_DIM(ext_q.totalSize, THREADS_PER_BLOCK);
            
            Utils::initialize_with_anyValue<<<gridNum, THREADS_PER_BLOCK, 0, stream>>>(iterations, ext_q.totalSize, 1);
            if (con_Y.totalSize > 0) {
                gridNum = CALC_GRID_DIM(con_Y.totalSize * ext_q.totalSize, THREADS_PER_BLOCK);
                enumerateY_to_get_judge_difference_set<<<gridNum, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, con_Y.partition, con_Y.totalSize, ext_q.partition, ext_q.totalSize,
                    g_state_viewer, (int) X.totalSize + 1 - (int) k, CompareOp::Less, iterations, intersection,
                    (int)con_Y.totalSize + (int)cur_Y.totalSize - (int)k, CompareOp::Less);
            }
            if (cur_Y.totalSize > 0) {
                gridNum = CALC_GRID_DIM(cur_Y.totalSize * ext_q.totalSize, THREADS_PER_BLOCK);
                enumerateY_to_get_judge_difference_set<<<gridNum, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, cur_Y.partition, cur_Y.totalSize, ext_q.partition, ext_q.totalSize,
                    g_state_viewer, (int) X.totalSize + 1 - (int) k, CompareOp::Less, iterations, intersection,
                    (int)con_Y.totalSize + (int)cur_Y.totalSize - (int)k, CompareOp::Less);
            }
            isMax = !Utils::any_satisfied(iterations, ext_q.totalSize, 1, CompareOp::Equal, stream);
            CUDA_ERROR_CHECK(cudaFreeAsync(iterations, stream));
        }
        CUDA_ERROR_CHECK(cudaFreeAsync(intersection, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_result, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_found_flag, stream));
    }
    auto gridNum = CALC_GRID_DIM(cur_Y.totalSize * ext_q.totalSize, THREADS_PER_BLOCK);
    changeGPrune<<<gridNum, THREADS_PER_BLOCK, 0, stream>>>(cur_Y.partition, cur_Y.totalSize, device_G_prune);
    if (ext_p.totalSize > 0) {
        uint* intersection;
        CUDA_ERROR_CHECK(cudaMallocAsync(&intersection, sizeof(uint) * ext_p.totalSize, stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(intersection, 0, sizeof(uint) * ext_p.totalSize, stream));
        uint* iterations;
        CUDA_ERROR_CHECK(cudaMallocAsync(&iterations, sizeof(uint) * ext_p.totalSize, stream));
        LAUNCH_PARTITION_KERNEL_V3(&ext_p, get_conYcurY_intersection_sizesX, stream, ext_p.get_gpu_viewer(), device_degrees,
                device_neighbors, device_neighbors_offset, hash_table.get_viewer(), intersection);
        gridNum = CALC_GRID_DIM(ext_p.totalSize, THREADS_PER_BLOCK);
        changeGPrune<<<gridNum, THREADS_PER_BLOCK, 0, stream>>>(ext_p.partition, ext_p.totalSize, intersection, ext_p.totalSize,
            static_cast<int>(theta) - static_cast<int>(k), device_G_prune, CompareOp::GreaterEqual);
        if (isMax) {
            uint gridDim = CALC_GRID_DIM(ext_p.totalSize, THREADS_PER_BLOCK);
            Utils::initialize_with_anyValue<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(iterations, ext_p.totalSize, 1);
            if (con_Y.totalSize > 0) {
                gridDim = CALC_GRID_DIM(ext_p.totalSize * con_Y.totalSize, THREADS_PER_BLOCK);
                enumerateY_to_get_judge_difference_set<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, con_Y.partition, con_Y.totalSize, ext_p.partition, ext_p.totalSize,
                    g_state_viewer, (int)X.totalSize + 1 - (int)k, CompareOp::Less, iterations, intersection,
                    (int)con_Y.totalSize + (int)cur_Y.totalSize - (int)k, CompareOp::Less);

            }
            if (cur_Y.totalSize > 0) {
                gridDim = CALC_GRID_DIM(ext_p.totalSize * cur_Y.totalSize, THREADS_PER_BLOCK);
                enumerateY_to_get_judge_difference_set<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, cur_Y.partition, cur_Y.totalSize, ext_p.partition, ext_p.totalSize,
                    g_state_viewer, (int)X.totalSize + 1 - (int)k, CompareOp::Less, iterations, intersection,
                    (int)con_Y.totalSize + (int)cur_Y.totalSize - (int)k, CompareOp::Less);
            }
            isMax = !Utils::any_satisfied(iterations, ext_p.totalSize, 1, CompareOp::Equal, stream);
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    }
    return isMax;
}

__global__ void get_conYcurY_intersection_sizesX(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y,
    bool* selected_allStates, uint varY_length, uint state_id,uint* counts,  const uint start, const uint group_size) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    uint group_id   = tid / group_size;
    uint laneID     = tid % group_size;
    uint group_num  = grid_size / group_size;
    uint bias = partition_data.get_global_offset(start);
    
    
    const bool* selected  = selected_allStates + varY_length * state_id;
    for (uint idx = group_id; idx < partition_data.size[start]; idx += group_num) {
        uint nodeIdx     = idx + bias;
        const uint node  = partition_data.partition[nodeIdx];
        uint offeset     = neighbors_offset[node];
        uint local_count = 0;
        for (uint i = laneID; i < degrees[node]; i += group_size) {
            const uint neighbor = neighbors[offeset + i];
            auto y_idx = con_cur_Y.get_value(neighbor);
            if (y_idx == varY_length + 1 || (y_idx != -1 && selected[y_idx]))
                local_count ++;
        }
        atomicAdd(&counts[nodeIdx], local_count);
    }
}

bool CheckMaximality(uint* device_degrees, uint* device_neighbors, uint* device_neighbors_offset,
    Utils::MultiHash::cudaMultiHashTable_gpu_viewer delta_G_temps, bool* device_G_prune,
    HashTable_viewer g_temp_viewer,
    arrMapTable g_state_viewer,
    int* device_G_label, Partition_v3& X, Partition_v3& con_Y,
    Partition_v3& base_cur_Y, Partition_v3& var_Y, Partition_v3& base_candY_p, Partition_v3& ext_p, Partition_v3& ext_q, uint state_id,
    bool* selected_allStates, uint* selected_count, Utils::HashTable::cudaHashTable3& hash_table, const uint theta, const uint k, cudaStream_t stream) {
    bool isMax = true;
    if (base_candY_p.totalSize + var_Y.totalSize > 0) {
        uint *result2, host_result2;
        CUDA_ERROR_CHECK(cudaMalloc(&result2, sizeof(uint)));
        CUDA_ERROR_CHECK(cudaMemset(result2, 0, sizeof(uint)));
        if (base_candY_p.totalSize > 0) {
            uint gridDim = min(MAX_GRID_DIM_X, (X.totalSize * base_candY_p.totalSize + 255) / 256);
            
            
            
            
            
            enumerate_Y_get_earlyStop2<<<gridDim, 256, 0, stream>>>(X.partition, X.totalSize, base_candY_p.partition,
                base_candY_p.totalSize, device_degrees, device_neighbors, device_neighbors_offset, selected_count,
                static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize + 1 - k), state_id, result2, device_G_label,
                g_temp_viewer, delta_G_temps, nullptr);
            CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_result2, result2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            if (host_result2 == 0) {
                CUDA_ERROR_CHECK(cudaFree(result2));
                return false;
            }
        }
        if (var_Y.totalSize > 0) {
            uint gridDim = min(MAX_GRID_DIM_X, (X.totalSize * var_Y.totalSize + 255) / 256);
            
            
            
            
            enumerate_Y_get_earlyStop2<<<gridDim, 256, 0, stream>>>(X.partition, X.totalSize, var_Y.partition,
                var_Y.totalSize, device_degrees, device_neighbors, device_neighbors_offset, selected_count,
                static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize + 1 - k), state_id, result2, device_G_label,
                g_temp_viewer, delta_G_temps, selected_allStates);
            CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_result2, result2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            if (host_result2 == 0) {
                CUDA_ERROR_CHECK(cudaFreeAsync(result2, stream));
                return false;
            }
        }
        CUDA_ERROR_CHECK(cudaFreeAsync(result2, stream));
    }
    if (ext_q.totalSize) {
        uint* intersection;
        uint *d_result, *d_result2;
        int* d_found_flag;
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_result, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_result2, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&d_found_flag, sizeof(int), stream));
        CUDA_ERROR_CHECK(cudaMallocAsync(&intersection, sizeof(uint) * ext_q.totalSize, stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_result, 0, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_result2, 0, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_found_flag, 0, sizeof(int), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(intersection, 0, sizeof(uint) * ext_q.totalSize, stream));
        LAUNCH_PARTITION_KERNEL_V3(&ext_q, get_conYcurY_intersection_sizesX, stream, ext_q.get_gpu_viewer(), device_degrees,
                device_neighbors, device_neighbors_offset, hash_table.get_viewer(), selected_allStates, var_Y.totalSize,
                state_id, intersection);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        auto blockDim = min(MAX_GRID_DIM_X, next_power_of_2(ext_q.totalSize / 256));
        any_satisfied_intersection_size_double_results<<<blockDim, 256, 0, stream>>>(intersection, ext_q.totalSize,
            con_Y.totalSize + base_cur_Y.totalSize, state_id, selected_count, CompareOp::Equal, d_result, d_found_flag,
            con_Y.totalSize + base_cur_Y.totalSize - k, d_result2, CompareOp::GreaterEqual);
        uint host_result, host_result2;
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_result, d_result, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_result2, d_result2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        if (host_result == 1) {
            CUDA_ERROR_CHECK(cudaFreeAsync(d_result2, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(intersection, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(d_result, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(d_found_flag, stream));
            return false;
        }
        if (host_result2 == 1) {
            uint* iterations;
            CUDA_ERROR_CHECK(cudaMallocAsync(&iterations, sizeof(uint) * ext_q.totalSize, stream));
            Utils::initialize_with_anyValue<<<512, 512, 0, stream>>>(iterations, ext_q.totalSize, 1);
            if (con_Y.totalSize > 0) {
                enumerateY_to_get_judge_difference_set<<<512, 512, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, con_Y.partition, con_Y.totalSize, ext_q.partition, ext_q.totalSize,
                    g_state_viewer, (int) X.totalSize + 1 - (int) k, CompareOp::Less, iterations, nullptr, state_id,
                    selected_count, intersection,
                    static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize) - static_cast<int>(k), CompareOp::Less);
            }
            if (base_cur_Y.totalSize > 0) {
                enumerateY_to_get_judge_difference_set<<<512, 512, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, base_cur_Y.partition, base_cur_Y.totalSize, ext_q.partition,
                    ext_q.totalSize, g_state_viewer, (int) X.totalSize + 1 - (int) k, CompareOp::Less, iterations,
                    nullptr, state_id, selected_count, intersection,
                    static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize) - static_cast<int>(k), CompareOp::Less);
            }
            if (var_Y.totalSize > 0) {
                enumerateY_to_get_judge_difference_set<<<512, 512, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, var_Y.partition, var_Y.totalSize, ext_q.partition, ext_q.totalSize,
                    g_state_viewer, (int) X.totalSize + 1 - (int) k, CompareOp::Less, iterations, selected_allStates,
                    state_id, selected_count, intersection,
                    static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize) - static_cast<int>(k), CompareOp::Less);
            }
            
            
            isMax = !Utils::any_satisfied(iterations, ext_q.totalSize, 1, CompareOp::Equal, stream);
            CUDA_ERROR_CHECK(cudaFreeAsync(iterations, stream));
        }
        CUDA_ERROR_CHECK(cudaFreeAsync(intersection, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_result, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_found_flag, stream));
        CUDA_ERROR_CHECK(cudaFreeAsync(d_result2, stream));
    }
    if (base_cur_Y.totalSize > 0)
        changeGPrune<<<512, 512, 0, stream>>>(base_cur_Y.partition, base_cur_Y.totalSize, device_G_prune);
    changeGPrune<<<512, 512, 0, stream>>>(selected_allStates, var_Y.partition, state_id, var_Y.totalSize, device_G_prune);
    if (ext_p.totalSize) {
        uint* intersection;
        CUDA_ERROR_CHECK(cudaMallocAsync(&intersection, sizeof(uint) * ext_p.totalSize, stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(intersection, 0, sizeof(uint) * ext_p.totalSize, stream));
        LAUNCH_PARTITION_KERNEL_V3(&ext_p, get_conYcurY_intersection_sizesX, stream, ext_p.get_gpu_viewer(), device_degrees,
                device_neighbors, device_neighbors_offset, hash_table.get_viewer(), selected_allStates, var_Y.totalSize,
                state_id, intersection);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        changeGPrune<<<512, 512, 0, stream>>>(ext_p.partition, ext_p.totalSize, intersection, ext_p.totalSize,
            theta - k, device_G_prune, CompareOp::GreaterEqual);
        if (isMax) {
            uint* iterations;
            CUDA_ERROR_CHECK(cudaMallocAsync(&iterations, sizeof(uint) * ext_p.totalSize, stream));
            Utils::initialize_with_anyValue<<<512, 512, 0, stream>>>(iterations, ext_p.totalSize, 1);
            if (con_Y.totalSize > 0) {
                enumerateY_to_get_judge_difference_set<<<512, 512, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, con_Y.partition, con_Y.totalSize, ext_p.partition, ext_p.totalSize,
                    g_state_viewer, X.totalSize + 1 - k, CompareOp::Less, iterations, nullptr, state_id, selected_count,
                    intersection, static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize) - static_cast<int>(k),
                    CompareOp::Less);
            }
            if (base_cur_Y.totalSize > 0) {
                enumerateY_to_get_judge_difference_set<<<512, 512, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, base_cur_Y.partition, base_cur_Y.totalSize, ext_p.partition,
                    ext_p.totalSize, g_state_viewer, X.totalSize + 1 - k, CompareOp::Less, iterations, nullptr,
                    state_id, selected_count, intersection,
                    static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize) - static_cast<int>(k), CompareOp::Less);
            }
            if (var_Y.totalSize > 0) {
                enumerateY_to_get_judge_difference_set<<<512, 512, 0, stream>>>(device_degrees, device_neighbors,
                    device_neighbors_offset, var_Y.partition, var_Y.totalSize, ext_p.partition, ext_p.totalSize,
                    g_state_viewer, X.totalSize + 1 - k, CompareOp::Less, iterations, selected_allStates, state_id,
                    selected_count, intersection,
                    static_cast<int>(con_Y.totalSize + base_cur_Y.totalSize) - static_cast<int>(k), CompareOp::Less);
            }
            isMax = !Utils::any_satisfied(iterations, ext_p.totalSize, 1, CompareOp::Equal, stream);
        }
        CUDA_ERROR_CHECK(cudaFreeAsync(intersection, stream));
    }
    return isMax;
}


#endif 
