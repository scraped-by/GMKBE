




#ifndef _FOR_BFS_HASHTABLEHASHTABLE_CUH_
#define _FOR_BFS_HASHTABLEHASHTABLE_CUH_
#include <for_checkmaximality-HashTableHashTable.cuh>
#include <for_special-HashTableHashTable.cuh>
#include <gpu_utils.cuh>
#include <hashTable.cuh>
#include <output.cuh>
#include <partition.cuh>

struct Layer {
    bool* selected_allStates;
    bool* excluded_allStates;
    uint state_count;
    int level{}; 
};

__global__ void init_new_layer(bool* new_selected, bool* new_excluded, bool* old_selected, bool* old_excluded,
    uint* extentions, uint old_state_num, uint varY_size, uint old_level, uint* counts) {
    
    const uint tid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint state_id = tid / varY_size;
    uint varY_id  = tid % varY_size;
    if (state_id < old_state_num + counts[0]) {
        if (state_id < old_state_num) {
            if (extentions[state_id] != -2) { 
                new_selected[extentions[state_id] * varY_size + varY_id] = old_selected[state_id * varY_size + varY_id];
                new_excluded[extentions[state_id] * varY_size + varY_id] = old_excluded[state_id * varY_size + varY_id];
                if (varY_id == 0) {
                    new_selected[extentions[state_id] * varY_size + old_level] = true;
                }
            }
        }
        if (state_id >= counts[0]) {
            new_selected[state_id * varY_size + varY_id] = old_selected[(state_id - counts[0]) * varY_size + varY_id];
            new_excluded[state_id * varY_size + varY_id] = old_excluded[(state_id - counts[0]) * varY_size + varY_id];
            if (varY_id == 0) {
                new_excluded[state_id * varY_size + old_level] = true;
            }
        }
    }
}




























__global__ void get_GLable_intersection_size0(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, int* G_label, uint* _count) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    for (auto nodeIdx = tid; nodeIdx < partition_data.size[0]; nodeIdx += grid_size) {
        const uint node = partition_data.partition[nodeIdx];
        uint local_count = 0;
        for (uint i = 0; i < degrees[node]; i++) {
            const uint neighbor = neighbors[neighbors_offset[node] + i];
            if (G_label[neighbor]) {
                local_count ++;
            }
        }
        atomicAdd(_count, local_count);
    }
}
__global__ void get_GLable_intersection_sizeX(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, int* G_label, uint* _count,
     const uint start, const uint group_size) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size  = blockDim.x * gridDim.x;
    const uint group_id   = tid / group_size;
    const uint laneID     = tid % group_size;
    const uint group_num  = grid_size / group_size;
    const uint bias = partition_data.get_global_offset(start);
    
    
    const uint end = partition_data.size[start];
    for (uint idx = group_id; idx < end; idx += group_num) {
        const uint nodeIdx    = idx + bias;
        const uint node = partition_data.partition[nodeIdx];
        const uint offeset    = neighbors_offset[node];
        uint local_count = 0;
        for (uint i = laneID; i < degrees[node]; i += group_size) {
            const uint neighbor = neighbors[offeset + i];
            if (G_label[neighbor])
                local_count ++;
        }
        atomicAdd(_count, local_count);
    }
}



























__global__ void get_conYcurY_intersection_sizes0(Partition_Device_viewer_v3 partition_data, uint* degrees, uint* neighbors,
    uint* neighbors_offset, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> con_cur_Y, bool* selected_allStates, uint varY_length, uint state_id,uint* counts) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    bool* selected  = selected_allStates + varY_length * state_id;
    for (auto nodeIdx = tid; nodeIdx < partition_data.size[0]; nodeIdx += grid_size) {
        const uint node = partition_data.partition[nodeIdx];
        uint local_count = 0;
        for (uint i = 0; i < degrees[node]; i++) {
            const uint neighbor = neighbors[neighbors_offset[node] + i];
            auto y_idx = con_cur_Y.get_value(neighbor);
            if (y_idx == varY_length + 1 || (y_idx != -1 && y_idx < varY_length && selected[y_idx]))
                local_count ++;
        }
        counts[nodeIdx] = local_count;
    }
}





__global__ void count_selected_nums(
    const bool* selected_allStates, uint* select_nums, uint varY_size, uint state_num) {
    const uint tid       = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid;; idx += grid_size) {
        uint varY_id  = idx % varY_size;
        uint state_id = idx / varY_size;
        if (state_id >= state_num) {
            break;
        }
        if (selected_allStates[state_id * varY_size + varY_id]) {
            atomicAdd(&select_nums[state_id], 1);
        }
    }
}


__global__ void enumerate_varY_calGTemp0(Partition_Device_viewer_v3 varY, uint* degrees, uint current_state_num,
    Layer current_level_states, uint* neighbors, uint* neighbors_offset,
    Utils::HashTable::cudaHashTable3_gpu_viewer<int32_t> G_label_gpu_viewer,
    Utils::MultiHash::cudaMultiHashTable_gpu_viewer delta_G_temps, uint varY_length,
    const bool* state_bar) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    for (auto enumerateID = tid; enumerateID < varY.size[0] * current_state_num; enumerateID += grid_size) {
        const uint stateID = enumerateID / varY.size[0];
        if (state_bar[stateID]) {
            auto innerID    = enumerateID % varY.size[0];
            const uint node = varY.partition[innerID];
            if (current_level_states.selected_allStates[stateID * varY_length + innerID]) {
                uint offeset = neighbors_offset[node];
                for (uint i = 0; i < degrees[node]; i++) {
                    const uint neighbor = neighbors[offeset + i];
                    if (G_label_gpu_viewer.get_value(neighbor) == 1) {
                        delta_G_temps.add(neighbor, stateID);
                    }
                }
            }
        }
    }
}

__global__ void enumerate_varY_calGTempX(Partition_Device_viewer_v3 varY, uint* degrees, uint current_state_num,
    Layer current_level_states, uint* neighbors, uint* neighbors_offset,
    int* G_label,
    Utils::MultiHash::cudaMultiHashTable_gpu_viewer delta_G_temps, uint varY_length,
    const bool* state_bar,  const uint start, const uint group_size) {
    
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size  = blockDim.x * gridDim.x;
    const uint group_id = tid / group_size;
    const uint group_inner_id = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset = varY.get_global_offset(start);
    for (uint enumerateID = group_id; enumerateID < varY.size[start] * current_state_num; enumerateID += group_num) {
        const uint stateID = enumerateID / varY.size[start];
        const uint nodeID    = (enumerateID % varY.size[start]) + offset;
        if (state_bar[stateID] && current_level_states.selected_allStates[stateID * varY_length + nodeID]) {
            const uint node = varY.partition[nodeID];
            const uint neighbor_offset    = neighbors_offset[node];
            for (uint i = group_inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[neighbor_offset + i];
                if (G_label[neighbor] == 1) {
                    assert(delta_G_temps.add(neighbor, stateID));
                }
            }
        }
    }
}



__global__ void check_states_by_candidates_blockwise(const uint* __restrict__ varY, uint varY_length,
    const uint* __restrict__ X, uint X_length, const uint* __restrict__ degrees, const uint* __restrict__ neighbors,
    const uint* __restrict__ neighbors_offset, int* G_label,
    Utils::HashTable::cudaHashTable3_gpu_viewer<int32_t> G_tmp_gpu_viewer,
    Utils::MultiHash::cudaMultiHashTable_gpu_viewer delta_G_temps,
    Layer current_level_states, 
    const int threshold, const uint* __restrict__ selected_counts, 
    uint current_state_num,
    uint* __restrict__ state_ok, 
    const bool* __restrict__ state_bar) {
    







    uint stateID  = blockIdx.x;
    uint varY_idx = blockIdx.y;
    if (stateID >= current_state_num || varY_idx >= varY_length) {
        return;
    }
    
    
    
    if (!state_bar[stateID]) {
        state_ok[stateID] = 1;
        return;
    }
    if (state_ok[stateID]) {
        return;
    }
    
    if (current_level_states.selected_allStates[stateID * varY_length + varY_idx]) {
        return;
    }
    uint u = varY[varY_idx];
    auto R = threshold + static_cast<int32_t>(selected_counts[stateID]); 

    __shared__ int fail; 
    if (threadIdx.x == 0) {
        fail = 0;
    }
    __syncthreads();
    
    for (uint xIdx = threadIdx.x; xIdx < X_length; xIdx += blockDim.x) {
        
        if (__ldg(&state_ok[stateID]) || fail) {
            break;
        }

        uint x = X[xIdx];
        
        auto delta = delta_G_temps.get_value(x, stateID);
        int32_t original_val = G_tmp_gpu_viewer.get_value(x);
        int32_t val   = original_val + delta;
        
        if (G_label[x] == 1) {
            const uint* __restrict__ nbrs = &neighbors[neighbors_offset[u]];
            if (Utils::binarySearch(x, nbrs, degrees[u])) {
                ++val;
            }
        }
        
        if (val < R) {
            atomicExch(&fail, 1);
            break; 
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && fail == 0) {
        atomicExch(&state_ok[stateID], 1);
    }
}

__global__ void X_insert(
    uint* keys, uint* values, uint* size, const uint capacity, const uint32_t* device_keys, const uint data_num) {
    const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const auto grid_size = blockDim.x * gridDim.x;

    for (uint i = idx; i < data_num; i += grid_size) {
        const uint key = device_keys[i];
        auto slot1     = Utils::HashTable::hash_func3a(key) % capacity;
        uint32_t old_k = atomicCAS(&keys[slot1], -1, key);
        if (old_k == -1) {
            values[slot1] = i;
            Utils::atomicAggInc(size);
            continue;
        }
        auto slot2 = Utils::HashTable::hash_func3b(key) % capacity;
        old_k      = atomicCAS(&keys[slot2], -1, key);
        if (old_k == -1) {
            values[slot2] = i;
            Utils::atomicAggInc(size);
            continue;
        }
        auto current_slot = ((Utils::HashTable::hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
        while (true) {
            uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
            if (old_k == -1) {
                values[current_slot] = i;
                Utils::atomicAggInc(size);
                break;
            }
            current_slot = (current_slot + 1) % capacity;
        }
    }
}

__global__ void enumerate_varY_to_get_varY_X_adjacentX(uint* degrees, uint* neighbors, uint* neighbors_offset,
    int* G_label, Partition_Device_viewer_v3 varY,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> X_hash_table, bool* varY_X_adjacent,
    uint X_size,  const uint start, const uint group_size) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    uint group_id = tid / group_size;
    uint group_inner_id = tid % group_size;
    uint group_num = grid_size / group_size;
    uint bias = varY.get_global_offset(start);
    
        
    if (group_id < varY.size[start]) {
        
        const uint end   = varY.size[start];
        for (uint idx = group_id; idx < end; idx += group_num) {
            const uint nodeIdx = bias + idx;
            const uint node = varY.partition[nodeIdx];
            uint offeset    = neighbors_offset[node];
            for (uint i = group_inner_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[offeset + i];
                if (G_label[neighbor]) {
                    auto x_idx = X_hash_table.get_value(neighbor);
                    if (x_idx != Utils::HashTable::emptyValue) {
                        varY_X_adjacent[nodeIdx * X_size + x_idx] = true;
                    }
                }
            }
        }
    }
}

__global__ void initialize_x_temp_count(
    uint* X, uint* x_temp_count, Utils::HashTable::cudaHashTable3_gpu_viewer<int32_t> g_temp, uint num_states, uint X_size) {
    const uint tid        = threadIdx.x + blockIdx.x * blockDim.x;
    uint X_inner_id = tid % X_size;
    uint state_id   = tid / X_size;
    uint X_element  = X[X_inner_id];
    if (state_id < num_states) {
        x_temp_count[state_id * X_size + X_inner_id] = g_temp.get_value(X_element);
    }
}

__global__ void initialize_x_temp_count(uint* x_temp_count, uint* original_x_temp_count, uint num_states, uint X_size) {
    const uint tid        = threadIdx.x + blockIdx.x * blockDim.x;
    uint X_inner_id = tid % X_size;
    uint state_id   = tid / X_size;
    if (state_id < num_states) {
        x_temp_count[state_id * X_size + X_inner_id] = original_x_temp_count[X_inner_id];
    }
}

__global__ void count_varY_support(bool* selected_allStates, bool* varY_X_adjacent, uint* x_temp_count, uint num_states,
    uint current_level, uint X_size, uint varY_size) {

    const uint tid  = threadIdx.x + blockIdx.x * blockDim.x;
    uint x_id = tid % X_size; 
    uint varY_id =
        (tid / X_size)
        % (current_level
            + 1); 
    uint state_id = tid / (X_size * (current_level + 1));
    if (state_id >= num_states) {
        return;
    }
    bool should_process =
        (varY_id < current_level && selected_allStates[state_id * varY_size + varY_id]) || (varY_id == current_level);
    if (should_process && varY_X_adjacent[varY_id * X_size + x_id]) {
        atomicAdd(&x_temp_count[state_id * X_size + x_id], 1);
    }
}

__global__ void enumerate_tempCount_to_get_extentions(uint* extentions, uint* x_temp_count, uint* selected_count,
    uint state_num, uint X_size, uint conY_size, uint curY_size, uint k, CompareOp compare_op) {
    const uint tid       = threadIdx.x + blockIdx.x * blockDim.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid;; idx += grid_size) {
        uint x_id     = idx % X_size;
        uint state_id = idx / X_size;
        if (state_id >= state_num) {
            break;
        }
        int threshold = conY_size + curY_size + selected_count[state_id] + 1 - k;
        if (extentions[state_id] == -1
            && Utils::get_judge(static_cast<int>(x_temp_count[state_id * X_size + x_id]), threshold, compare_op)) {
            atomicExch(&extentions[state_id], -2);
        }
        
    }
}

__global__ void count_if(uint* extentions, uint length_stateNum, uint* count) {
    const uint tid       = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < length_stateNum; idx += grid_size) {
        if (extentions[idx] == -1) { 
            extentions[idx] = Utils::atomicAggInc(count);
        }
    }
}

__global__ void count_if(
    uint* selected_counts, uint current_state_num, const int threshold, bool* state_bar, uint* counts, CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint i = tid; i < current_state_num; i += grid_size) {
        if (Utils::get_judge(static_cast<int>(selected_counts[i]), threshold, op)) {
            Utils::atomicAggInc(counts);
            state_bar[i] = true;
        }
    }
}



void ListBFS2_GPU(Partition_v3& X, Partition_v3& var_Y, Partition_v3& con_Y, Partition_v3& cur_Y, Partition_v3& candY_q,
    Partition_v3& ext_p, Partition_v3& ext_q, Utils::cpu_hashTable::cpu_hash_table& hashTable_G_temp,
    int* device_G_label, arrMapTable& g_state_viewer,
    bool* device_G_prune, uint*& device_degrees, uint*& device_neighbors, uint*& device_neighbors_offset,
    const uint theta, const uint k, cudaStream_t& stream) {
    vector<Layer> bfsQueue;
    bfsQueue.reserve(1);
    Layer initial;
    
    

    CUDA_ERROR_CHECK(cudaMalloc(&initial.selected_allStates, sizeof(bool) * var_Y.totalSize));
    CUDA_ERROR_CHECK(cudaMemset(initial.selected_allStates, false, sizeof(bool) * var_Y.totalSize));
    CUDA_ERROR_CHECK(cudaMalloc(&initial.excluded_allStates, sizeof(bool) * var_Y.totalSize));
    CUDA_ERROR_CHECK(cudaMemset(initial.excluded_allStates, false, sizeof(bool) * var_Y.totalSize));
    initial.level       = 0;
    initial.state_count = 1;
    bfsQueue.emplace_back(initial);
    

    bool* varY_X_adjacent; 
                           
    CUDA_ERROR_CHECK(cudaMalloc(&varY_X_adjacent, sizeof(bool) * var_Y.totalSize * X.totalSize));
    CUDA_ERROR_CHECK(cudaMemset(varY_X_adjacent, false, sizeof(bool) * var_Y.totalSize * X.totalSize));
    Utils::HashTable::cudaHashTable3 X_hashTable(X.totalSize, 0.4);

    X_insert<<<512, 512, 0, stream>>>(
        X_hashTable.keys, X_hashTable.values, X_hashTable.size, X_hashTable.capacity, X.partition, X.totalSize);

    auto X_hashTable_device_viewer = X_hashTable.get_viewer();
    auto g_temp_viewer             = hashTable_G_temp.get_viewer();
    auto varY_viewer               = var_Y.get_gpu_viewer();
    
    for (uint i = 0; i < Partition_v3::max_partition_num_; i++) {
        if (var_Y.host_size_.get()[i] > 0) {
            uint group_size = max((uint)pow(32, i), 32);
            const uint threadsNum = var_Y.host_size_.get()[i] * group_size;
            const uint grid_size = CALC_GRID_DIM(threadsNum, THREADS_PER_BLOCK);
            enumerate_varY_to_get_varY_X_adjacentX<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                device_neighbors_offset, device_G_label, varY_viewer, X_hashTable_device_viewer, varY_X_adjacent,
                X.totalSize, i, group_size);
        }
    }


    uint host_count     = 0;
    uint grid_num;
    uint* counts;
    CUDA_ERROR_CHECK(cudaMallocAsync(&counts, sizeof(uint), stream));
    CUDA_ERROR_CHECK(cudaMemsetAsync(counts, 0, sizeof(uint), stream));
    uint* original_x_temp_count;
    CUDA_ERROR_CHECK(cudaMallocAsync(&original_x_temp_count, sizeof(uint) * X.totalSize, stream));
    CUDA_ERROR_CHECK(cudaMemsetAsync(original_x_temp_count, 0, sizeof(uint) * X.totalSize, stream));
    grid_num = CALC_GRID_DIM(X.totalSize * 1, THREADS_PER_BLOCK);
    initialize_x_temp_count<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
        X.partition, original_x_temp_count, g_temp_viewer, 1, X.totalSize);


    uint* host_varY     = nullptr;
    bool* host_selected = nullptr;
    OutputFormat_class outputFormat;
    while (!bfsQueue.empty()) {
        auto current_state_num     = bfsQueue[0].state_count;
        auto current_level         = bfsQueue[0].level;
        Layer current_level_states = bfsQueue[0];
        bfsQueue.pop_back();
        uint* select_counts; 
        CUDA_ERROR_CHECK(cudaMalloc(&select_counts, sizeof(uint) * current_state_num));
        CUDA_ERROR_CHECK(cudaMemset(select_counts, 0, sizeof(uint) * current_state_num));
        grid_num = CALC_GRID_DIM(current_state_num * var_Y.totalSize, THREADS_PER_BLOCK);
        count_selected_nums<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
            current_level_states.selected_allStates, select_counts, var_Y.totalSize, current_state_num);
        
        
        
        
        
        
        
        
        if (current_level >= var_Y.totalSize) {
            uint *keys_allGTemp, *size_allGTemp;
            int32_t* values_allGTemp;
            uint* can_extends_device;
            uint* locks;
            bool* state_bar;
            CUDA_ERROR_CHECK(cudaMallocAsync(&can_extends_device, sizeof(uint) * current_state_num, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(can_extends_device, 0, sizeof(uint) * current_state_num, stream));
            CUDA_ERROR_CHECK(cudaMalloc(&state_bar, sizeof(bool) * current_state_num));
            CUDA_ERROR_CHECK(cudaMemset(state_bar, false, sizeof(bool) * current_state_num));
            
            
            grid_num = CALC_GRID_DIM(current_state_num, THREADS_PER_BLOCK);
            count_if<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(select_counts, current_state_num,
                static_cast<int>(theta - con_Y.totalSize - cur_Y.totalSize), state_bar, counts,
                CompareOp::GreaterEqual);
            uint satisfied_state_num;
            CUDA_ERROR_CHECK(
                cudaMemcpyAsync(&satisfied_state_num, counts, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(counts, 0, sizeof(uint), stream));
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            
            if (satisfied_state_num > 0) {
                for (uint i = 0; i < Partition_v3::max_partition_num_; i++) {
                    if (var_Y.host_size_.get()[i] > 0) {
                        const uint group_size = max((uint)pow(32, i), 32);
                        const uint threadsNum = var_Y.host_size_.get()[i] * group_size;
                        grid_num = CALC_GRID_DIM(threadsNum, THREADS_PER_BLOCK);
                        get_GLable_intersection_sizeX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                            varY_viewer, device_degrees, device_neighbors, device_neighbors_offset, device_G_label, counts, i, group_size);
                    }
                }
                uint intersection_size;
                CUDA_ERROR_CHECK(
                    cudaMemcpyAsync(&intersection_size, counts, sizeof(uint), cudaMemcpyDeviceToHost, stream));
                
                CUDA_ERROR_CHECK(cudaMemsetAsync(counts, 0, sizeof(uint), stream));
                CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
                intersection_size = static_cast<uint>(std::ceil(intersection_size * 2));
                CUDA_ERROR_CHECK(cudaMalloc(&keys_allGTemp, sizeof(uint) * intersection_size * satisfied_state_num));
                CUDA_ERROR_CHECK(cudaMalloc(&locks, sizeof(uint) * satisfied_state_num));
                CUDA_ERROR_CHECK(
                    cudaMalloc(&values_allGTemp, sizeof(int32_t) * intersection_size * satisfied_state_num));
                CUDA_ERROR_CHECK(cudaMalloc(&size_allGTemp, sizeof(uint) * satisfied_state_num));
                CUDA_ERROR_CHECK(cudaMemset(keys_allGTemp, -1, sizeof(uint) * intersection_size * satisfied_state_num));
                CUDA_ERROR_CHECK(cudaMemset(locks, -1, sizeof(uint) * satisfied_state_num));
                CUDA_ERROR_CHECK(
                    cudaMemset(values_allGTemp, 0, sizeof(int32_t) * intersection_size * satisfied_state_num));
                CUDA_ERROR_CHECK(cudaMemset(size_allGTemp, 0, sizeof(uint) * satisfied_state_num));
                Utils::MultiHash::cudaMultiHashTable_gpu_viewer delta_G_temps(
                    keys_allGTemp, values_allGTemp, size_allGTemp, locks, intersection_size, satisfied_state_num);
                for (uint i = 0; i < Partition_v3::max_partition_num_; i++) {
                    if (var_Y.host_size_.get()[i] > 0) {
                        uint group_size = max(32, static_cast<uint>(pow(32, i)));
                        const uint threadsNum = var_Y.host_size_.get()[i] * group_size;
                        const uint grid_size = CALC_GRID_DIM(threadsNum, THREADS_PER_BLOCK);
                        enumerate_varY_calGTempX<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(varY_viewer, device_degrees, current_state_num,
                            current_level_states, device_neighbors, device_neighbors_offset, device_G_label, delta_G_temps,
                            var_Y.totalSize, state_bar, i, group_size);
                    }
                }
                dim3 grid(current_state_num, var_Y.totalSize);
                auto block =
                    std::max(std::min(next_power_of_2(X.totalSize), static_cast<uint>(512)), static_cast<uint>(32));
                check_states_by_candidates_blockwise<<<grid, block, 0, stream>>>(var_Y.partition, var_Y.totalSize,
                    X.partition, X.totalSize, device_degrees, device_neighbors, device_neighbors_offset, device_G_label,
                    g_temp_viewer, delta_G_temps, current_level_states,
                    static_cast<int>(cur_Y.totalSize + con_Y.totalSize + 1) - static_cast<int>(k), select_counts,
                    current_state_num, can_extends_device, state_bar);
                
                
                grid_num = CALC_GRID_DIM(current_state_num, THREADS_PER_BLOCK);
                Utils::device_sum<<<current_state_num, THREADS_PER_BLOCK, 0, stream>>>(can_extends_device, counts, current_state_num);
                uint can_extends_count;
                CUDA_ERROR_CHECK(cudaMemcpyAsync(&can_extends_count, counts, sizeof(uint), cudaMemcpyDeviceToHost,
                stream)); CUDA_ERROR_CHECK(cudaMemsetAsync(counts, 0, sizeof(uint), stream));
                CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

                
                
                uint* can_extends = new uint[current_state_num];
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    can_extends, can_extends_device, sizeof(uint) * current_state_num, cudaMemcpyDeviceToHost, stream));
                Utils::HashTable::cudaHashTable3 hash_table(
                    cur_Y.totalSize + con_Y.totalSize + var_Y.totalSize, 0.4, stream);
                if (cur_Y.totalSize > 0) hash_table.insert(cur_Y.partition, cur_Y.totalSize, var_Y.totalSize + 1, stream);
                if (con_Y.totalSize > 0) hash_table.insert(con_Y.partition, con_Y.totalSize, var_Y.totalSize + 1, stream);
                if (var_Y.totalSize > 0) hash_table.insert(var_Y.partition, var_Y.totalSize, stream);
                
                
                
                
                
                

                for (uint _state_id = 0; _state_id < current_state_num; _state_id++) {
                    if (can_extends[_state_id] == 0) {
                        bool res = CheckMaximality(device_degrees, device_neighbors, device_neighbors_offset, delta_G_temps, device_G_prune, g_temp_viewer, g_state_viewer,
                            device_G_label, X, con_Y, cur_Y, var_Y, candY_q, ext_p, ext_q, _state_id,
                            current_level_states.selected_allStates, select_counts, hash_table, theta, k, stream);
                        
                        if (res) {
                            d_res_count++;
                            if (checkpoints.count(d_res_count)) {
                                TimePoint current_time = Clock::now();
                                
                                std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
                                double cost_time = elapsed.count();
                                
                                time_records.push_back({d_res_count, cost_time});
                                fmt::println("Output {}, cost Time: {} ms", d_res_count, cost_time);
                            }
                            if (outputFormat.empty()) {
                                uint *host_X = new uint[X.totalSize], *host_conY = new uint[con_Y.totalSize],
                                     *host_curY = new uint[cur_Y.totalSize];
                                host_varY       = new uint[var_Y.totalSize];
                                host_selected   = new bool[var_Y.totalSize];
                                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                                    host_X, X.partition, sizeof(uint) * X.totalSize, cudaMemcpyDeviceToHost, stream));
                                CUDA_ERROR_CHECK(cudaMemcpyAsync(host_conY, con_Y.partition,
                                    sizeof(uint) * con_Y.totalSize, cudaMemcpyDeviceToHost, stream));
                                CUDA_ERROR_CHECK(cudaMemcpyAsync(host_varY, var_Y.partition,
                                    sizeof(uint) * var_Y.totalSize, cudaMemcpyDeviceToHost, stream));
                                CUDA_ERROR_CHECK(cudaMemcpyAsync(host_curY, cur_Y.partition,
                                    sizeof(uint) * cur_Y.totalSize, cudaMemcpyDeviceToHost, stream));
                                CUDA_ERROR_CHECK(cudaMemcpyAsync(host_selected,
                                    current_level_states.selected_allStates + _state_id * var_Y.totalSize,
                                    sizeof(bool) * var_Y.totalSize, cudaMemcpyDeviceToHost, stream));
                                outputFormat.assign(
                                    host_X, X.totalSize, host_conY, con_Y.totalSize, host_curY, cur_Y.totalSize);
                                delete[] host_X, host_conY, host_curY;
                            } else {
                                CUDA_ERROR_CHECK(cudaMemcpyAsync(host_selected,
                                    current_level_states.selected_allStates + _state_id * var_Y.totalSize,
                                    sizeof(bool) * var_Y.totalSize, cudaMemcpyDeviceToHost, stream));
                            }
                            outputFormat.OutputResult(host_selected, host_varY, var_Y.totalSize, d_res_count);
                            if (d_res_count >= num)
                                return;
                        }
                    }
                }

                CUDA_ERROR_CHECK(cudaFreeAsync(counts, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(locks, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(keys_allGTemp, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(values_allGTemp, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(size_allGTemp, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(can_extends_device, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(state_bar, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(select_counts, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(current_level_states.selected_allStates, stream));
                CUDA_ERROR_CHECK(cudaFreeAsync(current_level_states.excluded_allStates, stream));

                break;
            }
        }
        else { 
            uint* x_temp_count;
            CUDA_ERROR_CHECK(cudaMallocAsync(&x_temp_count, sizeof(uint) * current_state_num * X.totalSize, stream));
            grid_num = CALC_GRID_DIM(X.totalSize * current_state_num, THREADS_PER_BLOCK);
            initialize_x_temp_count<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(x_temp_count, original_x_temp_count, current_state_num, X.totalSize);
            uint* extentions; 
                              
            CUDA_ERROR_CHECK(cudaMallocAsync(&extentions, sizeof(uint) * current_state_num, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(extentions, -1, sizeof(uint) * current_state_num, stream));
            uint totalThreadNum = (current_level + 1) * X.totalSize * current_state_num;
            grid_num      = CALC_GRID_DIM(totalThreadNum, THREADS_PER_BLOCK);
            count_varY_support<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(current_level_states.selected_allStates,
                varY_X_adjacent, x_temp_count, current_state_num, current_level, X.totalSize, var_Y.totalSize);

            
            
            
            
            
            
            
            totalThreadNum = X.totalSize * current_state_num;
            grid_num      = CALC_GRID_DIM(totalThreadNum, THREADS_PER_BLOCK);
            enumerate_tempCount_to_get_extentions<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(extentions, x_temp_count, select_counts,
                current_state_num, X.totalSize, con_Y.totalSize, cur_Y.totalSize, k, CompareOp::Less);
            totalThreadNum = current_state_num;
            grid_num      = CALC_GRID_DIM(totalThreadNum, THREADS_PER_BLOCK);
            count_if<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(extentions, current_state_num, counts);

            CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_count, counts, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            Layer new_layer;
            new_layer.state_count = current_state_num + host_count;
            new_layer.level       = current_level_states.level + 1;
            CUDA_ERROR_CHECK(cudaMallocAsync(
                &new_layer.selected_allStates, sizeof(bool) * var_Y.totalSize * new_layer.state_count, stream));
            CUDA_ERROR_CHECK(cudaMallocAsync(
                &new_layer.excluded_allStates, sizeof(bool) * var_Y.totalSize * new_layer.state_count, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(
                new_layer.selected_allStates, false, sizeof(bool) * var_Y.totalSize * new_layer.state_count, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(
                new_layer.excluded_allStates, false, sizeof(bool) * var_Y.totalSize * new_layer.state_count, stream));
            totalThreadNum = var_Y.totalSize * current_state_num * 2;
            grid_num      = CALC_GRID_DIM(totalThreadNum, THREADS_PER_BLOCK);
            init_new_layer<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(new_layer.selected_allStates, new_layer.excluded_allStates,
                current_level_states.selected_allStates, current_level_states.excluded_allStates, extentions,
                current_state_num, var_Y.totalSize, current_level, counts);
            
            
            
            
            
            
            
            
            
            


            CUDA_ERROR_CHECK(cudaFreeAsync(x_temp_count, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(select_counts, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(extentions, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(current_level_states.excluded_allStates, stream));
            CUDA_ERROR_CHECK(cudaFreeAsync(current_level_states.selected_allStates, stream));
            
            bfsQueue.emplace_back(new_layer);
            CUDA_ERROR_CHECK(cudaMemsetAsync(counts, 0, sizeof(uint), stream));
        }
    }
    CUDA_ERROR_CHECK(cudaFreeAsync(varY_X_adjacent, stream));

    if (host_varY) {
        delete[] host_varY;
    }
    if (host_selected) {
        delete[] host_selected;
    }
}

#endif 
