



#ifndef FOR_GENERATE_CUH
#define FOR_GENERATE_CUH
#include <for_BFS.cuh>
#include <for_special.cuh>
#include <for_test.h>
#include <output.cuh>
#include <hashTable.cuh>
#include <partition.cuh>
#include <tmp_load.h>
__global__ void invalidX_insert_before(uint* X_partition, uint X_length, uint* degrees, uint* invalidX_size,
    const int threshold, arrMapTable g_state_viewer) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;

    
    for (uint idx = tid; idx < X_length; idx += grid_size) {
        uint node       = X_partition[idx];
        const int value = g_state_viewer.get_value(node);
        
        if (value < threshold) {
            uint i = Partition_Device_viewer_v3::get_partition_idx_from_degree(degrees[node]);
            atomicAdd(&invalidX_size[i], 1);
        }
    }
}

__global__ void invalidX_insert_after(uint* X_partition, uint X_length, uint* degrees, Partition_Device_viewer_v3 invalidX,
    int threshold, arrMapTable g_state_viewer) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < X_length; idx += grid_size) {
        uint node = X_partition[idx];
        int value = g_state_viewer.get_value(node);
        if (value < threshold)
            invalidX.insert_node_to_partition_from_degree(node, degrees[node]);
    }
}



__global__ void enumerateNX_insert_conY_and_varY_before(uint* N_X_partition, uint N_X_length, uint* varY_size,
    uint* conY_size, uint* degrees, uint invalidX_length,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> intersection_of_allInvalidX,
    const int* G_label) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < N_X_length; idx += grid_size) {
        const uint node        = N_X_partition[idx];
        const uint i           = Partition_Device_viewer_v3::get_partition_idx_from_degree(degrees[node]);
        const uint32_t value1  = intersection_of_allInvalidX.get_value(node);
        if (value1 == invalidX_length - 1 && G_label[node]) {
            atomicAdd(&conY_size[i], 1);
        } else {
            atomicAdd(&varY_size[i], 1);
        }
    }
}

__global__ void enumerateNX_insert_conY_and_varY_after(Partition_Device_viewer_v3 N_X, uint N_X_length,
    Partition_Device_viewer_v3 varY, Partition_Device_viewer_v3 conY, uint* degrees, uint invalidX_length,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> intersection_of_allInvalidX,
    const int* G_label) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < N_X_length; idx += grid_size) {
        const uint node = N_X.partition[idx];
        uint32_t value1 = intersection_of_allInvalidX.get_value(node);
        if (value1 == invalidX_length - 1 && G_label[node])
            conY.insert_node_to_partition_from_degree(node, degrees[node]);
        else
            varY.insert_node_to_partition_from_degree(node, degrees[node]);
    }
}



__global__ void enumerateConY_insertGtemp_GpRruneX(Partition_Device_viewer_v3 conY, uint* degrees,
    uint* neighbors, uint* neighbors_offset, int* G_label,
    HashTable_viewer g_temp_viewer, bool* device_G_prune, const uint start, const uint group_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size  = blockDim.x * gridDim.x;
    const uint warp_id = tid / group_size;
    const uint lane_id = tid % group_size;
    const uint warp_num = grid_size / group_size;
    uint bias = 0;
    if (start > 0) {
        uint startIdx = start;
        while (startIdx)
            bias += conY.size[-- startIdx];
    }
    if (warp_id < conY.size[start]) {
        const uint end   = conY.size[start];
        for (uint idx = warp_id; idx < end; idx += warp_num) {
            const uint nodeIdx = bias + idx;
            const uint node = conY.partition[nodeIdx];
            uint offset    = neighbors_offset[node];
            for (uint i = lane_id; i < degrees[node]; i += group_size) {
                const uint neighbor = neighbors[offset + i];
                if (G_label[neighbor]) {
                    assert(g_temp_viewer.add(neighbor, 1));
                }
            }
            device_G_prune[node] = true;
        }
    }
}


__global__ void intersection_of_CandQ_GLabel0(Partition_Device_viewer_v3 Cand_q, uint* degrees,
    uint* neighbors, uint* neighbors_offset, uint* intersections,
    const int *G_label) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    for (auto nodeIdx = tid; nodeIdx < Cand_q.size[0]; nodeIdx += grid_size) {
        const uint node = Cand_q.partition[nodeIdx];
        for (uint i = 0; i < degrees[node]; i++) {
            const uint neighbor = neighbors[neighbors_offset[node] + i];
            if (G_label[neighbor])
                intersections[nodeIdx]++;
        }
    }
}

__global__ void intersection_of_CandQ_GLabelX(Partition_Device_viewer_v3 Cand_q, uint* degrees, const uint* neighbors,
    const uint* neighbors_offset, uint* intersections,
    const int *G_label, const uint start, const uint group_size) {
    const uint tid        = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size  = blockDim.x * gridDim.x;
    uint group_id = tid / group_size;
    uint group_inner_id = tid % group_size;
    uint group_num = grid_size / group_size;
    uint local_count = 0;
    const uint bias = Cand_q.get_global_offset(start);
    const uint end   = Cand_q.size[start];
    for (uint idx = group_id; idx < end; idx += group_num) {
        const uint nodeIdx = bias + idx;
        const uint node = Cand_q.partition[nodeIdx];
        uint offeset    = neighbors_offset[node];
        for (uint i = group_inner_id; i < degrees[node]; i += group_size) {
            const uint neighbor = neighbors[offeset + i];
            if (G_label[neighbor]) local_count++;
        }
        
        
        
        
        
        atomicAdd(&intersections[nodeIdx], local_count);
        local_count = 0;
    }
}

__global__ void insert_extQ_before(
    uint* extQ_size, uint* cand_q_size, uint Cand_q_length, uint* intersections, int threshold, CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < Cand_q_length; idx += grid_size) {
        if (Utils::get_judge(static_cast<int>(intersections[idx]), threshold, op)) {
            const uint i = Partition_Device_viewer_v3::get_partition_idx_from_global_index(cand_q_size, idx);
            atomicAdd(&extQ_size[i], 1);
        }
    }
}

__global__ void insert_extQ_after(Partition_Device_viewer_v3 extQ, Partition_Device_viewer_v3 Cand_q, uint Cand_q_length,
    uint* intersections, int threshold, CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < Cand_q_length; idx += grid_size) {
        if (Utils::get_judge(static_cast<int>(intersections[idx]), threshold, op)) {
            int partition_idx = Partition_Device_viewer_v3::get_partition_idx_from_global_index(Cand_q.size, idx);
            extQ.insert_node_to_partition_from_partitionIdx(Cand_q.partition[idx], partition_idx);
        }
    }
}

__global__ void enumerateX_setZero(
    uint* X, uint size, HashTable_viewer g_temp_viewer) {
    const uint tid       = threadIdx.x + blockIdx.x * blockDim.x;
    uint grid_size = blockDim.x * gridDim.x;
    uint local_count = 0;
    for (uint i = tid; i < size; i += grid_size) {
        g_temp_viewer.delete_element(X[i]);
        local_count ++;
    }
    
    
}

__global__ void invalidX_neigherbors_intersection_X(uint* degrees, uint* neighbors, uint* neighbors_offset,
    Partition_Device_viewer_v3 invalidX, Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> all_intersection,
     const uint start, const uint group_size) {
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id = tid / group_size;
    const uint inner_id = tid % group_size;
    const uint group_num = grid_size / group_size;

    const uint bias = invalidX.get_global_offset(start);;
    for (uint idx = group_id; idx < invalidX.size[start]; idx += group_num) {
        const uint nodeIdx = idx + bias;
        const uint node = invalidX.partition[nodeIdx];
        const uint* neighbors_of_node = neighbors + neighbors_offset[node];
        for (uint neighborIdx = inner_id; neighborIdx < degrees[node]; neighborIdx += group_size) {
            uint neighbor = neighbors_of_node[neighborIdx];
            all_intersection.add(neighbor, 1);
        }
    }
}

__global__ void X_neigherbors_intersection_Cand_exts_warp(
    const uint* __restrict__ degrees,
    const uint* __restrict__ neighbors,
    const uint* __restrict__ neighbors_offset,
    Partition_Device_viewer_v3 Cand_exts,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> X_neighbors,
    const uint X_degree, uint* is_exist,
    const uint start 
    ) {
    if (*is_exist) return;
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id = tid / 32;
    const uint inner_id = tid % 32;
    const uint group_num = grid_size / 32;

    const uint bias = Cand_exts.get_global_offset(start);
    for (uint idx = group_id; idx < Cand_exts.size[start]; idx += group_num) {
        if (*is_exist) break;
        const uint nodeIdx = idx + bias;
        const uint node = Cand_exts.partition[nodeIdx];
        const uint degree = degrees[node];
        if (degree < X_degree) {
            continue;
        }
        const uint* neighbors_of_node = neighbors + neighbors_offset[node];
        uint local_count = 0;
        for (uint neighborIdx = inner_id; neighborIdx < degree; neighborIdx += 32) {
            uint neighbor = neighbors_of_node[neighborIdx];
            if (X_neighbors.find_key(neighbor))
                local_count ++;
        }
        local_count = __reduce_add_sync(0xFFFFFFFF, local_count);
        if (inner_id == 0 && local_count == X_degree)
            atomicExch(is_exist, 1);
    }
}

template <int BLOCK_DIM>
__global__ void X_neigherbors_intersection_Cand_exts_Block_CUB(
    const uint* __restrict__ degrees,
    const uint* __restrict__ neighbors,
    const uint* __restrict__ neighbors_offset,
    Partition_Device_viewer_v3 Cand_exts,
    Utils::HashTable::cudaHashTable3_gpu_viewer<uint32_t> X_neighbors,
    const uint X_degree,
    uint* is_exist,
    const uint start 
) {
    if (*is_exist) return;
    using BlockReduce = cub::BlockReduce<uint, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const uint idx_in_partition = blockIdx.x;
    if (idx_in_partition >= Cand_exts.size[start]) return;

    const uint bias = Cand_exts.get_global_offset(start);
    const uint nodeIdx = idx_in_partition + bias;
    const uint node = Cand_exts.partition[nodeIdx];
    const uint node_deg = degrees[node];

    if (node_deg < X_degree) return;

    const uint* neighbors_of_node = neighbors + neighbors_offset[node];
    uint local_count = 0;

    
    for (uint i = threadIdx.x; i < node_deg; i += BLOCK_DIM) {
        uint neighbor = neighbors_of_node[i];
        if (X_neighbors.find_key(neighbor)) {
            local_count++;
        }
    }

    
    
    
    uint total_intersection = BlockReduce(temp_storage).Sum(local_count);
    
    if (threadIdx.x == 0) {
        if (total_intersection == X_degree) {
            *is_exist = 1;
        }
    }
}

tuple<bool, bool> Generate_GPU(std::shared_ptr<Partition_v3>& X, std::shared_ptr<Partition_v3>& N_X, std::shared_ptr<Partition_v3>& Cand_exts, std::shared_ptr<Partition_v3>& Cand_q,
    Utils::cpu_hashTable::cpu_hash_table& hashTable_G_temp, int*& device_G_label,
    Utils::cpu_hashTable::cpu_hash_table& hashTable_G_state, arrMapTable &device_arrGstate, bool*& device_G_prune, uint*& device_degrees,
    uint*& device_neighbors, uint*& device_neighbors_offset, vector<uint> host_neighbors_offset, const uint graph_size, const uint bipartite_index, const uint theta, const uint k, const uint component_length, const uint temp_node, const uint temp_degree, cudaStream_t& stream) {
    






    
    auto g_temp_viewer  = hashTable_G_temp.get_viewer();
    auto g_state_viewer = device_arrGstate;
    bool after_prune_y = false, after_prune_x = false;
    if (d_res_count >= num) {
        return {after_prune_x, after_prune_y};
    }
    Partition_v3 invalid_X;
    uint grid_dim = CALC_GRID_DIM(X->totalSize, THREADS_PER_BLOCK);
    invalidX_insert_before<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
        X->partition, X->totalSize, device_degrees, invalid_X.size_.get(), static_cast<int>(N_X->totalSize) - static_cast<int>(k), g_state_viewer);


    invalid_X.malloc_partition(stream);
    if (invalid_X.totalSize == 1 && X->totalSize == 1) {
        Utils::HashTable::cudaHashTable3 X_neighbors(temp_degree, 0.3, stream);
        X_neighbors.insert(device_neighbors + host_neighbors_offset[temp_node], temp_degree, 1, stream);
        uint *d_is_exits, is_exits;
        CUDA_ERROR_CHECK(cudaMallocAsync((void**)&d_is_exits, sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(d_is_exits, 0, sizeof(uint), stream));
        for (uint i = 0; i < Partition_v3::max_partition_num_; i ++) {
            if (X->host_size_.get()[i] != 0) {
                for(; i < Partition_v3::max_partition_num_; i ++) {
                    if (Cand_exts->host_size_.get()[i] == 0) continue;
                    if (i < 2) {
                        grid_dim = CALC_GRID_DIM(Cand_exts->host_size_.get()[i] * 32, THREADS_PER_BLOCK);
                        X_neigherbors_intersection_Cand_exts_warp<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
                            device_degrees, device_neighbors, device_neighbors_offset,
                            Cand_exts->get_gpu_viewer(), X_neighbors.get_viewer(), temp_degree, d_is_exits, i);
                    }
                    else {
                        grid_dim = Cand_exts->host_size_.get()[i];
                        X_neigherbors_intersection_Cand_exts_Block_CUB<THREADS_PER_BLOCK><<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors, device_neighbors_offset,
                            Cand_exts->get_gpu_viewer(), X_neighbors.get_viewer(), temp_degree, d_is_exits, i);
                    }
                }
            }
        }
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&is_exits, d_is_exits, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        if (is_exits == 1 && static_cast<int>(component_length) >= static_cast<int>(X->totalSize) + 1 - (int)k) {
            return {after_prune_x, after_prune_y};
        }
    }

    Partition_v3 constant_Y;
    std::shared_ptr<Partition_v3> var_Y = nullptr;
    Partition_v3 ext_q;
    Partition_v3 cur_Y;
    Partition_v3 candY_p;
    {
        if (invalid_X.totalSize > 0) {
            var_Y = std::make_shared<Partition_v3>();
            
            auto invalidX_device_viewer = invalid_X.get_gpu_viewer();
            invalidX_insert_after<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
                X->partition, X->totalSize, device_degrees, invalidX_device_viewer, N_X->totalSize - k, g_state_viewer);
            int partition_idx = 0;
            while (partition_idx < Partition_v3::max_partition_num_ && invalid_X.host_size_.get()[partition_idx] == 0) {
                partition_idx++;
            }
            int maxDegree_thisPartition = pow(32, partition_idx + 1);
            Utils::HashTable::cudaHashTable3 neighbors_of_node(maxDegree_thisPartition, 0.4, stream);
            for (int idx = partition_idx; idx < Partition_v3::max_partition_num_; idx++) {
                if (invalid_X.host_size_.get()[idx] > 0) {
                    const uint group_size = max(32, (uint)pow(32, partition_idx));
                    const uint threadNum = invalid_X.host_size_.get()[idx] * group_size;
                    const uint grid_size = CALC_GRID_DIM(threadNum, THREADS_PER_BLOCK);
                    invalidX_neigherbors_intersection_X<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                        device_neighbors_offset, invalidX_device_viewer, neighbors_of_node.get_viewer(),
                        idx, group_size);
                }
            }
            grid_dim = CALC_GRID_DIM(N_X->totalSize, THREADS_PER_BLOCK);
            enumerateNX_insert_conY_and_varY_before<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(N_X->partition, N_X->totalSize, var_Y->size_.get(),
                constant_Y.size_.get(), device_degrees, invalid_X.totalSize, neighbors_of_node.get_viewer(), device_G_label);
            constant_Y.malloc_partition(stream);
            var_Y->malloc_partition(stream);
            auto conY_viewer = constant_Y.get_gpu_viewer();
            enumerateNX_insert_conY_and_varY_after<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(N_X->get_gpu_viewer(), N_X->totalSize,
                var_Y->get_gpu_viewer(), conY_viewer, device_degrees, invalid_X.totalSize,
                neighbors_of_node.get_viewer(), device_G_label);
            for (uint i = 0; i < Partition_v3::max_partition_num_; i++) {
                if (constant_Y.host_size_.get()[i] > 0) {
                    uint group_size = max((uint)pow(32, i), 32);
                    uint threadNum = constant_Y.host_size_.get()[i] * group_size;
                    const uint grid_size = CALC_GRID_DIM(threadNum, THREADS_PER_BLOCK);
                    enumerateConY_insertGtemp_GpRruneX<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(conY_viewer, device_degrees, device_neighbors,
                        device_neighbors_offset, device_G_label, g_temp_viewer, device_G_prune, i, group_size);
                }
            }
            after_prune_y = not constant_Y.empty();
        }
        else {
            var_Y = N_X;
        }
    }
    if (Cand_q->totalSize > 0) {
        uint* intersections;
        CUDA_ERROR_CHECK(cudaMallocAsync(&intersections, sizeof(uint) * Cand_q->totalSize, stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(intersections, 0, sizeof(uint) * Cand_q->totalSize, stream));
        for (uint i = 0; i < Partition_v3::max_partition_num_; i++) {
            if (Cand_q->host_size_.get()[i] > 0) {
                uint group_size = max((uint)pow(32, i), 32);
                const uint threadNum = Cand_q->host_size_.get()[i] * group_size;
                const uint grid_size = CALC_GRID_DIM(threadNum, THREADS_PER_BLOCK);
                intersection_of_CandQ_GLabelX<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(Cand_q->get_gpu_viewer(), device_degrees,
                    device_neighbors, device_neighbors_offset, intersections, device_G_label, i,  group_size);
            }
        }
        uint grid_dim = CALC_GRID_DIM(Cand_q->totalSize, THREADS_PER_BLOCK);
        insert_extQ_before<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(ext_q.size_.get(), Cand_q->size_.get(), Cand_q->totalSize, intersections,
            static_cast<int>(theta) - static_cast<int>(k), CompareOp::GreaterEqual);
        ext_q.malloc_partition(stream);
        if (ext_q.totalSize > 0)
            insert_extQ_after<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(ext_q.get_gpu_viewer(), Cand_q->get_gpu_viewer(),
            Cand_q->totalSize, intersections, static_cast<int>(theta) - static_cast<int>(k),
            CompareOp::GreaterEqual);
        CUDA_ERROR_CHECK(cudaFreeAsync(intersections, stream));
    }

    
        if (invalid_X.totalSize == 0 && (constant_Y.totalSize + var_Y->totalSize >= theta)) {
        Utils::HashTable::cudaHashTable3 hash_table(
            cur_Y.totalSize + constant_Y.totalSize + var_Y->totalSize, 0.4, stream);
        if (constant_Y.totalSize) {
            hash_table.insert(constant_Y.partition, constant_Y.totalSize, var_Y->totalSize + 1, stream);
        }
        if (var_Y->totalSize) {
            hash_table.insert(var_Y->partition, var_Y->totalSize, stream);
        }
        if (var_Y->totalSize == 0)
            return {after_prune_x, after_prune_y};
        OutputFormat_class outputFormat;
        fmt::print("CHECK");
        cout << endl;
        bool res = CheckMaximality_for_Gnerate(device_degrees, device_neighbors, device_neighbors_offset,
            device_G_prune, g_state_viewer, hash_table, *X, constant_Y, *var_Y, *Cand_exts, ext_q, theta, k, stream);
        after_prune_x = true;
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
                uint *host_X = new uint[X->totalSize], *host_conY = new uint[constant_Y.totalSize],
                     *host_varY = new uint[var_Y->totalSize];
                CUDA_ERROR_CHECK(
                    cudaMemcpyAsync(host_X, X->partition, sizeof(uint) * X->totalSize, cudaMemcpyDeviceToHost, stream));
                CUDA_ERROR_CHECK(cudaMemcpyAsync(host_conY, constant_Y.partition, sizeof(uint) * constant_Y.totalSize,
                    cudaMemcpyDeviceToHost, stream));
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    host_varY, var_Y->partition, sizeof(uint) * var_Y->totalSize, cudaMemcpyDeviceToHost, stream));
                outputFormat.assign(host_X, X->totalSize, host_conY, constant_Y.totalSize, host_varY, var_Y->totalSize);
                delete[] host_X, host_conY, host_varY;
            }
            outputFormat.OutputResult(d_res_count);
            
            if (d_res_count >= num)
                return {after_prune_x, after_prune_y};
        }
    }
        else if (invalid_X.totalSize == 1 && var_Y->totalSize > 0) {
            if (static_cast<int>(constant_Y.totalSize) >= static_cast<int>(theta - k)
                && constant_Y.totalSize + var_Y->totalSize >= theta) {
                fmt::print("SPECIAL");
                cout << endl;
                if (static_cast<int>(X->totalSize - k) + 1 <= 0) {
                    fmt::print("optimized");
                    cout << endl;
                    Special_GPU_optimized(*X, *var_Y, constant_Y, *Cand_exts, ext_q, g_state_viewer, device_G_prune, device_degrees,
                    device_neighbors, device_neighbors_offset, graph_size - bipartite_index, theta, k, OutputResults, stream);
                }
                else {
                    fmt::print("original");
                    cout << endl;
                    Special_GPU(*X, *var_Y, constant_Y, *Cand_exts, ext_q, g_state_viewer, device_G_prune, device_degrees,
                    device_neighbors, device_neighbors_offset, component_length, theta, k, OutputResults, stream);
                }

                after_prune_x = true;
            }
        
        }
        else if (var_Y->totalSize > 0) {
            fmt::print("LISTBFS2");
            cout << endl;
            ListBFS2_GPU(*X, *var_Y, constant_Y, cur_Y, candY_p, *Cand_exts, ext_q, hashTable_G_temp, device_G_label,
                g_state_viewer, device_G_prune, device_degrees, device_neighbors, device_neighbors_offset, theta, k, stream);
            after_prune_x = true;
        }
    
    dim3 blockDim(max(min(256, next_power_of_2(X->totalSize)), 32));
    dim3 gridDim((X->totalSize + blockDim.x - 1) / blockDim.x);
    enumerateX_setZero<<<gridDim, blockDim, 0, stream>>>(X->partition, X->totalSize, g_temp_viewer);
    return {after_prune_x, after_prune_y};
}

#endif 
