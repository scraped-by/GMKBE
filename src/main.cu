#include <algorithm>
#include <args.hxx>
#include <cassert>
#include <cstring>
#include <cub/cub.cuh>
#include <cuda/barrier>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <for_BFS.cuh>
#include <for_checkmaximality.cuh>
#include <for_generate.cuh>
#include <fstream>
#include <gpu_utils.cuh>
#include <graph.h>
#include <hashTable.cuh>
#include <iostream>
#include <kcore_24_10_08/common.h>
#include <kcore_24_10_08/kcore.h>
#include <memory>
#include <numeric>
#include <output.cuh>
#include <partition.cuh>
#include <queue>
#include <ranges>
#include <sstream>
#include <string>
#include <tmp_load.h>
#include <utils.h>
#include <vector>

uint res_count      = 0;
int id              = 0;
int State_tmp_count = 0;
bool verbose        = false;



















































struct State {
    uint parentLayer_candidate_node = -1;

    
    std::shared_ptr<Partition_v3> X                     = nullptr;
    std::shared_ptr<Partition_v3> N_X                   = nullptr;
    std::shared_ptr<Partition_v3> Cand_exts             = nullptr;
    std::shared_ptr<Partition_v3> parentLayer_Component = nullptr;
    std::shared_ptr<Partition_v3>Cand_q                 = nullptr;


    bool isRecurveive{}, buildNew = true, inWhile = false;
    int flag = 2;
    
    std::reference_wrapper<std::vector<uint>> host_Cand_q;

    
    std::vector<uint> node_cache; 
    int cache_idx = 0;            
    static const constexpr uint BATCH_SIZE = 512;

    __host__ __device__ State() = default;

    
    __host__ State(std::shared_ptr<Partition_v3> x, std::shared_ptr<Partition_v3> n_x,
                   std::shared_ptr<Partition_v3> cand_exts, std::reference_wrapper<std::vector<uint>> h_cand_q, std::shared_ptr<Partition_v3> cand_q,
                   bool is_recurveive = false)
            : X(std::move(x)), N_X(std::move(n_x)), Cand_exts(std::move(cand_exts)), host_Cand_q(h_cand_q), Cand_q(std::move(cand_q)),
              isRecurveive(is_recurveive) {
        
    }


    
    
    
    __host__ uint fetch_next_node_batch(cudaStream_t stream = 0) {
        
        if (cache_idx >= node_cache.size()) {
            uint remaining_size = Cand_exts->totalSize;
            if (remaining_size == 0) {
                
                return (uint)-1;
            }

            
            uint batch = std::min(BATCH_SIZE, remaining_size);

            
            node_cache.resize(batch);

            
            
            
            CUDA_ERROR_CHECK(cudaMemcpyAsync(node_cache.data(),
                                             Cand_exts->partition,
                                             batch * sizeof(uint),
                                             cudaMemcpyDeviceToHost,
                                             stream));

            
            
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

            
            cache_idx = 0;
        }

        
        uint host_node = node_cache[cache_idx++];

        
        
        
        
        
        Cand_exts->advance_and_update(stream);

        return host_node;
    }
};

struct newState {
    
    std::shared_ptr<Partition_v3> new_X         = nullptr;
    std::shared_ptr<Partition_v3> new_N_X       = nullptr;
    std::shared_ptr<Partition_v3> new_Cand_exts = nullptr;
    std::shared_ptr<CandidateQueue_v3> new_Cand_q = nullptr;

    std::vector<uint> host_new_Cand_q;

    __host__ newState() = default;

    
    __host__ ~newState() = default;

    
    __host__ void initialize(const uint X_size, uint* Cand_exts_ptr) {
        
        
        new_X         = std::make_shared<Partition_v3>(X_size);
        new_N_X       = std::make_shared<Partition_v3>(); 
        new_Cand_exts = std::make_shared<Partition_v3>();
        new_Cand_q = std::make_shared<CandidateQueue_v3>(Cand_exts_ptr, Partition_v3::max_partition_num_);
    }
};

__global__ void insert_G_state(
        uint* tmp_node, uint* degrees, uint* neighbors, uint* neighbors_offset, arrMapTable G_state) {
    const uint tid                 = threadIdx.x + blockIdx.x * blockDim.x;
    const uint grid_size           = blockDim.x * gridDim.x;
    const uint* tmp_node_neighbors = neighbors + neighbors_offset[tmp_node[0]];
    if (tid == 0) {
        assert(G_state.set_value(tmp_node[0], degrees[tmp_node[0]]));
        
    }
    for (uint i = tid; i < degrees[tmp_node[0]]; i += grid_size) {
        const uint neighbor = tmp_node_neighbors[i];
        assert(G_state.add(neighbor, 1));
        
    }
}


__global__ void insert_new_N_X_pre(const uint* degrees, arrMapTable G_state, int* G_label,
                                   Partition_Device_viewer_v3 new_N_X, const uint bipartite_index, const uint Graph_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint node = bipartite_index + tid; node < Graph_size; node += grid_size) {
        const int value = G_state.get_value(node);
        
        
        if (value > 0) {
            const uint partition_idx = new_N_X.get_partition_idx_from_degree(degrees[node]);
            atomicAdd(&new_N_X.size[partition_idx], 1);
        } else {
            G_label[node] = 0;
            
        }
    }
}


__global__ void insert_new_N_X(const uint* degrees, arrMapTable G_state, Partition_Device_viewer_v3 new_N_X,
                               const uint bipartite_index, const uint Graph_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint node = bipartite_index + tid; node < Graph_size; node += grid_size) {
        int value = G_state.get_value(node);
        if (value > 0) {
            new_N_X.insert_node_to_partition_from_degree(node, degrees[node]);
        }
    }
}


__global__ void intersection_with_oneNode_and_CandExts_sharedMem(const uint* degrees, const uint* neighbors,
                                                                 const uint* neighbors_offset, int* intersection, const int* G_label, Partition_Device_viewer_v3 partitions,
                                                                 const uint* tmp_node, const int threshold, const CompareOp op, const uint start, const uint group_size) {
    const uint tid                    = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size              = blockDim.x * gridDim.x;
    const uint group_id               = tid / group_size;
    const uint inner_id               = tid % group_size;
    const uint group_num              = grid_size / group_size;
    const uint the_node               = tmp_node[0];
    const uint degree_of_tmp_node     = degrees[the_node];
    const uint* neighbors_of_tmp_node = neighbors + neighbors_offset[the_node];
    extern __shared__ uint shared_neighbor_of_tmp_node[];
    for (uint idx = threadIdx.x; idx < degree_of_tmp_node; idx += blockDim.x) {
        shared_neighbor_of_tmp_node[idx] = neighbors_of_tmp_node[idx];
    }
    __syncthreads();
    uint global_offset = partitions.get_global_offset(start);
    for (uint idx = group_id; idx < partitions.size[start]; idx += group_num) {
        const uint nodeIdx    = idx + global_offset;
        const uint node       = partitions.partition[nodeIdx];
        const uint nodeDegree = degrees[node];
        if (Utils::get_judge(nodeDegree, threshold, op)) {
            continue;
        }
        const uint* neighbors_ = neighbors + neighbors_offset[node];
        int local_count        = 0;
        for (uint neighborIdx = inner_id; neighborIdx < nodeDegree; neighborIdx += group_size) {
            const uint neighbor    = neighbors_[neighborIdx];
            const int gLable_value = G_label[neighbor];
            bool found;
            if (degree_of_tmp_node <= 32) {
                found = Utils::sequentialSearch_unrolled(neighbor, shared_neighbor_of_tmp_node, degree_of_tmp_node);
            } else {
                found = Utils::binarySearch(neighbor, shared_neighbor_of_tmp_node, degree_of_tmp_node);
            }
            if (found && gLable_value == 1) {
                local_count++;
            }
        }
        atomicAdd(&intersection[nodeIdx], local_count);
        
        
        
        
        
        
        
        
    }
}


__global__ void intersection_with_oneNode_and_CandExts(const uint* degrees, const uint* neighbors,
                                                       const uint* neighbors_offset, int* intersection, int* G_label, Partition_Device_viewer_v3 partitions,
                                                       const uint* tmp_node, const int threshold, const CompareOp op, const uint start, const uint group_size) {
    const uint tid                    = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size              = blockDim.x * gridDim.x;
    const uint group_id               = tid / group_size;
    const uint inner_id               = tid % group_size;
    const uint group_num              = grid_size / group_size;
    const uint the_node               = tmp_node[0];
    const uint degree_of_tmp_node     = degrees[the_node];
    const uint* neighbors_of_tmp_node = neighbors + neighbors_offset[the_node];
    const uint global_offset          = partitions.get_global_offset(start);
    for (uint idx = group_id; idx < partitions.size[start]; idx += group_num) {
        const uint nodeIdx    = idx + global_offset;
        const uint node       = partitions.partition[nodeIdx];
        const uint nodeDegree = degrees[node];
        if (Utils::get_judge(static_cast<int>(nodeDegree), threshold, op)) {
            continue;
        }
        const uint* neighbors_ = neighbors + neighbors_offset[node];
        int local_count        = 0;
        for (uint neighborIdx = inner_id; neighborIdx < nodeDegree; neighborIdx += group_size) {
            const uint neighbor    = neighbors_[neighborIdx];
            const int gLabel_value = G_label[neighbor];
            bool found;
            if (degree_of_tmp_node < 16) {
                found = Utils::sequentialSearch_unrolled(neighbor, neighbors_of_tmp_node, degree_of_tmp_node);
            } else {
                found = Utils::binarySearch(neighbor, neighbors_of_tmp_node, degree_of_tmp_node);
            }
            if (found && gLabel_value == 1) {
                local_count++;
            }
        }
        atomicAdd(&intersection[nodeIdx], local_count);
    }
}

__global__ void intersection_with_oneNode_and_X_sharedMem(const uint* degrees, const uint* neighbors,
                                                          const uint* neighbors_offset, uint* intersection, Partition_Device_viewer_v3 partitions, const uint* tmp_node,
                                                          const uint start, const uint group_size) {
    const uint tid                    = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size              = blockDim.x * gridDim.x;
    const uint group_id               = tid / group_size;
    const uint inner_id               = tid % group_size;
    const uint group_num              = grid_size / group_size;
    const uint the_node               = tmp_node[0];
    const uint degree_of_tmp_node     = degrees[the_node];
    const uint* neighbors_of_tmp_node = neighbors + neighbors_offset[the_node];
    extern __shared__ uint shared_neighbor_of_tmp_node[];
    for (uint idx = threadIdx.x; idx < degree_of_tmp_node; idx += blockDim.x) {
        shared_neighbor_of_tmp_node[idx] = neighbors_of_tmp_node[idx];
    }
    __syncthreads();
    uint global_offset = partitions.get_global_offset(start);
    for (uint idx = group_id; idx < partitions.size[start]; idx += group_num) {
        const uint nodeIdx     = idx + global_offset;
        const uint node        = partitions.partition[nodeIdx];
        const uint nodeDegree  = degrees[node];
        const uint* neighbors_ = neighbors + neighbors_offset[node];
        uint local_count       = 0;
        for (uint neighborIdx = inner_id; neighborIdx < nodeDegree; neighborIdx += group_size) {
            const uint neighbor = neighbors_[neighborIdx];
            bool found;
            if (degree_of_tmp_node < 16) {
                found = Utils::sequentialSearch_unrolled(neighbor, shared_neighbor_of_tmp_node, degree_of_tmp_node);
            } else {
                found = Utils::binarySearch(neighbor, shared_neighbor_of_tmp_node, degree_of_tmp_node);
            }
            if (found) {
                local_count++;
            }
        }
        atomicAdd(&intersection[nodeIdx], local_count);
    }
}

__global__ void intersection_with_oneNode_and_X(const uint* degrees, const uint* neighbors,
                                                const uint* neighbors_offset, uint* intersection, Partition_Device_viewer_v3 partitions, const uint* tmp_node,
                                                const uint start, const uint group_size) {
    const uint tid                    = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size              = blockDim.x * gridDim.x;
    const uint group_id               = tid / group_size;
    const uint inner_id               = tid % group_size;
    const uint group_num              = grid_size / group_size;
    const uint the_node               = tmp_node[0];
    const uint degree_of_tmp_node     = degrees[the_node];
    const uint* neighbors_of_tmp_node = neighbors + neighbors_offset[the_node];
    uint global_offset                = partitions.get_global_offset(start);
    for (uint idx = group_id; idx < partitions.size[start]; idx += group_num) {
        const uint nodeIdx     = idx + global_offset;
        const uint node        = partitions.partition[nodeIdx];
        const uint nodeDegree  = degrees[node];
        const uint* neighbors_ = neighbors + neighbors_offset[node];
        uint local_count       = 0;
        for (uint neighborIdx = inner_id; neighborIdx < nodeDegree; neighborIdx += group_size) {
            const uint neighbor = neighbors_[neighborIdx];
            bool found;
            if (degree_of_tmp_node < 16) {
                found = Utils::sequentialSearch_unrolled(neighbor, neighbors_of_tmp_node, degree_of_tmp_node);
            } else {
                found = Utils::binarySearch(neighbor, neighbors_of_tmp_node, degree_of_tmp_node);
            }
            if (found) {
                local_count++;
            }
        }
        local_count = Utils::__reduce_warp_cub_sum(local_count);
        if (inner_id == 0) {
            atomicAdd(&intersection[nodeIdx], local_count);
        }
    }
}

__global__ void insert_new_Cand_exts_pre(uint* new_Cand_exts_size, const uint* Cand_exts_size, const int* intersection,
                                         const uint intersection_length, const int threshold, const CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < intersection_length; idx += grid_size) {
        if (Utils::get_judge(intersection[idx], threshold, op)) {
            continue;
        }
        const uint partition_idx = Partition_Device_viewer_v3::get_partition_idx_from_global_index(Cand_exts_size, idx);
        atomicAdd(&new_Cand_exts_size[partition_idx], 1);
    }
}

__global__ void insert_new_Cand_exts(Partition_Device_viewer_v3 new_Cand_exts, const uint* Cand_exts_partition,
                                     const uint* Cand_exts_size, const int* intersection, const uint intersection_length, const int threshold,
                                     const CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < intersection_length; idx += grid_size) {
        if (Utils::get_judge(intersection[idx], threshold, op)) {
            continue;
        }
        const uint partition_idx = Partition_Device_viewer_v3::get_partition_idx_from_global_index(Cand_exts_size, idx);
        new_Cand_exts.insert_node_to_partition_from_partitionIdx(Cand_exts_partition[idx], partition_idx);
    }
}


__global__ void insert_tmpNode_to_G_label_and_newX(
        const uint tmp_node, int* G_label, Partition_Device_viewer_v3 new_X, const uint partition_idx) {
    if (threadIdx.x == 0) {
        new_X.insert_node_to_partition_from_partitionIdx(tmp_node, partition_idx);
        G_label[tmp_node] = 1;
        
    }
}

__global__ void initialize_G_label(int* G_label, const uint bipartite_idx, const uint graph_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint node = tid + bipartite_idx; node < graph_size; node += grid_size) {
        G_label[node] = 1;
        
    }
}

__global__ void resetTmpNode_GLabel_GState(const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
                                           const uint tmp_node, int* G_label, arrMapTable G_state) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid == 0) {
        G_label[tmp_node] = 0;
        G_state.delete_element(tmp_node);
        
        
    }
    const uint* neighbors_of_node = neighbors + neighbors_offset[tmp_node];
    const uint degree             = degrees[tmp_node];
    for (uint idx = tid; idx < degree; idx += grid_size) {
        G_state.atomic_dec(neighbors_of_node[idx]);
        
    }
}

template<int threadsPerBlock>
__global__ void insertGState(const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
                             const uint tmp_node, int* G_label, arrMapTable G_state) {
    const uint tid                   = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size             = blockDim.x * gridDim.x;
    const uint degree                = degrees[tmp_node];
    const uint* neighbors_of_tmpNode = neighbors + neighbors_offset[tmp_node];
    int local_count                  = 0;
    if (tid == 0) {
        assert(G_state.set_value(tmp_node, degree));
        
    }

    for (uint idx = tid; idx < degree; idx += grid_size) {
        const uint neighbor = neighbors_of_tmpNode[idx];
        if (G_label[neighbor] == 1) {
            local_count++;
            assert(G_state.add(neighbor, 1));
            
        }
    }

    using BlockReduce = cub::BlockReduce<int, threadsPerBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int block_sum = BlockReduce(temp_storage).Sum(local_count);
    if (tid == 0) {
        
        assert(G_state.set_value(tmp_node, block_sum));
        
    }
}

__global__ void enumerateNX_insertNewNX_pre(Partition_Device_viewer_v3 NX, const uint NX_length,
                                            Partition_Device_viewer_v3 new_N_X, arrMapTable G_state, int* G_label, int* G_state_value, const int threshold,
                                            const CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < NX_length; idx += grid_size) {
        const uint node          = NX.partition[idx];
        const uint partition_idx = NX.get_partition_idx_from_global_index(NX.size, idx);
        const int value          = G_state.get_value(node);
        G_state_value[idx]       = value;
        if (Utils::get_judge(value, threshold, op)) {
            atomicAdd(&new_N_X.size[partition_idx], 1);
        } else {
            G_label[node] = 0;
            
        }
    }
}

__global__ void enumerateNX_insertNewNX(const uint* degrees, const uint* NX_partition, const uint NX_length,
                                        Partition_Device_viewer_v3 new_N_X, const int* G_state_value, const int threshold, const CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < NX_length; idx += grid_size) {
        const uint node = NX_partition[idx];
        if (Utils::get_judge(G_state_value[idx], threshold, op)) {
            new_N_X.insert_node_to_partition_from_degree(node, degrees[node]);
        }
    }
}

__global__ void enumerateNX_resetGstateX(const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
                                         const int* G_state_value, Partition_Device_viewer_v3 NX, int* G_label, arrMapTable G_state, const int threshold,
                                         const CompareOp op, const uint start, const uint group_size, const uint id = 0) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id  = tid / group_size;
    const uint inner_id  = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset    = NX.get_global_offset(start);
    for (uint idx = group_id; idx < NX.size[start]; idx += group_num) {
        const uint nodeIdx = offset + idx;
        const uint node    = NX.partition[nodeIdx];
        if (!Utils::get_judge(G_state_value[nodeIdx], threshold, op) && G_state_value[nodeIdx] != 0) {
            const uint* neighbors_of_tmpNode = neighbors + neighbors_offset[node];
            for (uint neighborIdx = inner_id; neighborIdx < degrees[node]; neighborIdx += group_size) {
                const uint neighbor = neighbors_of_tmpNode[neighborIdx];
                if (G_label[neighbor] == 1) {
                    
                    G_state.atomic_dec(neighbor);
                    
                }
            }
        }
    }
}

__global__ void getIntersection_of_CandQ_GLabelX(const uint* degrees, const uint* neighbors,
                                                 const uint* neighbors_offset, Partition_Device_viewer_v3 Candq, const uint newNX_length, const int* G_label,
                                                 int* intersection, const uint start, const uint group_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id  = tid / group_size;
    const uint inner_id  = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset    = Candq.get_global_offset(start);
    for (uint idx = group_id; idx < Candq.size[start]; idx += group_num) {
        const uint nodeIdx = offset + idx;
        const uint node    = Candq.partition[nodeIdx];
        if (degrees[node] >= newNX_length) {
            const uint* neighbor_of_node = neighbors + neighbors_offset[node];
            for (uint neighborIdx = inner_id; neighborIdx < degrees[node]; neighborIdx += group_size) {
                const uint neighbor = neighbor_of_node[neighborIdx];
                if (G_label[neighbor] == 1) {
                    Utils::atomicAggInc(&intersection[nodeIdx]);
                }
            }
        }
    }
}

__global__ void intersection_of_CandExts_and_G_label(const uint* degrees, const uint* neighbors,
                                                     const uint* neighbors_offset, Partition_Device_viewer_v3 CandExts, int* intersection, const int* G_label,
                                                     const uint start, uint group_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id  = tid / group_size;
    const uint inner_id  = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset    = CandExts.get_global_offset(start);
    for (uint idx = group_id; idx < CandExts.size[start]; idx += group_num) {
        const uint nodeIdx           = offset + idx;
        const uint node              = CandExts.partition[nodeIdx];
        const uint* neighbor_of_node = neighbors + neighbors_offset[node];
        for (uint neighborIdx = inner_id; neighborIdx < degrees[node]; neighborIdx += group_size) {
            const uint neighbor = neighbor_of_node[neighborIdx];
            if (G_label[neighbor] == 1) {
                atomicAdd(&intersection[nodeIdx], 1);
            }
        }
    }
}

__global__ void insertComponentNewX_newCandExts_pre(Partition_Device_viewer_v3 CandExts, const uint CandExts_length,
                                                    const int* intersection, Partition_Device_viewer_v3 Component, Partition_Device_viewer_v3 newX,
                                                    Partition_Device_viewer_v3 newCandExts, uint new_N_X_length, int* G_label, arrMapTable G_state, const int threshold,
                                                    const int threshold2, const CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < CandExts_length; idx += grid_size) {
        const int tmp_count     = intersection[idx];
        const uint partitionIdx = CandExts.get_partition_idx_from_size(idx);
        const uint node         = CandExts.partition[idx];
        if (Utils::get_judge(tmp_count, threshold, op)) {
            atomicAdd(&Component.size[partitionIdx], 1);
            atomicAdd(&newX.size[partitionIdx], 1);
            G_label[node] = 1;
            
            assert(G_state.set_value(node, new_N_X_length));
            
        } else if (Utils::get_judge(tmp_count, threshold2, op)) {
            atomicAdd(&newCandExts.size[partitionIdx], 1);
        }
    }
}

__global__ void insertComponentNewX_newCandExts(Partition_Device_viewer_v3 CandExts, const uint CandExts_length,
                                                int* intersection, Partition_Device_viewer_v3 Component, Partition_Device_viewer_v3 newX,
                                                Partition_Device_viewer_v3 newCandExts, const int threshold, const int threshold2, const CompareOp op) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < CandExts_length; idx += grid_size) {
        const int tmp_count     = intersection[idx];
        const uint partitionIdx = CandExts.get_partition_idx_from_size(idx);
        const uint node         = CandExts.partition[idx];
        if (Utils::get_judge(tmp_count, threshold, op)) {
            Component.insert_node_to_partition_from_partitionIdx(node, partitionIdx);
            newX.insert_node_to_partition_from_partitionIdx(node, partitionIdx);
        } else if (Utils::get_judge(tmp_count, threshold2, op)) {
            
            const uint insert_pos = atomicAdd(&newCandExts.idxs[partitionIdx], 1);
            
            const uint global_offset = newCandExts.get_global_offset(partitionIdx);
            
            newCandExts.partition[global_offset + insert_pos] = node;
        }
    }
}

__global__ void enumerateNewNX_Gstate_addComponent(
        const uint* newNX_partition, const uint newNX_length, const int componentLength, arrMapTable G_state) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < newNX_length; idx += grid_size) {
        const uint node = newNX_partition[idx];
        assert(G_state.add(node, componentLength));
        
    }
}

__global__ void intersection_of_new_Cand_extsNeighbor_and_new_Xneighbors(const uint* degrees, const uint* neighbors,
                                                                         const uint* neighbors_offset, Partition_Device_viewer_v3 new_Cand_exts, Partition_Device_viewer_v3 new_X,
                                                                         int* G_label, int* judge, const int threshold, const CompareOp op, uint id = -1) {
    const uint new_Cand_exts_idx = blockIdx.y;
    const uint newX_idx          = blockIdx.x;
    const uint tid               = threadIdx.x;
    if (judge[new_Cand_exts_idx] == 0) {
        return;
    }

    const uint cand_node       = new_Cand_exts.partition[new_Cand_exts_idx];
    const uint* cand_neighbors = neighbors + neighbors_offset[cand_node];
    const uint newX_node       = new_X.partition[newX_idx];
    const uint* newX_neighbors = neighbors + neighbors_offset[newX_node];

    int local_count = 0;

    for (uint neighborIdx = tid; neighborIdx < degrees[newX_node]; neighborIdx += blockDim.x) {
        const uint newX_neighbor = newX_neighbors[neighborIdx];
        if (G_label[newX_neighbor] == 1 && Utils::binarySearch(newX_neighbor, cand_neighbors, degrees[cand_node])) {
            local_count++;
        }
    }
    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int intersection_size = BlockReduce(temp_storage).Sum(local_count);
    
    if (tid == 0) {
        
        if (Utils::get_judge(intersection_size, threshold, op)) {
            atomicExch(&judge[new_Cand_exts_idx], 0);
        }
    }
}

__global__ void intersection_of_new_Cand_extsNeighbor_and_new_Xneighbors_gridstride(const uint* degrees,
                                                                                    const uint* neighbors, const uint* neighbors_offset, Partition_Device_viewer_v3 new_Cand_exts,
                                                                                    Partition_Device_viewer_v3 new_X, int* G_label, int* judge, const int threshold, const CompareOp op,
                                                                                    const uint total_X_size, const uint total_Cand_exts_size) {

    
    for (uint new_Cand_exts_idx = blockIdx.y; new_Cand_exts_idx < total_Cand_exts_size;
         new_Cand_exts_idx += gridDim.y) {

        
        
        
        
        if (judge[new_Cand_exts_idx] == 0) {
            continue;
        }

        
        for (uint newX_idx = blockIdx.x; newX_idx < total_X_size; newX_idx += gridDim.x) {

            
            
            
            const uint tid = threadIdx.x;

            
            if (judge[new_Cand_exts_idx] == 0) {
                
                break;
            }

            const uint cand_node       = new_Cand_exts.partition[new_Cand_exts_idx];
            const uint* cand_neighbors = neighbors + neighbors_offset[cand_node];
            const uint newX_node       = new_X.partition[newX_idx];
            const uint* newX_neighbors = neighbors + neighbors_offset[newX_node];

            int local_count = 0;

            for (uint neighborIdx = tid; neighborIdx < degrees[newX_node]; neighborIdx += blockDim.x) {
                const uint newX_neighbor = newX_neighbors[neighborIdx];
                if (G_label[newX_neighbor] == 1
                    && Utils::binarySearch(newX_neighbor, cand_neighbors, degrees[cand_node])) {
                    local_count++;
                }
            }

            using BlockReduce = cub::BlockReduce<int, 256>; 
            __shared__ typename BlockReduce::TempStorage temp_storage;
            const int intersection_size = BlockReduce(temp_storage).Sum(local_count);

            if (tid == 0) {
                if (Utils::get_judge(intersection_size, threshold, op)) {
                    atomicExch(&judge[new_Cand_exts_idx], 0);
                }
            }
            __syncthreads();
        }
    }
}


__global__ void insert_newCand_pre(
        Partition_Device_viewer_v3 newCand, const uint newCand_length, uint* new_partition_size, const int* judge) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < newCand_length; idx += grid_size) {
        if (judge[idx] == 1) {
            const uint partitionIdx = newCand.get_partition_idx_from_size(idx);
            atomicAdd(&new_partition_size[partitionIdx], 1);
        }
    }
}

__global__ void insert_newCand(const uint* degrees, const uint* newCand_partition, const uint newCand_length,
                               Partition_Device_viewer_v3 new_partition, const int* judge) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < newCand_length; idx += grid_size) {
        if (judge[idx] == 1) {
            const uint node = newCand_partition[idx];
            new_partition.insert_node_to_partition_from_degree(node, degrees[node]);
        }
    }
}

__global__ void enumerateNewNX_resetGLable_and_GState(const uint* degrees, const uint* neighbors,
                                                      const uint* neighbors_offset, Partition_Device_viewer_v3 new_N_X, const bool* G_prune, int* G_label,
                                                      arrMapTable G_state, const uint start = -1, const uint group_size = -1) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_id  = tid / group_size;
    const uint inner_idx = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset    = new_N_X.get_global_offset(start);
    for (uint idx = group_id; idx < new_N_X.size[start]; idx += group_num) {
        const uint nodeIdx = idx + offset;
        const uint node    = new_N_X.partition[nodeIdx];
        
        
        if (G_prune[node]) {
            continue;
        }
        if (inner_idx == 0) {
            
            G_label[node] = -1;
            
        }
        const uint* neighbors_of_node = neighbors + neighbors_offset[node];
        for (uint neighborIdx = inner_idx; neighborIdx < degrees[node]; neighborIdx += group_size) {
            const uint neighbor = neighbors_of_node[neighborIdx];
            
            
            if (G_label[neighbor] == 1) {
                G_state.atomic_dec(neighbor);
                
            }
        }
    }
}

__global__ void createNew_NewNX_pre(
        Partition_Device_viewer_v3 new_N_X, uint length, uint* new_tmp_size, const bool* G_prune) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < length; idx += grid_size) {
        const uint node = new_N_X.partition[idx];
        if (G_prune[node] == false) {
            
            
            continue;
        }
        const uint partiton_idx = new_N_X.get_partition_idx_from_size(idx);
        atomicAdd(&new_tmp_size[partiton_idx], 1);
    }
}

__global__ void createNew_NewNX(
        Partition_Device_viewer_v3 new_N_X, uint length, Partition_Device_viewer_v3 new_tmp, bool* G_prune) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < length; idx += grid_size) {
        const uint node = new_N_X.partition[idx];
        if (G_prune[node] == false) {
            
            
            continue;
        }
        const uint partition_idx = new_N_X.get_partition_idx_from_size(idx);
        new_tmp.insert_node_to_partition_from_partitionIdx(node, partition_idx);
    }
}

__global__ void changeGprune(uint* new_N_X, uint length, bool* G_prune) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < length; idx += grid_size) {
        const uint node = new_N_X[idx];
        if (G_prune[node])G_prune[node] = false;
        
        
        
        
        
    }
}

__global__ void resetG_prune(uint* new_Cand_exts, uint length, bool* G_prune) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < length; idx += grid_size) {
        const uint node = new_Cand_exts[idx];
        if (G_prune[node]) {
            G_prune[node] = false;
            
        }
    }
}

__global__ void resetG_state_G_label_component(const uint temp_node, const uint* component_partition,
                                               const uint component_length, int* G_label, arrMapTable G_state) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid == 0) {
        G_state.delete_element(temp_node);
        
        G_label[temp_node] = 0;
    }
    for (uint idx = tid; idx < component_length; idx += grid_size) {
        const uint node = component_partition[tid];
        G_label[node]   = 0;
        G_state.delete_element(node);
        
        
    }
}

__global__ void resetG_state_NX(const uint* N_X_partition, const uint NX_length, const int component_length,
                                const int* G_label, arrMapTable G_state) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < NX_length; idx += grid_size) {
        const uint node            = N_X_partition[idx];
        const int32_t& value_label = G_label[node];
        if (value_label == 1 || value_label == -1) {
            G_state.atomic_dec(node, component_length);
            
        }
    }
}

__global__ void resetG_state_NX_neighbors(const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
                                          Partition_Device_viewer_v3 NX, const int* G_label, arrMapTable G_state, const uint start, const uint group_size) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    const uint group_idx = tid / group_size;
    const uint inner_idx = tid % group_size;
    const uint group_num = grid_size / group_size;
    const uint offset    = NX.get_global_offset(start);
    for (uint idx = group_idx; idx < NX.size[start]; idx += group_num) {
        const uint nodeIdx = idx + offset;
        const uint node    = NX.partition[nodeIdx];
        if (G_label[node] == 0 || G_label[node] == -1) {
            const int32_t value = G_state.get_value(node);
            if (value != 0) {
                const uint* neighbor_of_node = neighbors + neighbors_offset[node];
                for (uint neighborIdx = inner_idx; neighborIdx < degrees[node]; neighborIdx += group_size) {
                    const uint neighbor = neighbor_of_node[neighborIdx];
                    if (G_label[neighbor] == 1) {
                        assert(G_state.add(neighbor, 1));
                        
                    }
                }
            }
        }
    }
}

__global__ void resetG_label_NX(const uint* N_X_partition, const uint NX_length, int* G_label) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < NX_length; idx += grid_size) {
        const uint node = N_X_partition[idx];
        G_label[node]   = 1;
        
    }
}

__global__ void resetGstate_temp_node(const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
                                      const uint temp_node, const int* G_label, arrMapTable G_state) {
    const uint tid               = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size         = blockDim.x * gridDim.x;
    const uint* neighbor_of_node = neighbors + neighbors_offset[temp_node];
    for (uint neighbor_idx = tid; neighbor_idx < degrees[temp_node]; neighbor_idx += grid_size) {
        const uint neighbor = neighbor_of_node[neighbor_idx];
        if (G_label[neighbor] == 1) {
            G_state.atomic_dec(neighbor);
            
        }
    }
}

__global__ void resetGState_GLabel_temp_node(const uint* degrees, const uint* neighbors, const uint* neighbors_offset,
                                             const uint temp_node, arrMapTable G_state, int* G_label) {
    const uint tid       = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    if (tid == 0) {
        G_state.delete_element(temp_node);
        
        G_label[temp_node] = 0;
    }
    const uint* neighbors_of_temp_node = neighbors + neighbors_offset[temp_node];
    for (uint neighborIdx = tid; neighborIdx < degrees[temp_node]; neighborIdx += grid_size) {
        uint neighbor = neighbors_of_temp_node[neighborIdx];
        G_state.atomic_dec(neighbor);
        
    }
}

__global__ void prune_new_Cand_exts(uint* new_Cand_exts, uint new_Cand_exts_length, uint *new_Cand_exts_sizes, bool* device_G_prune, uint* new_partition_size) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < new_Cand_exts_length; idx += grid_size) {
        uint node = new_Cand_exts[idx];
        if (!device_G_prune[node]) continue;
        uint partition_idx = Partition_Device_viewer_v3::get_partition_idx_from_global_index(new_Cand_exts_sizes, idx);
        atomicAdd(&new_partition_size[partition_idx], 1);
    }
}

__global__ void insert_new_Cand_exts(uint* new_Cand_exts, uint new_Cand_exts_length, uint *new_Cand_exts_sizes, bool* device_G_prune, Partition_Device_viewer_v3 new_partition) {
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint grid_size = blockDim.x * gridDim.x;
    for (uint idx = tid; idx < new_Cand_exts_length; idx += grid_size) {
        uint node = new_Cand_exts[idx];
        if (!device_G_prune[node]) continue;
        uint partition_idx = Partition_Device_viewer_v3::get_partition_idx_from_global_index(new_Cand_exts_sizes, idx);
        new_partition.insert_node_to_partition_from_partitionIdx(node, partition_idx);
    }
}

void enumerate_GPU(Partition_v3& X, Partition_v3& N_X, Partition_v3& Cand_exts, Partition_v3& Cand_q,
                   vector<uint>& host_Cand_q, Utils::cpu_hashTable::cpu_hash_table& hashTable_G_state, int* device_G_label,
                   arrMapTable& device_arrGstate, bool* device_G_prune, Utils::cpu_hashTable::cpu_hash_table& hashTable_G_temp,
                   uint* device_degrees, uint* device_neighbors, uint* device_neighbors_offset, vector<uint>& host_degrees,
                   vector<uint>& host_neighbors, vector<uint>& host_neighbors_offset, const uint graph_size, const uint k,
                   const uint theta, const uint bipartite_idx, cudaStream_t& stream) {
    if (d_res_count >= num) {
        return;
    }
    stack<State> s;
    stack<newState> s_new;
    s.emplace(std::make_shared<Partition_v3>(std::move(X)), std::make_shared<Partition_v3>(std::move(N_X)),
              std::make_shared<Partition_v3>(std::move(Cand_exts)), std::ref(host_Cand_q), std::make_shared<Partition_v3>(std::move(Cand_q)));
    
    auto G_state_ = device_arrGstate;
    while (not s.empty()) {
        if (s.top().buildNew) {
            s_new.push(newState());
        }
        bool recursion = false;

        State &current_state                           = s.top();
        auto X                     = s.top().X;
        auto N_X                   = s.top().N_X;
        auto Cand_exts             = s.top().Cand_exts;
        auto Cand_q                = s.top().Cand_q;
        vector<uint>& _host_Cand_q = s.top().host_Cand_q.get();

        std::shared_ptr<Partition_v3> new_N_X;
        std::shared_ptr<Partition_v3> new_X;
        std::shared_ptr<Partition_v3> new_Cand_exts;
        std::shared_ptr<CandidateQueue_v3> new_Cand_q;
        newState& top_new_state = s_new.top();

        if (!top_new_state.new_X) {
            top_new_state.initialize(X->total_size(), Cand_exts->partition_original_ptr_.get());
        }
        new_N_X       = top_new_state.new_N_X;
        new_X         = top_new_state.new_X;
        new_Cand_exts = top_new_state.new_Cand_exts;
        new_Cand_q    = s_new.top().new_Cand_q;
        vector<uint>& host_new_Cand_q = s_new.top().host_new_Cand_q;

        if (verbose ) {
            fmt::println("id: {}", id++);
            fmt::println("X.size = {}", X->totalSize);
            fmt::println("N_X.size = {}", N_X->totalSize);
            fmt::println("Cand_exts.size = {}", Cand_exts->totalSize);
            fmt::println("Cand_q.size = {}", _host_Cand_q.size());
            if (s.top().parentLayer_Component) {
                fmt::println("Component.size = {}", s.top().parentLayer_Component->totalSize);
            }


            fmt::println("new_N_X.size = {}", new_N_X->totalSize);
            fmt::println("new_X.size = {}", new_X->totalSize);
            fmt::println("new_Cand_exts.size = {}", new_Cand_exts->totalSize);
            
            uint* tmp_Cand_exts = new uint[Cand_exts->totalSize];
            CUDA_ERROR_CHECK(cudaMemcpy(
                    tmp_Cand_exts, Cand_exts->partition, Cand_exts->totalSize * sizeof(uint), cudaMemcpyDeviceToHost));
            sort(tmp_Cand_exts, tmp_Cand_exts + Cand_exts->totalSize);
            fmt::print("new_Cand_exts: ");
            for (uint i = 0; i < min(Cand_exts->totalSize, 5); i++) {
                fmt::print("{}, ", tmp_Cand_exts[i]);
            }
            fmt::print("\n");
            fmt::println("------------------------------------------");
            cout << endl;
        }

        

        
        
        
        
        
        int Cand_exts_idx_bias = Cand_exts->get_bias();
        while (Cand_exts->totalSize) {
            s.top().inWhile = true;
            const uint tmpNode_Cand_ext_partition_idx =
                    Cand_exts->get_partition_idx_from_global_idx(Cand_exts_idx_bias);
            
            
            
            if (X->totalSize != 0) {
                
                new_X->CopyPartitionFromX(*X, tmpNode_Cand_ext_partition_idx, stream);
            } else {
                if (Cand_exts_idx_bias != 0) {
                    
                    const uint tmpNode_Cand_ext_partition_idx_last =
                            Cand_exts->get_partition_idx_from_global_idx(Cand_exts_idx_bias - 1);
                    new_X->host_size_.get()[tmpNode_Cand_ext_partition_idx_last]--;
                    CUDA_ERROR_CHECK(cudaMemcpyAsync(new_X->idxs_.get(), new_X->host_size_.get(),
                                                     sizeof(uint) * Partition_v3::max_partition_num_, cudaMemcpyHostToDevice));
                }
                new_X->host_size_.get()[tmpNode_Cand_ext_partition_idx]++;
                CUDA_ERROR_CHECK(cudaMemcpyAsync(new_X->size_.get(), new_X->host_size_.get(),
                                                 sizeof(uint) * Partition_v3::max_partition_num_, cudaMemcpyHostToDevice));
            }
            new_X->totalSize = X->totalSize + 1;
            
            
            
            new_N_X->clean(stream);
            new_Cand_exts->clean(stream);
            uint host_tmp_node = current_state.fetch_next_node_batch(stream);
            uint* device_tmp_node = Cand_exts->partition_original_ptr_.get() + Cand_exts_idx_bias;
            
            
            Cand_exts_idx_bias++;
            
            
            
            if (verbose ) {
                fmt::println("temp_node = {}, degree = {}", host_tmp_node, host_degrees[host_tmp_node]);
            }
            insert_tmpNode_to_G_label_and_newX<<<1, 32, 0, stream>>>(
                    host_tmp_node, device_G_label, new_X->get_gpu_viewer(), tmpNode_Cand_ext_partition_idx);
            if (static_cast<int>(new_X->totalSize) <= k + 1 && static_cast<int>(new_X->totalSize) < theta) {
                
                uint grid_size = CALC_GRID_DIM(host_degrees[host_tmp_node], THREADS_PER_BLOCK);
                insert_G_state<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
                        device_tmp_node, device_degrees, device_neighbors, device_neighbors_offset, G_state_);
                if (static_cast<int>(new_X->totalSize) == k + 1) {
                    grid_size = CALC_GRID_DIM(graph_size - bipartite_idx, THREADS_PER_BLOCK);
                    insert_new_N_X_pre<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(device_degrees, G_state_,
                                                                                    device_G_label, new_N_X->get_gpu_viewer(), bipartite_idx, graph_size);
                    new_N_X->malloc_partition(stream);
                    insert_new_N_X<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
                            device_degrees, G_state_, new_N_X->get_gpu_viewer(), bipartite_idx, graph_size);
                }
                else {
                    *new_N_X = *N_X;
                }
                int* intersection_Cand_exts;
                CUDA_ERROR_CHECK(cudaMallocAsync(&intersection_Cand_exts, sizeof(int) * Cand_exts->totalSize, stream));
                CUDA_ERROR_CHECK(
                        cudaMemsetAsync(intersection_Cand_exts, 0, sizeof(int) * Cand_exts->totalSize, stream));
                if (tmpNode_Cand_ext_partition_idx < 3) {
                    for (uint start = 0; start < Partition_v3::max_partition_num_; start++) {
                        if (Cand_exts->host_size_.get()[start] == 0) {
                            continue;
                        }
                        uint group_size     = pow(32, max(1, start));
                        uint totalThreadNum = group_size * Cand_exts->host_size_.get()[start];
                        uint gridDim        = CALC_GRID_DIM(totalThreadNum, THREADS_PER_BLOCK);
                        intersection_with_oneNode_and_CandExts_sharedMem<<<gridDim, THREADS_PER_BLOCK,
                        host_degrees[host_tmp_node] * sizeof(uint), stream>>>(device_degrees, device_neighbors,
                                                                              device_neighbors_offset, intersection_Cand_exts, device_G_label,
                                                                              Cand_exts->get_gpu_viewer(), device_tmp_node, theta - k, CompareOp::Less, start,
                                                                              group_size);
                    }
                }
                else {
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    LAUNCH_PARTITION_KERNEL_V3(Cand_exts, intersection_with_oneNode_and_CandExts, stream,
                                               device_degrees, device_neighbors, device_neighbors_offset, intersection_Cand_exts,
                                               device_G_label, Cand_exts->get_gpu_viewer(), device_tmp_node, theta - k, CompareOp::Less);
                }
                if (Cand_exts->totalSize > 0) {
                    uint gridDim = CALC_GRID_DIM(Cand_exts->totalSize, THREADS_PER_BLOCK);
                    insert_new_Cand_exts_pre<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(new_Cand_exts->size_.get(),
                                                                                        Cand_exts->size_.get(), intersection_Cand_exts, Cand_exts->totalSize,
                                                                                        static_cast<int>(theta) - 2 * k, CompareOp::Less);
                    new_Cand_exts->malloc_partition(stream);
                    insert_new_Cand_exts<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(new_Cand_exts->get_gpu_viewer(),
                                                                                    Cand_exts->partition, Cand_exts->size_.get(), intersection_Cand_exts, Cand_exts->totalSize,
                                                                                    static_cast<int>(theta) - 2 * k, CompareOp::Less);
                }
                CUDA_ERROR_CHECK(cudaFreeAsync(intersection_Cand_exts, stream));

                if (static_cast<int>(host_degrees[host_tmp_node]) >= static_cast<int>(theta) - static_cast<int>(k)) {
                    bool canExtension    = true;
                    uint* intersection_X = nullptr;
                    if (X->totalSize > 0) {
                        CUDA_ERROR_CHECK(cudaMallocAsync(&intersection_X, sizeof(uint) * X->totalSize, stream));
                        CUDA_ERROR_CHECK(cudaMemsetAsync(intersection_X, 0, sizeof(uint) * X->totalSize, stream));
                        if (tmpNode_Cand_ext_partition_idx < 3) {
                            for (uint start = 0; start < Partition_v3::max_partition_num_; start++) {
                                if (X->host_size_.get()[start] == 0) {
                                    continue;
                                }
                                uint group_size     = pow(32, max(1, start));
                                uint totalThreadNum = group_size * X->host_size_.get()[start];
                                uint gridDim        = CALC_GRID_DIM(totalThreadNum, THREADS_PER_BLOCK);
                                intersection_with_oneNode_and_X_sharedMem<<<gridDim, THREADS_PER_BLOCK,
                                host_degrees[host_tmp_node] * sizeof(uint), stream>>>(device_degrees,
                                                                                      device_neighbors, device_neighbors_offset, intersection_X, X->get_gpu_viewer(),
                                                                                      device_tmp_node, start, group_size);
                            }
                        }
                        else {
                            LAUNCH_PARTITION_KERNEL_V3(X, intersection_with_oneNode_and_X, stream,
                                                       device_degrees, device_neighbors, device_neighbors_offset, intersection_X,
                                                       X->get_gpu_viewer(), device_tmp_node);
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                        }
                    }
                    if (intersection_X) {
                        canExtension =
                                !Utils::any_satisfied(intersection_X, X->totalSize, theta - 2 * k, CompareOp::Less, stream);
                    }
                    CUDA_ERROR_CHECK(cudaFreeAsync(intersection_X, stream));
                    if (canExtension) {
                        auto state_new_Cand_q = std::make_shared<Partition_v3>(std::move(Partition_v3::MergeAndCreate(*Cand_q, *new_Cand_q, stream)));
                        State _state(new_X, new_N_X, new_Cand_exts, host_new_Cand_q, state_new_Cand_q);
                        s.top().flag                      = 0;
                        s.top().isRecurveive              = true;
                        _state.parentLayer_candidate_node = host_tmp_node;
                        s.push(_state);
                        recursion = true;
                        break;
                    }
                }
                uint grid_dim = CALC_GRID_DIM(host_degrees[host_tmp_node], THREADS_PER_BLOCK);
                new_Cand_q->push(tmpNode_Cand_ext_partition_idx, stream);
                
                resetTmpNode_GLabel_GState<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
                        device_degrees, device_neighbors, device_neighbors_offset, host_tmp_node, device_G_label, G_state_);
                if (new_X->totalSize == k + 1) {
                    grid_dim = CALC_GRID_DIM(graph_size - bipartite_idx, THREADS_PER_BLOCK);
                    initialize_G_label<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
                            device_G_label, bipartite_idx, graph_size);
                }
            }
            else {
                
                
                int* device_workspace        = nullptr;
                uint deviceWorkSpace_maxSize = std::max({N_X->totalSize, Cand_exts->totalSize,
                                                         static_cast<uint>(Cand_q->totalSize), new_Cand_exts->totalSize});
                CUDA_ERROR_CHECK(cudaMalloc(&device_workspace, sizeof(int) * deviceWorkSpace_maxSize));
                CUDA_ERROR_CHECK(cudaMemset(device_workspace, 0, sizeof(int) * deviceWorkSpace_maxSize));

                int* G_state_value = device_workspace;
                int G_state_size   = N_X->totalSize;

                
                
                uint grid_num = CALC_GRID_DIM(host_degrees[host_tmp_node], THREADS_PER_BLOCK);
                insertGState<512><<<1, 512, 0, stream>>>(
                        device_degrees, device_neighbors, device_neighbors_offset, host_tmp_node, device_G_label, G_state_);
                
                
                if (N_X->totalSize > 0) {
                    grid_num = CALC_GRID_DIM(N_X->totalSize, THREADS_PER_BLOCK);
                    
                    enumerateNX_insertNewNX_pre<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(N_X->get_gpu_viewer(),
                                                                                            N_X->totalSize, new_N_X->get_gpu_viewer(), G_state_, device_G_label, G_state_value,
                                                                                            static_cast<int>(new_X->totalSize) - static_cast<int>(k), CompareOp::GreaterEqual);
                    new_N_X->malloc_partition(stream);
                    
                    if (new_N_X->totalSize > 0) {
                        enumerateNX_insertNewNX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(device_degrees,
                                                                                            N_X->partition, N_X->totalSize, new_N_X->get_gpu_viewer(), G_state_value,
                                                                                            static_cast<int>(new_X->totalSize) - static_cast<int>(k), CompareOp::GreaterEqual);
                    }
                }
                
                
                
                LAUNCH_PARTITION_KERNEL_V3(N_X, enumerateNX_resetGstateX, stream, device_degrees, device_neighbors,
                                           device_neighbors_offset, G_state_value, N_X->get_gpu_viewer(), device_G_label, G_state_,
                                           static_cast<int>(new_X->totalSize) - static_cast<int>(k), CompareOp::GreaterEqual);
                CUDA_ERROR_CHECK(cudaMemsetAsync(G_state_value, 0, sizeof(int) * N_X->totalSize, stream));
                bool Extension    = not new_N_X->empty();
                int* intersection = device_workspace;
                if (not Cand_q->empty()) {
                    
                    
                    
                    
                    CUDA_ERROR_CHECK(cudaMemsetAsync(device_workspace, 0, sizeof(int) * G_state_size, stream));
                    LAUNCH_PARTITION_KERNEL_V3(Cand_q, getIntersection_of_CandQ_GLabelX, stream,
                                               device_degrees, device_neighbors, device_neighbors_offset, Cand_q->get_gpu_viewer(),
                                               new_N_X->totalSize, device_G_label, intersection);
                    
                    
                    
                    Extension = !Utils::any_satisfied(
                            intersection, Cand_q->totalSize, new_N_X->totalSize, CompareOp::GreaterEqual, stream);
                }
                auto Component = std::make_shared<Partition_v3>();
                if (Extension) {
                    if (Cand_exts->totalSize > 0) {
                        CUDA_ERROR_CHECK(cudaMemsetAsync(intersection, 0, Cand_exts->totalSize * sizeof(int), stream));
                        LAUNCH_PARTITION_KERNEL_V3(Cand_exts, intersection_of_CandExts_and_G_label, stream,
                                                   device_degrees, device_neighbors, device_neighbors_offset, Cand_exts->get_gpu_viewer(),
                                                   intersection, device_G_label);
                        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
                        grid_num = CALC_GRID_DIM(Cand_exts->totalSize, THREADS_PER_BLOCK);
                        insertComponentNewX_newCandExts_pre<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                                Cand_exts->get_gpu_viewer(), Cand_exts->totalSize, intersection,
                                Component->get_gpu_viewer(), new_X->get_gpu_viewer(), new_Cand_exts->get_gpu_viewer(),
                                new_N_X->totalSize, device_G_label, G_state_, static_cast<int>(new_N_X->totalSize),
                                static_cast<int>(theta) - static_cast<int>(k), CompareOp::GreaterEqual);
                        new_X->try_to_remalloc_partition(stream);
                        new_Cand_exts->try_to_remalloc_partition(stream);
                        Component->malloc_partition(stream);
                        insertComponentNewX_newCandExts<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                                Cand_exts->get_gpu_viewer(), Cand_exts->totalSize, intersection,
                                Component->get_gpu_viewer(), new_X->get_gpu_viewer(), new_Cand_exts->get_gpu_viewer(),
                                static_cast<int>(new_N_X->totalSize), static_cast<int>(theta) - static_cast<int>(k),
                                CompareOp::GreaterEqual);
                    }
                    
                    
                    
                    
                    
                    
                    
                    if (Component->totalSize > 0 && new_N_X->totalSize > 0) {
                        grid_num = CALC_GRID_DIM(new_N_X->totalSize, THREADS_PER_BLOCK);
                        enumerateNewNX_Gstate_addComponent<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                                new_N_X->partition, new_N_X->totalSize, Component->totalSize, G_state_);
                    }
                    if (verbose) {
                        output("Component", Component, fmt::fg(fmt::color::red));
                    }
                }
                if (Extension) {
                    int* judge = nullptr;
                    if (new_Cand_exts->totalSize > deviceWorkSpace_maxSize) {
                        CUDA_ERROR_CHECK(cudaMalloc(&judge, new_Cand_exts->totalSize * sizeof(uint)));
                    } else {
                        judge = device_workspace;
                    }

                    if (new_Cand_exts->totalSize > 0) {
                        grid_num = max(CALC_GRID_DIM(new_Cand_exts->totalSize, THREADS_PER_BLOCK), 2);
                        Utils::initialize_with_anyValue<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                                judge, new_Cand_exts->totalSize, 1);
                        if (new_X->totalSize > 0) {
                            if (new_X->totalSize * new_Cand_exts->totalSize <= 1024) {
                                dim3 gridDim(new_X->totalSize, new_Cand_exts->totalSize, 1);
                                intersection_of_new_Cand_extsNeighbor_and_new_Xneighbors<<<gridDim, THREADS_PER_BLOCK,
                                0, stream>>>(device_degrees, device_neighbors, device_neighbors_offset,
                                             new_Cand_exts->get_gpu_viewer(), new_X->get_gpu_viewer(), device_G_label, judge,
                                             static_cast<int>(theta) - 2 * (int) k, CompareOp::Less);
                                CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
                            } else {
                                dim3 gridDim(min((unsigned int) new_X->totalSize, MAX_GRID_DIM_X),
                                             min((unsigned int) new_Cand_exts->totalSize, MAX_GRID_DIM_Y), 1);
                                intersection_of_new_Cand_extsNeighbor_and_new_Xneighbors_gridstride<<<gridDim,
                                THREADS_PER_BLOCK, 0, stream>>>(device_degrees, device_neighbors,
                                                                device_neighbors_offset, new_Cand_exts->get_gpu_viewer(), new_X->get_gpu_viewer(),
                                                                device_G_label, judge, static_cast<int>(theta) - 2 * (int) k, CompareOp::Less,
                                                                new_X->totalSize, new_Cand_exts->totalSize);
                            }
                        }
                        Partition_v3 tmp_new_partition;
                        insert_newCand_pre<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(new_Cand_exts->get_gpu_viewer(),
                                                                                       new_Cand_exts->totalSize, tmp_new_partition.size_.get(), judge);
                        auto [totalSize, host_sizes] = tmp_new_partition.cal_new_size(stream);
                        if (totalSize != new_Cand_exts->totalSize) {
                            tmp_new_partition.malloc_partition(totalSize, host_sizes, stream);
                            insert_newCand<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(device_degrees,
                                                                                       new_Cand_exts->partition, new_Cand_exts->totalSize, tmp_new_partition.get_gpu_viewer(),
                                                                                       judge);
                            
                            *new_Cand_exts = std::move(tmp_new_partition);
                        }
                    }
                    std::shared_ptr<Partition_v3> state_new_Cand_q = nullptr;

                    if (new_X->totalSize >= theta) {
                        state_new_Cand_q = std::make_shared<Partition_v3>(std::move(Partition_v3::MergeAndCreate(*Cand_q, *new_Cand_q, stream)));
                        if (verbose ) {
                            fmt::println("Generate output: ");
                            output("new_X", new_X);
                            output("new_N_X", new_N_X);
                            output("new_Cand_exts", new_Cand_exts);
                            sort(host_new_Cand_q.begin(), host_new_Cand_q.end());
                            fmt::println("new_Cand_q(size = {}) = {}", host_new_Cand_q.size(), host_new_Cand_q);
                            cout << endl;
                        }
                        
                        
                        
                        
                        
                        

                        auto [after_prune_x, after_prune_y] = Generate_GPU(new_X, new_N_X, new_Cand_exts, state_new_Cand_q, hashTable_G_temp,
                                                                           device_G_label, hashTable_G_state, device_arrGstate, device_G_prune, device_degrees,
                                                                           device_neighbors, device_neighbors_offset, host_neighbors_offset, graph_size, bipartite_idx, theta, k, Component->totalSize,
                                                                           host_tmp_node, host_degrees[host_tmp_node], stream);
                        
                        
                        
                        if (d_res_count >= num) {
                            return;
                        }
                        if (after_prune_y) {
                            LAUNCH_PARTITION_KERNEL_V3(new_N_X, enumerateNewNX_resetGLable_and_GState, stream,
                                                       device_degrees, device_neighbors, device_neighbors_offset, new_N_X->get_gpu_viewer(),
                                                       device_G_prune, device_G_label, G_state_);
                            Partition_v3 tmp_new_partition;
                            grid_num = CALC_GRID_DIM(new_N_X->totalSize, THREADS_PER_BLOCK);
                            createNew_NewNX_pre<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(new_N_X->get_gpu_viewer(),
                                                                                            new_N_X->totalSize, tmp_new_partition.size_.get(), device_G_prune);
                            auto [length, new_host_sizes] = tmp_new_partition.cal_new_size(stream);
                            if (length != new_N_X->totalSize) {
                                tmp_new_partition.malloc_partition(length, new_host_sizes, stream);
                                createNew_NewNX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(new_N_X->get_gpu_viewer(),
                                                                                            new_N_X->totalSize, tmp_new_partition.get_gpu_viewer(), device_G_prune);
                            }
                            changeGprune<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                                    new_N_X->partition, new_N_X->totalSize, device_G_prune);
                            if (length != new_N_X->totalSize) *new_N_X = std::move(tmp_new_partition);
                        }
                        if (after_prune_x) {
                            grid_num = CALC_GRID_DIM(new_Cand_exts->totalSize, THREADS_PER_BLOCK);
                            Partition_v3 tmp_new_partition;
                            prune_new_Cand_exts<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(new_Cand_exts->partition, new_Cand_exts->totalSize,
                                                                                            new_Cand_exts->size_.get(), device_G_prune, tmp_new_partition.size_.get());
                            auto [length, new_host_sizes] = tmp_new_partition.cal_new_size(stream);
                            if (length != new_Cand_exts->totalSize) {
                                tmp_new_partition.malloc_partition(length, new_host_sizes, stream);
                                if (length > 0)
                                    insert_new_Cand_exts<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(new_Cand_exts->partition, new_Cand_exts->totalSize,
                                                                                                     new_Cand_exts->size_.get(), device_G_prune, tmp_new_partition.get_gpu_viewer());
                            }
                            changeGprune<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                                    new_Cand_exts->partition, new_Cand_exts->totalSize, device_G_prune);
                            if (length != new_Cand_exts->totalSize)
                                *new_Cand_exts = std::move(tmp_new_partition);
                        }

                        
                        
                    }
                    if (judge != device_workspace) {
                        CUDA_ERROR_CHECK(cudaFree(judge));
                    } else {
                        CUDA_ERROR_CHECK(cudaFree(device_workspace));
                        device_workspace = nullptr;
                    }
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                }
                if (device_workspace != nullptr) {
                    CUDA_ERROR_CHECK(cudaFree(device_workspace));
                }
                new_Cand_q->push(tmpNode_Cand_ext_partition_idx, stream);
                
                grid_num = CALC_GRID_DIM(Component->totalSize, THREADS_PER_BLOCK);
                resetG_state_G_label_component<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                        host_tmp_node, Component->partition, Component->totalSize, device_G_label, G_state_);
                if (N_X->totalSize > 0) {
                    grid_num = CALC_GRID_DIM(N_X->totalSize, THREADS_PER_BLOCK);
                    resetG_state_NX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                            N_X->partition, N_X->totalSize, Component->totalSize, device_G_label, G_state_);
                    LAUNCH_PARTITION_KERNEL_V3(N_X, resetG_state_NX_neighbors, stream, device_degrees, device_neighbors,
                                               device_neighbors_offset, N_X->get_gpu_viewer(), device_G_label, G_state_);
                    resetG_label_NX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                            N_X->partition, N_X->totalSize, device_G_label);
                }
                grid_num = CALC_GRID_DIM(host_degrees[host_tmp_node], THREADS_PER_BLOCK);
                resetGstate_temp_node<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                        device_degrees, device_neighbors, device_neighbors_offset, host_tmp_node, device_G_label, G_state_);
            }
        }
        if (recursion) {
            continue;
        }
        if (s.empty()) {
            break;
        }
        uint host_tmp_node    = s.top().parentLayer_candidate_node;
        auto Component_return = s.top().parentLayer_Component;
        while (not s.empty()) {
            if (not s.top().inWhile || not s.top().isRecurveive || s.top().Cand_exts->empty()) {
                s.pop();
                s_new.pop();
            }
            if (s.empty()) {
                break;
            }
            auto temp_Cand_exts             = s.top().Cand_exts;
            


            auto tmp_new_Cand_q_ = s_new.top().new_Cand_q;
            int tmp_Cand_exts_idx_bias = temp_Cand_exts->get_bias();
            const uint tmpNode_Cand_ext_partition_idx =
                    temp_Cand_exts->get_partition_idx_from_global_idx(tmp_Cand_exts_idx_bias);

            if (s.top().flag == 1) {
                auto tmp_N_X = s.top().N_X;
                
                tmp_new_Cand_q_->push(tmpNode_Cand_ext_partition_idx, stream);
                uint grid_num = CALC_GRID_DIM(Component_return->totalSize, THREADS_PER_BLOCK);
                resetG_state_G_label_component<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(host_tmp_node,
                                                                                           Component_return->partition, Component_return->totalSize, device_G_label, G_state_);
                if (tmp_N_X->totalSize > 0) {
                    uint grid_num = CALC_GRID_DIM(tmp_N_X->totalSize, THREADS_PER_BLOCK);
                    resetG_state_NX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                            tmp_N_X->partition, tmp_N_X->totalSize, Component_return->totalSize, device_G_label, G_state_);
                    LAUNCH_PARTITION_KERNEL_V3(tmp_N_X, resetG_state_NX_neighbors, stream, device_degrees,
                                               device_neighbors, device_neighbors_offset, tmp_N_X->get_gpu_viewer(), device_G_label, G_state_);
                    resetG_label_NX<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                            tmp_N_X->partition, tmp_N_X->totalSize, device_G_label);
                }
                grid_num = CALC_GRID_DIM(host_degrees[host_tmp_node], THREADS_PER_BLOCK);
                resetGstate_temp_node<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                        device_degrees, device_neighbors, device_neighbors_offset, host_tmp_node, device_G_label, G_state_);
            } else if (s.top().flag == 0) {
                auto tmp_new_X = s_new.top().new_X;
                
                tmp_new_Cand_q_->push(tmpNode_Cand_ext_partition_idx, stream);
                uint grid_num = CALC_GRID_DIM(host_degrees[host_tmp_node], THREADS_PER_BLOCK);
                resetGState_GLabel_temp_node<<<grid_num, THREADS_PER_BLOCK, 0, stream>>>(
                        device_degrees, device_neighbors, device_neighbors_offset, host_tmp_node, G_state_, device_G_label);
                
                if (tmp_new_X->totalSize == k + 1) {
                    uint grid_dim = CALC_GRID_DIM(graph_size - bipartite_idx, THREADS_PER_BLOCK);
                    initialize_G_label<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
                            device_G_label, bipartite_idx, graph_size);
                }
            }
            if (not s.empty() and not temp_Cand_exts->empty()) {
                s.top().buildNew = false;
                break;
            }
            host_tmp_node    = s.top().parentLayer_candidate_node;
            Component_return = s.top().parentLayer_Component;
            s.pop();
            s_new.pop();
        }
        if (not s.empty() && not s.top().inWhile && not s.top().isRecurveive) {
            s.pop();
            s_new.pop();
        }
    }
}

void testEnumerate(std::filesystem::path benchmark_file_path, uint k) {
    vector<uint> neighbors, degrees, neighbors_offset;
    uint graph_size, bipartite_idx;
    uint theta = 1;
    load_graph_from_bin<uint>(
            benchmark_file_path, neighbors, degrees, neighbors_offset, graph_size, bipartite_idx);
    fmt::println("----------------------");
    uint edge_num = accumulate(degrees.begin(), degrees.end(), 0);
    CUDA_ERROR_CHECK(cudaMalloc(&global_graph_parameter::G_temp, graph_size * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemset(global_graph_parameter::G_temp, -1, graph_size * sizeof(int)));

    fmt::print("Graph size: {}, edge num: {}, bipartite index: {}, k = {}, theta = {}", graph_size, edge_num,
               bipartite_idx, k, theta);
    cout << endl;
    std::string file_suffix;
#ifdef ABLATION_FIXED_GROUP_SIZE
    
    
    
    file_suffix = ".gpu" _VAL_TO_STR(ABLATION_FIXED_GROUP_SIZE);
#else
    file_suffix = ".gpu" + to_string(k);
#endif
    time_recorder.init(benchmark_file_path.string(), k, num, file_suffix);





    uint max_degree = max(degrees[bipartite_idx - 1], degrees[graph_size - 1]); 
    Partition_v3::set_max_partition_num_from_degree(max_degree);

    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

    Utils::cpu_hashTable::cpu_hash_table hashTable_G_state(max_degree * 1.5, 0.4);
    
    Utils::cpu_hashTable::cpu_hash_table hashTable_G_temp(max_degree * 10, 0.4);
    bool* device_G_prune;
    int* device_G_arrState;
    int* device_G_label;
    CUDA_ERROR_CHECK(cudaMalloc(&device_G_label, sizeof(int) * (graph_size)));
    CUDA_ERROR_CHECK(cudaMalloc(&device_G_arrState, sizeof(int) * graph_size));
    CUDA_ERROR_CHECK(cudaMalloc(&device_G_prune, sizeof(bool) * graph_size));
    CUDA_ERROR_CHECK(cudaMemset(device_G_prune, 0, sizeof(bool) * graph_size))
    CUDA_ERROR_CHECK(cudaMemset(device_G_arrState, 0, sizeof(int) * graph_size));
    uint grid = CALC_GRID_DIM(graph_size - bipartite_idx, THREADS_PER_BLOCK);
    initialize_G_label<<<grid, THREADS_PER_BLOCK, 0, stream>>>(device_G_label, bipartite_idx, graph_size);

    vector<uint> _Cand_exts, _N_X, _X, _Cand_q;
    _Cand_exts.reserve(bipartite_idx);
    _N_X.reserve(graph_size - bipartite_idx);
    std::unordered_map<int, vector<int>> tmp_Cand_exts, tmp_N_X;

    for (int i = 0; i < bipartite_idx; ++i) {
        int index = static_cast<int>(std::log(degrees[i]) / std::log(32));
        if (tmp_Cand_exts.find(index) == tmp_Cand_exts.end()) {
            tmp_Cand_exts[index] = std::vector<int>();
        }
        tmp_Cand_exts[index].push_back(i);
    }
    for (int i = bipartite_idx; i < graph_size; i++) {
        int index = static_cast<int>(std::log(degrees[i]) / std::log(32));
        if (tmp_N_X.find(index) == tmp_N_X.end()) {
            tmp_N_X[index] = std::vector<int>();
        }
        tmp_N_X[index].push_back(i);
    }
    for (int i = 0; i < 10 ; i++) {
        if (tmp_Cand_exts.find(i) == tmp_Cand_exts.end()) continue;
        auto &tmp_vec = tmp_Cand_exts[i];
        for (auto &ele: tmp_vec)
            _Cand_exts.emplace_back(ele);
    }
    for (int i = 0; i < 10; i++) {
        if (tmp_N_X.find(i) == tmp_N_X.end()) continue;
        auto &tmp_vec = tmp_N_X[i];
        for (auto &ele: tmp_vec)
            _N_X.emplace_back(ele);
    }
    vector<uint> partition_X_(Partition_v3::max_partition_num_, 0), partition_N_X_(Partition_v3::max_partition_num_, 0),
            partition_Cand_exts(Partition_v3::max_partition_num_, 0),
            partition_Cand_q_(Partition_v3::max_partition_num_, 0);
    Partition_v3 X, N_X, cand_q, Cand_exts;


    getPartition_v3(partition_X_, _X, degrees.data(), Partition_v3::max_partition_num_);
    getPartition_v3(partition_N_X_, _N_X, degrees.data(), Partition_v3::max_partition_num_);
    getPartition_v3(partition_Cand_exts, _Cand_exts, degrees.data(), Partition_v3::max_partition_num_);
    getPartition_v3(partition_Cand_q_, _Cand_q, degrees.data(), Partition_v3::max_partition_num_);

    X.from_vector(_X, partition_X_);
    N_X.from_vector(_N_X, partition_N_X_);
    Cand_exts.from_vector(_Cand_exts, partition_Cand_exts);
    cand_q.from_vector(_Cand_q, partition_Cand_q_);
    
    

    uint *device_degrees, *device_neighbors, *device_neighbors_offset;
    CUDA_ERROR_CHECK(cudaMallocAsync(&device_degrees, sizeof(uint) * degrees.size(), stream));
    CUDA_ERROR_CHECK(cudaMallocAsync(&device_neighbors, sizeof(uint) * neighbors.size(), stream));
    CUDA_ERROR_CHECK(cudaMallocAsync(&device_neighbors_offset, sizeof(uint) * neighbors_offset.size(), stream));

    CUDA_ERROR_CHECK(
            cudaMemcpyAsync(device_degrees, degrees.data(), degrees.size() * sizeof(uint), cudaMemcpyHostToDevice, stream));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
            device_neighbors, neighbors.data(), neighbors.size() * sizeof(uint), cudaMemcpyHostToDevice, stream));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(device_neighbors_offset, neighbors_offset.data(),
                                     neighbors_offset.size() * sizeof(uint), cudaMemcpyHostToDevice, stream));
    vector<uint> Cand_q;
    uint32_t* keys = nullptr;
    uint32_t* size;
    CUDA_ERROR_CHECK(cudaMallocAsync(&size, sizeof(uint32_t), stream));
    CUDA_ERROR_CHECK(cudaMemsetAsync(size, 0, sizeof(uint32_t), stream));

    arrMapTable G_State_arrMap(keys, device_G_arrState, size, graph_size, 0xFFFFFFFF, 0);
    
    
    
    start_time = chrono::high_resolution_clock::now();
    enumerate_GPU(X, N_X, Cand_exts, cand_q, Cand_q, hashTable_G_state, device_G_label, G_State_arrMap, device_G_prune,
                  hashTable_G_temp, device_degrees, device_neighbors, device_neighbors_offset, degrees, neighbors,
                  neighbors_offset, graph_size, k, theta, bipartite_idx, stream);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    auto end      = chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start_time;
    double ms     = elapsed.count();
    fmt::println("Running Time: {} ms", ms);
    std::cout << "\n===== 性能测试报告 =====" << std::endl;
    std::cout << "结果数量(r)\t耗时(ms)" << std::endl;
    for (const auto& record : time_records) {
        std::cout << record.first << "\t\t" << record.second << std::endl;
    }
    std::cout << "========================" << std::endl;
}


int main(int argc, char** argv) {
    args::ArgumentParser parser("GiMB, an algorithm for enumerating all large maximal biplexes\n");

    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Group required(parser, "", args::Group::Validators::All);

    args::ValueFlag<std::string> benchmark_file(parser, "benchmark", "Path to benchmark", {'f', "file"}, "");

    args::ValueFlag<int> Results(parser, "Num of results", "Num of results", {'r', "r"}, 10000000);
    args::ValueFlag<int> mediateResult(parser, "Output miediate results", "Output miediate results", {'m'}, 0);
    args::ValueFlag<int> Quiete(parser, "quiete", "quiete or not", {'q', "q"}, 0);
    args::ValueFlag<int> Device(parser, "device", "Device name", {'d', "d"}, 0);
    args::ValueFlag<int> group_size(parser, "group size for ablation study", "group size for ablation study", {'g', "g"}, -1);
    args::ValueFlag<uint> K(parser, "K", "K", {'k', "k"}, 1);
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 0;
    } catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 0;
    }
    char filepath[1024] = ".........";
    strncpy(filepath, args::get(benchmark_file).c_str(), 1024);
    num           = args::get(Results);
    verbose       = args::get(mediateResult);
    OutputResults = args::get(Quiete);
    uint k = args::get(K);
    for (uint i = 0;; i++) {
        int tmp = pow(10, i);
        if (tmp < num)
            checkpoints.insert(tmp);
        else {
            checkpoints.insert(num);
            break;
        }
    }


    std::filesystem::path benchmark_file_path = filepath;
    gpu_id                                = args::get(Device); 
    CUDA_ERROR_CHECK(cudaSetDevice(gpu_id)); 
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    fmt::println("Using GPU: {}, GPU id: {}", prop.name, gpu_id);

    testEnumerate(benchmark_file_path, k);
    time_recorder.save();
    return 0;
}
