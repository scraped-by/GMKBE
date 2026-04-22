



#ifndef HASHTABLE_H
#define HASHTABLE_H
#include <kcore_24_10_08/common.h>
namespace Utils::HashTable {
    using Key   = uint32_t;
    using Value = uint32_t;

    __constant__ uint32_t PRIME32_1 = 0x9E3779B1;
    __constant__ uint32_t PRIME32_2 = 0x85EBCA77;
    __constant__ uint32_t PRIME32_3 = 0xC2B2AE3D;
    __device__ __forceinline__ uint32_t xxhash32(uint32_t key, uint32_t seed) {
        uint32_t h32 = seed + PRIME32_1;
        h32 += key * PRIME32_2;
        h32 = (h32 ^ (h32 >> 15)) * PRIME32_3;
        h32 ^= h32 >> 13;
        return h32;
    }

    
    __constant__ static const uint32_t c_xxhash32_primes[4] = {
        0x9E3779B1U, 
        0x85EBCA77U, 
        0xC2B2AE3DU, 
        0x27D4EB2FU 
    };

    
    __device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) {
        return (x << r) | (x >> (32 - r));
    }

    
    __device__ __forceinline__ uint32_t xxhash32_key(uint32_t key, uint32_t seed) {
        
        uint32_t acc = seed + c_xxhash32_primes[0];

        
        acc += key * c_xxhash32_primes[1];
        acc = rotl32(acc, 13);
        acc *= c_xxhash32_primes[0];

        
        acc ^= acc >> 15;
        acc += c_xxhash32_primes[2]; 
        acc *= c_xxhash32_primes[3];

        
        acc ^= acc >> 16;
        acc = rotl32(acc, 5);
        return acc;
    }

    __constant__ float A = 0.61803f;

    struct __align__(8) Pair {
        uint32_t key   = -1;
        uint32_t value = -1;
    };

    class cudaHashTable {
    private:
        ushort laneWidth;
        uint32_t laneID;

    public:
        bool isMulHash;
        uint32_t emptyValue        = -1;
        uint32_t emptyKey          = -1;
        Pair* __restrict__ buckets = nullptr;
        uint* size;
        size_t capacity;

        __device__ __forceinline__ cudaHashTable(uint32_t dataLength, float loadFactor, ushort laneWidth, uint32_t laneID,
            const uint32_t emptyKey = -1, const uint32_t emptyValue = -1, const bool isMulHash = true)
            : laneWidth(laneWidth), laneID(laneID), isMulHash(isMulHash), emptyValue(emptyValue), emptyKey(emptyKey) {
            capacity = (size_t) (dataLength / loadFactor);
        }

        template <uint tileSize>
        __device__ __forceinline__ cudaHashTable(cg::thread_block_tile<tileSize> group, uint32_t dataLength, float loadFactor,
            uint32_t emptyValue, uint32_t emptyKey)
            : capacity(dataLength / loadFactor), laneWidth(group.size()), laneID(group.thread_rank()),
              emptyValue(emptyValue), emptyKey(emptyKey) {
            if (laneID == 0) {
                buckets = (Pair*) malloc(capacity * sizeof(Pair));
            }
            group.sync();
            buckets = group.shfl(buckets, 0);
            initialize_buckets();
        }

        template <uint tileSize>
        __device__ __forceinline__ void ini(cg::thread_block_tile<tileSize> group) {
            laneID    = group.thread_rank();
            laneWidth = group.size();
            if (laneID == 0) {
                buckets = (Pair*) malloc(capacity * sizeof(Pair));
            }
            group.sync();
            buckets = group.shfl(buckets);
            initialize_buckets();
        }

        __device__ __forceinline__ void initialize_buckets() {
            if (buckets == nullptr) {
                return;
            }
            for (uint i = laneID; i < capacity; i += laneWidth) {
                buckets[i].key   = emptyKey;
                buckets[i].value = emptyValue;
            }
        }

        __device__ __forceinline__ size_t remainder_hash(const uint32_t& key) {
            return key % capacity;
        }

        __device__ uint32_t hash(uint32_t k) {
            k ^= k >> 16;
            k *= 0x85ebca6b;
            k ^= k >> 13;
            k *= 0xc2b2ae35;
            k ^= k >> 16;
            return k & (capacity - 1);
        }

        __device__ __forceinline__ void insert_and_inc(const uint32_t key, bool aggInc = false) {
            
            auto i = xxhash32_key(key, 1031) % capacity;
            while (true) {
                const uint32_t old_k = atomicCAS(&buckets[i].key, emptyKey, key);
                if (old_k == emptyKey) {
                    atomicCAS(&buckets[i].value, emptyValue, 1);
                    Utils::atomicAggInc(size);
                    return;
                }
                if (key == buckets[i].key) {
                    if (aggInc) {
                        Utils::atomicAggInc(&buckets[i].value);
                    } else {
                        atomicAdd(&buckets[i].value, 1);
                    }
                    return;
                }
                i = (i + 1) % capacity;
            }
        }

        __device__ __forceinline__ bool insert(const uint32_t key, const uint32_t value) {
            
            auto i = xxhash32_key(key, 1031) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&buckets[i].key, emptyKey, key);
                if (old_k == emptyKey) {
                    buckets[i].value = value;
                    Utils::atomicAggInc(size);
                    return true;
                } else if (old_k == key) {
                    return false;
                }
                i = (i + 1) % capacity;
            }
        }

        __device__ __forceinline__ bool insert(const cg::thread_block_tile<2>& group, const uint32_t key, const uint32_t value) {
            
            
            auto i = (xxhash32_key(key, 1031) % capacity
                         + cooperative_groups::__v1::thread_block_tile<2, void>::thread_rank())
                   % capacity;
            while (true) {
                uint32_t old_k = buckets[i].key;
                if (group.any(old_k == key)) {
                    return false;
                }
                auto const empty_mask = group.ballot(old_k == emptyKey);
                if (empty_mask) {
                    auto const candidate =
                        __ffs(empty_mask) - 1; 
                    if (cooperative_groups::__v1::thread_block_tile<2, void>::thread_rank() == candidate) {
                        
                        if (atomicCAS(&buckets[i].key, emptyKey, key) == emptyKey) {
                            Utils::atomicAggInc(size);
                            buckets[i].value = value;
                        }
                    }
                    return true;
                } else {
                    i = (i + 2) % capacity;
                }
            }
        }

        __device__ __forceinline__ size_t get_idx(const uint32_t& key) {
            
            auto i = xxhash32_key(key, 1031) % capacity;
            while (true) {
                const auto cur_k = buckets[i].key;
                if (cur_k == key) {
                    return i;
                }
                if (cur_k == emptyKey) {
                    return -1;
                }
                i = (i + 1) % capacity;
            }
        }

        __device__ __forceinline__ uint32_t get_value(const uint32_t& key) {
            const size_t i = get_idx(key);
            if (i == -1) {
                return emptyValue;
            }
            return buckets[i].value;
        }

        __device__ __forceinline__ bool find_key(const uint32_t& key) {
            return get_value(key) != emptyValue;
        }

        __device__ __forceinline__ uint32_t get_value(cg::thread_block_tile<2> group, uint32_t key) {
            
            auto i = (xxhash32_key(key, 1031) % capacity + group.thread_rank()) % capacity;
            while (true) {
                auto cur_k = buckets[i].key;
                if (group.any(cur_k == key)) {
                    return buckets[i].value;
                } else if (group.any(cur_k == emptyKey)) {
                    return emptyValue;
                }
                i = (i + 2) % capacity;
            }
        }

        __device__ __forceinline__ ~cudaHashTable() {
            if (laneID == 0 && buckets != nullptr) {
                free(buckets);
            }
        }
    };

    __constant__ uint32_t emptyValue = 0xFFFFFFFF;
    __constant__ uint32_t emptyKey   = 0xFFFFFFFF;
    __constant__ uint32_t deletedKey = 0xFFFFFFFE;

    __global__ void cudaHashTable2_insert(uint* keys, uint* values, uint* size, const uint capacity,
        const uint32_t* device_keys, const uint value, const uint data_nums) {
        const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
        const auto grid_size = blockDim.x * gridDim.x;
        for (uint i = idx; i < data_nums; i += grid_size) {
            const uint key = device_keys[i];
            auto slot      = xxhash32_key(key, 1031) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[slot], -1, key);
                if (old_k == -1) {
                    values[slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                } else if (old_k == key) {
                    break;
                }
                slot = (slot + 1) % capacity;
            }
        }
    }

    class cudaHashTable2 {
    public:
        uint32_t* keys;
        uint32_t* values;
        
        uint* size;
        uint capacity;

        cudaHashTable2(uint32_t dataLength, float loadFactor, cudaStream_t stream = nullptr) {
            capacity = static_cast<uint>(dataLength / loadFactor);
            
            CUDA_ERROR_CHECK(cudaMallocAsync(&keys, capacity * sizeof(uint32_t), stream));
            CUDA_ERROR_CHECK(cudaMallocAsync(&values, capacity * sizeof(uint32_t), stream));
            CUDA_ERROR_CHECK(cudaMallocAsync(&size, sizeof(uint), stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(keys, -1, capacity * sizeof(uint), stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(values, -1, capacity * sizeof(uint), stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(size, 0, sizeof(uint), stream));
        }

        __device__ __forceinline__ void insert_and_inc(const uint32_t key, bool aggInc = false) {
            
            auto i = xxhash32_key(key, 1031) % capacity;
            while (true) {
                const uint32_t old_k = atomicCAS(&keys[i], emptyKey, key);
                if (old_k == emptyKey) {
                    atomicCAS(&values[i], emptyValue, 1);
                    Utils::atomicAggInc(size);
                    return;
                }
                if (key == keys[i]) {
                    if (aggInc) {
                        Utils::atomicAggInc(&values[i]);
                    } else {
                        atomicAdd(&values[i], 1);
                    }
                    return;
                }
                i = (i + 1) % capacity;
            }
        }

        __device__ __forceinline__ bool insert(const uint32_t key, const uint32_t value) {
            
            auto i = xxhash32_key(key, 1031) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[i], emptyKey, key);
                if (old_k == emptyKey) {
                    values[i] = value;
                    Utils::atomicAggInc(size);
                    return true;
                } else if (old_k == key) {
                    return false;
                }
                i = (i + 1) % capacity;
            }
        }

        __device__ __forceinline__ bool insert(const cg::thread_block_tile<2>& group, const uint32_t key, const uint32_t value) {
            
            
            auto i = (xxhash32_key(key, 1031) % capacity
                         + cooperative_groups::__v1::thread_block_tile<2, void>::thread_rank())
                   % capacity;
            while (true) {
                uint32_t old_k = keys[i];
                if (group.any(old_k == key)) {
                    return false;
                }
                auto const empty_mask = group.ballot(old_k == emptyKey);
                if (empty_mask) {
                    auto const candidate =
                        __ffs(empty_mask) - 1; 
                    if (cooperative_groups::__v1::thread_block_tile<2, void>::thread_rank() == candidate) {
                        
                        if (atomicCAS(&keys[i], emptyKey, key) == emptyKey) {
                            Utils::atomicAggInc(size);
                            values[i] = value;
                        }
                    }
                    return true;
                } else {
                    i = (i + 2) % capacity;
                }
            }
        }

        __device__ __forceinline__ uint32_t get_idx(const uint32_t& key) {
            
            auto i = xxhash32_key(key, 1031) % capacity;
            while (true) {
                const auto cur_k = keys[i];
                if (cur_k == key) {
                    return i;
                }
                if (cur_k == emptyKey) {
                    return -1;
                }
                i = (i + 1) % capacity;
            }
        }

        __device__ __forceinline__ uint32_t get_value(const uint32_t& key) {
            const uint32_t i = get_idx(key);
            if (i == -1) {
                return emptyValue;
            }
            return values[i];
        }

        __device__ __forceinline__ bool find_key(const uint32_t& key) {
            return get_value(key) != emptyValue;
        }

        __device__ __forceinline__ uint32_t get_value(cg::thread_block_tile<2> group, uint32_t key) {
            
            auto i = (xxhash32_key(key, 1031) % capacity + group.thread_rank()) % capacity;
            while (true) {
                auto cur_k = keys[i];
                if (group.any(cur_k == key)) {
                    return values[i];
                } else if (group.any(cur_k == emptyKey)) {
                    return emptyValue;
                }
                i = (i + 2) % capacity;
            }
        }

        __host__ void insert(
            const uint32_t* device_keys, uint32_t data_num, uint value = 1, cudaStream_t stream = nullptr) {
            cudaHashTable2_insert<<<256, 256, 0, stream>>>(keys, values, size, capacity, device_keys, value, data_num);
        }

        __host__ __device__ ~cudaHashTable2() {}
    };

    __device__ __forceinline__ uint32_t hash_func3a(uint32_t x) {
        x = (x ^ (x >> 16)) * 0x85ebca6b;
        x = (x ^ (x >> 13)) * 0xc2b2ae35;
        x = x ^ (x >> 16);
        return x;
    }

    __device__ __forceinline__ uint32_t hash_func3b(uint32_t x) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 13) ^ x) * 0x9E3779B1;
        x = (x >> 16) ^ x;
        return x;
    }

    __device__ __forceinline__ uint32_t hash_func3c(uint32_t x) {
        x = (x * 0xabcdef) % 2;
        return x;
    }

    
    __global__ void cudaHashTable3_insert(uint* keys, uint* values, uint* size, const uint capacity,
        const uint32_t* device_keys, const uint value, const uint data_num) {
        const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
        const auto grid_size = blockDim.x * gridDim.x;
        for (uint i = idx; i < data_num; i += grid_size) {
            const uint key = device_keys[i];
            auto slot1     = hash_func3a(key) % capacity;
            uint32_t old_k = atomicCAS(&keys[slot1], -1, key);
            if (old_k == -1) {
                values[slot1] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto slot2 = hash_func3b(key) % capacity;
            old_k      = atomicCAS(&keys[slot2], -1, key);
            if (old_k == -1) {
                values[slot2] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
                if (old_k == -1) {
                    values[current_slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }
    }

    
    __global__ void cudaHashTable3_insert(uint* keys, uint* values, uint* size, const uint capacity,
        const uint32_t* device_keys, const uint* device_valus, const uint data_num) {
        const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
        const auto grid_size = blockDim.x * gridDim.x;

        for (uint i = idx; i < data_num; i += grid_size) {
            const uint key   = device_keys[i];
            const uint value = device_valus[i];
            auto slot1       = hash_func3a(key) % capacity;
            uint32_t old_k   = atomicCAS(&keys[slot1], -1, key);
            if (old_k == -1) {
                values[slot1] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto slot2 = hash_func3b(key) % capacity;
            old_k      = atomicCAS(&keys[slot2], -1, key);
            if (old_k == -1) {
                values[slot2] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
                if (old_k == -1) {
                    values[current_slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }
    }

    
    __global__ void cudaHashTable3_insert(
        uint* keys, uint* values, uint* size, const uint capacity, const uint32_t* device_keys, const uint data_num) {
        const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
        const auto grid_size = blockDim.x * gridDim.x;

        for (uint i = idx; i < data_num; i += grid_size) {
            const uint key   = device_keys[i];
            const uint value = i;
            auto slot1       = hash_func3a(key) % capacity;
            uint32_t old_k   = atomicCAS(&keys[slot1], -1, key);
            if (old_k == -1) {
                values[slot1] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto slot2 = hash_func3b(key) % capacity;
            old_k      = atomicCAS(&keys[slot2], -1, key);
            if (old_k == -1) {
                values[slot2] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
                if (old_k == -1) {
                    values[current_slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }
    }

    __global__ void cudaHashTable3_condition_insert(uint* keys, uint* values, uint* size, const uint capacity,
        const uint32_t* device_keys, const uint32_t* conditions, const uint value, const uint data_num) {
        const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
        const auto grid_size = blockDim.x * gridDim.x;
        for (uint i = idx; i < data_num; i += grid_size) {
            if (conditions[i] == 0) {
                continue;
            }
            const uint key = device_keys[i];
            auto slot1     = hash_func3a(key) % capacity;
            uint32_t old_k = atomicCAS(&keys[slot1], -1, key);
            if (old_k == -1) {
                values[slot1] = value;
                Utils::atomicAggInc(size);
                continue;
            } else if (old_k == key) {
                continue;
            }
            auto slot2 = hash_func3b(key) % capacity;
            old_k      = atomicCAS(&keys[slot2], -1, key);
            if (old_k == -1) {
                values[slot2] = value;
                Utils::atomicAggInc(size);
                continue;
            } else if (old_k == key) {
                continue;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
                if (old_k == -1) {
                    values[current_slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                } else if (old_k == key) {
                    break;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }
    }
    __global__ void cudaHashTable3_condition_insert(uint* keys, uint* values, uint* size, const uint capacity,
        const uint32_t* device_keys, const bool* conditions, const uint value, const uint data_num) {
        const auto idx       = blockIdx.x * blockDim.x + threadIdx.x;
        const auto grid_size = blockDim.x * gridDim.x;
        for (uint i = idx; i < data_num; i += grid_size) {
            if (!conditions[i]) {
                continue;
            }
            const uint key = device_keys[i];
            auto slot1     = hash_func3a(key) % capacity;
            uint32_t old_k = atomicCAS(&keys[slot1], -1, key);
            if (old_k == -1) {
                values[slot1] = value;
                Utils::atomicAggInc(size);
                continue;
            } else if (old_k == key) {
                continue;
            }
            auto slot2 = hash_func3b(key) % capacity;
            old_k      = atomicCAS(&keys[slot2], -1, key);
            if (old_k == -1) {
                values[slot2] = value;
                Utils::atomicAggInc(size);
                continue;
            } else if (old_k == key) {
                continue;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
                if (old_k == -1) {
                    values[current_slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                } else if (old_k == key) {
                    break;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }
    }

    __global__ void cudaHashTable3_insert_idx(uint* keys, uint* values, uint* size, const uint capacity,
        const uint* partitions, G_pointers device_graph, uint idx, const uint value) {
        const uint tid                = threadIdx.x + blockIdx.x * blockDim.x;
        uint grid_size          = blockDim.x * gridDim.x;
        uint node               = partitions[idx];
        const uint data_num     = device_graph.degrees[node];
        const uint* device_keys = device_graph.neighbors + device_graph.neighbors_offset[node];
        for (uint i = tid; i < data_num; i += grid_size) {
            const uint key = device_keys[i];
            auto slot1     = hash_func3a(key) % capacity;
            uint32_t old_k = atomicCAS(&keys[slot1], -1, key);
            if (old_k == -1) {
                values[slot1] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto slot2 = hash_func3b(key) % capacity;
            old_k      = atomicCAS(&keys[slot2], -1, key);
            if (old_k == -1) {
                values[slot2] = value;
                Utils::atomicAggInc(size);
                continue;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                uint32_t old_k = atomicCAS(&keys[current_slot], -1, key);
                if (old_k == -1) {
                    values[current_slot] = value;
                    Utils::atomicAggInc(size);
                    break;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }
    }

    template <typename ValueType>
    __global__ void cudaHashTable3_gpu_viewer_insert(uint* datas, uint num, ValueType value, uint*keys, uint* values, uint *size) {
        const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint grid_size = blockDim.x * gridDim.x;
        for (uint i = tid; i < num; i += grid_size) {

        }

    }

    template <typename ValueType>
    struct cudaHashTable3_gpu_viewer2 {
    public:
        uint32_t* keys;
        ValueType* values;
        uint* size;
        const uint capacity; 
        const ValueType emptyValue;
        const uint32_t emptyKey;
        const uint32_t deletedKey;

        
        
        
        __host__ cudaHashTable3_gpu_viewer2(uint32_t* keys, ValueType* values, uint* size, uint capacity,
            uint32_t emptyKey = 0xFFFFFFFF, ValueType emptyValue = ValueType{}, uint32_t deletedKey = 0xFFFFFFFE)
            : keys(keys), values(values), size(size), capacity(capacity), emptyKey(emptyKey), emptyValue(emptyValue), deletedKey(deletedKey) {

            
            static_assert(sizeof(ValueType) == 4 || sizeof(ValueType) == 8,
                "ValueType must be 32-bit or 64-bit for CUDA atomic operations");

            
            
        }

    public:
        

        
        
        __device__ __forceinline__ uint32_t get_idx(const uint32_t& key) const { 
            
            auto slot1 = hash_func3a(key) % capacity;
            auto cur_k = keys[slot1];
            if (cur_k == key) return slot1;
            if (cur_k == emptyKey) return capacity; 

            
            auto slot2 = hash_func3b(key) % capacity;
            cur_k = keys[slot2];
            if (cur_k == key) return slot2;
            if (cur_k == emptyKey) return capacity; 

            
            
            auto pivot_slot = (hash_func3c(key) == 0 ? slot1 : slot2);
            for (uint i = 1; i < capacity; ++i) {
                auto current_slot = (pivot_slot + i) % capacity;
                cur_k = keys[current_slot];
                if (cur_k == key) return current_slot;
                if (cur_k == emptyKey) return capacity; 
                
            }

            return capacity; 
        }

        
        __device__ __forceinline__ ValueType get_value(const uint32_t& key) const { 
            const uint32_t idx = get_idx(key);
            return (idx == capacity) ? emptyValue : values[idx];
        }

        
        
        
        

        
        
        __device__ __forceinline__ bool set_value(const uint32_t& key, const ValueType value) {
            uint32_t old_k_dummy; 
            return insert_internal(key, value, 0, false, &old_k_dummy);
        }

        
        __device__ __forceinline__ bool add(const uint32_t& key, ValueType add_num = ValueType(1)) {
            uint32_t old_k_dummy;
            return insert_internal(key, add_num, add_num, true, &old_k_dummy);
        }

        
        __device__ __forceinline__ bool insert(uint32_t key, ValueType value) {
            uint32_t old_k;
            bool success = insert_internal(key, value, 0, false, &old_k);
            
            return success && (old_k == emptyKey || old_k == deletedKey);
        }

    private:
        
        __device__ __forceinline__ bool insert_internal(const uint32_t key, ValueType value_on_insert, ValueType value_on_update, bool is_atomic_add, uint32_t* old_key_out) {
            auto try_claim_and_operate = [&](uint32_t slot) -> int {
                
                uint32_t old_k = keys[slot];

                
                while (true) {
                    if (old_k == key) { 
                        if (is_atomic_add) {
                            atomicAdd(&values[slot], value_on_update);
                        } else {
                            values[slot] = value_on_insert;
                        }
                        *old_key_out = key;
                        return 1; 
                    }

                    if (old_k == emptyKey || old_k == deletedKey) { 
                        uint32_t cas_old_val = old_k;
                        uint32_t returned_val = atomicCAS(&keys[slot], cas_old_val, key);
                        if (returned_val == cas_old_val) { 
                            if (is_atomic_add) {
                                atomicAdd(&values[slot], value_on_insert);
                            } else {
                                values[slot] = value_on_insert;
                            }
                            if (cas_old_val == emptyKey) {
                                Utils::atomicAggInc(size);
                            }
                            *old_key_out = cas_old_val;
                            return 1; 
                        }
                        
                        old_k = returned_val;
                        continue;
                    }

                    
                    return 0;
                }
            };

            
            int result = try_claim_and_operate(hash_func3a(key) % capacity);
            if (result != 0) return result == 1;

            result = try_claim_and_operate(hash_func3b(key) % capacity);
            if (result != 0) return result == 1;

            auto pivot_slot = (hash_func3c(key) == 0 ? hash_func3a(key) : hash_func3b(key)) % capacity;
            for (uint i = 1; i < capacity; ++i) {
                auto current_slot = (pivot_slot + i) % capacity;
                result = try_claim_and_operate(current_slot);
                if (result != 0) return result == 1;
            }
            return false; 
        }


    public:
        
        __device__ __forceinline__ bool delete_element(const uint32_t& key) {
            const uint32_t idx = get_idx(key);
            if (idx == capacity) return false;

            
            uint32_t old_key = atomicCAS(&keys[idx], key, deletedKey);

            if (old_key == key) {
                values[idx] = emptyValue;
                atomicAggInc(size, static_cast<uint>(-1));
                return true;
            }
            return false;
        }

        
        __device__ __forceinline__ void atomic_dec(const uint32_t& key, const ValueType& dec_value = ValueType(-1)) {
            const uint tableIdx = get_idx(key);
            if (tableIdx != capacity) {
                
                
                if (atomicAdd(&values[tableIdx], dec_value) + dec_value == 0) {
                    
                    keys[tableIdx] = deletedKey;
                    
                    
                }
            }
        }

        __device__ __forceinline__ bool find_key(const uint32_t& key) const { 
            return get_idx(key) != capacity;
        }

        __device__ __forceinline__ bool find_key_and_add_value(const uint32_t& key, const ValueType add_num = ValueType(1)) {
            const uint32_t idx = get_idx(key);
            if (idx != capacity) {
                atomicAdd(&values[idx], add_num);
                return true;
            }
            return false;
        }
    };

    template <typename ValueType>
    struct cudaHashTable3_gpu_viewer {
    private:
        __device__ __forceinline__ bool insert_internal(const uint32_t key, ValueType value_on_insert, ValueType value_on_update, bool is_atomic_add, uint32_t* old_key_out) {
            auto try_claim_and_operate = [&](uint32_t slot) -> int {
                
                uint32_t old_k = keys[slot];

                
                while (true) {
                    if (old_k == key) { 
                        if (is_atomic_add) {
                            atomicAdd(&values[slot], value_on_update);
                        } else {
                            values[slot] = value_on_insert;
                        }
                        *old_key_out = key;
                        return 1; 
                    }

                    if (old_k == emptyKey || old_k == deletedKey) { 
                        uint32_t cas_old_val = old_k;
                        uint32_t returned_val = atomicCAS(&keys[slot], cas_old_val, key);
                        if (returned_val == cas_old_val) { 
                            if (is_atomic_add) {
                                atomicAdd(&values[slot], value_on_insert);
                            } else {
                                values[slot] = value_on_insert;
                            }
                            if (cas_old_val == emptyKey) {
                                Utils::atomicAggInc(size);
                            }
                            *old_key_out = cas_old_val;
                            return 1; 
                        }
                        
                        old_k = returned_val;
                        continue;
                    }

                    
                    return 0;
                }
            };

            
            int result = try_claim_and_operate(hash_func3a(key) % capacity);
            if (result != 0) return result == 1;

            result = try_claim_and_operate(hash_func3b(key) % capacity);
            if (result != 0) return result == 1;

            auto pivot_slot = (hash_func3c(key) == 0 ? hash_func3a(key) : hash_func3b(key)) % capacity;
            for (uint i = 1; i < capacity; ++i) {
                auto current_slot = (pivot_slot + i) % capacity;
                result = try_claim_and_operate(current_slot);
                if (result != 0) return result == 1;
            }
            return false; 
        }

    public:
        uint32_t* keys;
        ValueType* values;
        uint* size;
        uint capacity;
        ValueType emptyValue; 
        uint32_t emptyKey;
        uint32_t deletedKey;


        __host__ cudaHashTable3_gpu_viewer(uint32_t* keys, ValueType* values, uint* size, uint capacity,
            uint32_t emptyKey = 0xFFFFFFFF, ValueType emptyValue = ValueType(-1), uint32_t deletedKey = 0xFFFFFFFE) 
            : keys(keys), values(values), size(size), capacity(capacity), emptyKey(emptyKey), emptyValue(emptyValue), deletedKey(deletedKey) {

            
            static_assert(sizeof(ValueType) == 4 || sizeof(ValueType) == 8,
                "ValueType must be 32-bit or 64-bit for CUDA atomic operations");
        }

        __device__ __forceinline__ uint32_t get_idx(const uint32_t& key) const { 
            
            auto slot1 = hash_func3a(key) % capacity;
            auto cur_k = keys[slot1];
            if (cur_k == key) return slot1;
            if (cur_k == emptyKey) return capacity; 

            
            auto slot2 = hash_func3b(key) % capacity;
            cur_k = keys[slot2];
            if (cur_k == key) return slot2;
            if (cur_k == emptyKey) return capacity; 
            
            auto pivot_slot = (hash_func3c(key) == 0 ? slot1 : slot2);
            for (uint i = 1; i < capacity; ++i) {
                auto current_slot = (pivot_slot + i) % capacity;
                cur_k = keys[current_slot];
                if (cur_k == key) return current_slot;
                if (cur_k == emptyKey) return capacity; 
                
            }
            return capacity; 
        }

        __device__ __forceinline__ ValueType get_value(const uint32_t& key) const { 
            const uint32_t idx = get_idx(key);
            return (idx == capacity) ? emptyValue : values[idx];
        }

        
        __device__ __forceinline__ bool set_value(const uint32_t& key, const ValueType value) {
            uint32_t old_k_dummy; 
            return insert_internal(key, value, 0, false, &old_k_dummy);
        }


        __device__ __forceinline__ bool delete_element(const uint32_t& key) {
            const uint32_t idx = get_idx(key);
            if (idx == capacity) return false;

            
            uint32_t old_key = atomicCAS(&keys[idx], key, deletedKey);

            if (old_key == key) {
                values[idx] = emptyValue;
                atomicAggInc(size, static_cast<uint>(-1));
                return true;
            }
            return false;
        }

        __device__ __forceinline__ bool atomic_dec(const uint32_t& key, const ValueType& value = ValueType(1)) {
            const uint tableIdx = get_idx(key);
            if (tableIdx == capacity) return false;
            const ValueType dec_value = -1 * value;
            if (atomicAdd(&values[tableIdx], dec_value) + dec_value == 0) {
                keys[tableIdx] = deletedKey;
            }
            return true;
        }

        __device__ __forceinline__ bool dec(const uint32_t& key, const ValueType& value = ValueType(1)) {
            const uint tableIdx = get_idx(key);
            if (tableIdx == capacity)
                return false;
            const ValueType dec_value = -1 * value;
            if (tableIdx != capacity) {
                if (values[tableIdx] + dec_value == 0)
                    keys[tableIdx] = emptyKey;
                values[tableIdx] += dec_value;
            }
            return true;
        }

        __device__ __forceinline__ bool find_key(const uint32_t& key) const{
            return get_idx(key) != capacity;
        }

        __device__ __forceinline__ bool find_key_and_add_value(const uint32_t& key, const ValueType add_num = ValueType(1)) {
            const uint32_t idx = get_idx(key);
            if (idx != capacity) {
                atomicAdd(&values[idx], add_num);
                return true;
            }
            return false;
        }

        
        __device__ __forceinline__ bool add(const uint32_t& key, ValueType add_num = ValueType(1)) {
            uint32_t old_k_dummy;
            return insert_internal(key, add_num, add_num, true, &old_k_dummy);
        }

        
        __device__ __forceinline__ bool insert(uint32_t key, ValueType value) {
            uint32_t old_k;
            bool success = insert_internal(key, value, 0, false, &old_k);

            
            return success && (old_k == emptyKey || old_k == deletedKey);
        }

        __device__ __forceinline__ bool insert_increase(const uint key) {
            assert(emptyValue == 0);

            auto try_slot = [&](uint32_t slot) -> bool {
                uint32_t old_k = atomicCAS(&keys[slot], emptyKey, key);
                if (old_k == emptyKey) {
                    
                    atomicAdd(&values[slot], 1);
                    Utils::atomicAggInc(size);
                    return true; 
                } else if (old_k == key) {
                    
                    atomicAdd(&values[slot], 1);
                    return true; 
                }
                return false; 
            };

            
            int result = try_slot(hash_func3a(key) % capacity);
            if (result) return true;

            
            result = try_slot(hash_func3b(key) % capacity);
            if (result) return true;

            
            auto pivot_slot = hash_func3c(key) == 0 ?
                             hash_func3a(key) % capacity :
                             hash_func3b(key) % capacity;
            auto current_slot = (pivot_slot + 1) % capacity;

            while (current_slot != pivot_slot) {
                result = try_slot(current_slot);
                if (result) return true;
                current_slot = (current_slot + 1) % capacity;
            }
            return false; 
        }
    };

    













    template <typename ValueType>
    class cudaArrayMap_gpu_viewer {
    private:
        bool flag = false;

    public:
        
        uint32_t* keys;     
        ValueType* values;  
        uint* size;         
        uint capacity;      
        ValueType emptyValue; 
        uint32_t emptyKey;  
        uint32_t deletedKey;

        










        __host__ cudaArrayMap_gpu_viewer(const uint capacity, cudaStream_t stream = nullptr, uint32_t emptyKey = 0xFFFFFFFF, ValueType emptyValue = ValueType(-1), uint32_t deletedKey = 0xFFFFFFFE):capacity(capacity) {
            flag = true;
            CUDA_ERROR_CHECK(cudaMallocAsync(&values, capacity * sizeof(ValueType), stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(values, 0, sizeof(ValueType) * capacity, stream));
            CUDA_ERROR_CHECK(cudaMallocAsync(&size, sizeof(uint32_t), stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(size, 0, sizeof(uint32_t), stream));
            static_assert(sizeof(ValueType) == 4 || sizeof(ValueType) == 8,
                "ValueType must be 32-bit or 64-bit for CUDA atomic operations");
        }
        __host__ cudaArrayMap_gpu_viewer(uint32_t* keys, ValueType* values, uint* size, const uint capacity,
            const uint32_t emptyKey = 0xFFFFFFFF, ValueType emptyValue = ValueType(-1), const uint32_t deletedKey = 0xFFFFFFFE)
            : keys(keys), values(values), size(size), capacity(capacity), emptyKey(emptyKey), emptyValue(emptyValue), deletedKey(deletedKey) {

            static_assert(sizeof(ValueType) == 4 || sizeof(ValueType) == 8,
                "ValueType must be 32-bit or 64-bit for CUDA atomic operations");
        }
        __host__ void clean() {
            if (flag) {
                CUDA_ERROR_CHECK(cudaFree(values));
                CUDA_ERROR_CHECK(cudaFree(size));
            }
        }
        



        __device__ __forceinline__ uint32_t get_idx(const uint32_t& key) const {
            
            
            
            
            if (key < capacity) {
                return key;
            }
            return capacity; 
        }

        



        __device__ __forceinline__ ValueType get_value(const uint32_t& key) const {
            if (key >= capacity) {
                return emptyValue; 
            }
            return values[key]; 
        }

        



        __device__ __forceinline__ bool set_value(const uint32_t& key, const ValueType value) {
            if (key >= capacity) {
                return false;
            }
            
            ValueType old_val = atomicExch(&values[key], value);

            
            if (old_val == emptyValue && value != emptyValue) {
                atomicAdd(size, 1); 
            } else if (old_val != emptyValue && value == emptyValue) {
                atomicSub(size, 1); 
            }
            return true;
        }

        



        __device__ __forceinline__ bool add(const uint32_t& key, ValueType add_num) {
            if (key >= capacity) {
                return false;
            }

            
            
            ValueType old_val = values[key]; 
            while (true) {
                if (old_val == emptyValue) {
                    
                    ValueType cas_return = atomicCAS(&values[key], emptyValue, add_num);
                    if (cas_return == emptyValue) {
                        
                        atomicAdd(size, 1);
                        return true;
                    }
                    
                    
                    old_val = cas_return;
                } else {
                    
                    atomicAdd(&values[key], add_num);
                    return true;
                }
            }
        }

        



        __device__ __forceinline__ bool insert(uint32_t key, ValueType value) {
            if (key >= capacity || value == emptyValue) {
                return false; 
            }

            
            ValueType cas_return = atomicCAS(&values[key], emptyValue, value);

            if (cas_return == emptyValue) {
                
                atomicAdd(size, 1);
                return true;
            }

            
            return false;
        }

        



        __device__ __forceinline__ bool delete_element(const uint32_t& key) {
            if (key >= capacity) {
                return false;
            }
            
            ValueType old_val = atomicExch(&values[key], emptyValue);

            
            if (old_val != emptyValue) {
                atomicSub(size, 1);
                return true;
            }

            
            return false;
        }

        



        __device__ __forceinline__ void delete_usingKey_without_decrease(const uint32_t& key) {
            if (key < capacity) {
                values[key] = emptyValue; 
            }
        }

        



        __device__ __forceinline__ void atomic_dec(const uint32_t& key, const ValueType& value = ValueType(1)) {
            if (key >= capacity) {
                return;
            }
            
            if (atomicSub(&values[key], value) == value) {
                
                
                ValueType current_val = 0;
                if (atomicCAS(&values[key], current_val, emptyValue) == current_val) {
                    atomicSub(size, 1);
                }
            }
        }

        


        __device__ __forceinline__ bool dec(const uint32_t& key, const ValueType& value = ValueType(1)) {
            if (key >= capacity) {
                return false;
            }
            ValueType old_val = values[key];
            if (old_val != emptyValue) {
                values[key] -= value;
                if (values[key] == 0 && old_val != 0) { 
                    values[key] = emptyValue;
                    atomicSub(size, 1); 
                }
            }
            return true;
        }

        



        __device__ __forceinline__ bool find_key(const uint32_t& key) const {
            if (key >= capacity) {
                return false;
            }
            return values[key] != emptyValue;
        }

        



        __device__ __forceinline__ bool find_key_and_add_value(const uint32_t& key, const ValueType add_num = ValueType(1)) {
            if (key >= capacity) {
                return false;
            }

            ValueType old_val = values[key]; 
            
            while (old_val != emptyValue) {
                ValueType new_val = old_val + add_num;
                ValueType cas_return = atomicCAS(&values[key], old_val, new_val);
                if (cas_return == old_val) {
                    
                    return true;
                }
                
                old_val = cas_return;
            }

            
            return false;
        }

        



        __device__ __forceinline__ bool insert_increase(const uint key) {
            
            return add(key, static_cast<ValueType>(1));
        }

    private:
        
        
        
        
    };


    class cudaHashTable3 {
        



    public:
        void* d_buffer = nullptr;
        size_t totalBufferSize = 0;

        uint32_t* keys = nullptr;
        uint32_t* values = nullptr;
        
        uint* size = nullptr;
        uint capacity;

        cudaHashTable3() = default;

        cudaHashTable3(uint32_t dataLength, float loadFactor, cudaStream_t stream = nullptr) {
            initialize(dataLength, loadFactor, stream);
        }
        
        
        
        
        
        
        
        
        
        inline size_t alignUp(size_t offset, size_t alignment) {
            return (offset + alignment - 1) & ~(alignment - 1);
        }
        __host__ void initialize(uint32_t dataLength, float loadFactor, cudaStream_t stream = nullptr) {
            capacity = static_cast<uint>(dataLength / loadFactor);

            
            const size_t ALIGNMENT = 256;

            
            size_t keysSizeRaw = capacity * sizeof(uint32_t);
            size_t valuesSizeRaw = capacity * sizeof(uint32_t);
            size_t sizeVarRaw = sizeof(uint);

            
            
            size_t offset_keys = 0;

            
            size_t offset_values = alignUp(offset_keys + keysSizeRaw, ALIGNMENT);

            
            size_t offset_size = alignUp(offset_values + valuesSizeRaw, ALIGNMENT);

            
            totalBufferSize = offset_size + sizeVarRaw;
            
            totalBufferSize = alignUp(totalBufferSize, ALIGNMENT);

            
            CUDA_ERROR_CHECK(cudaMallocAsync(&d_buffer, totalBufferSize, stream));

            
            
            uint8_t* ptr_base = static_cast<uint8_t*>(d_buffer);

            keys = reinterpret_cast<uint32_t*>(ptr_base + offset_keys);
            values = reinterpret_cast<uint32_t*>(ptr_base + offset_values);
            size = reinterpret_cast<uint*>(ptr_base + offset_size);

            
            
            
            
            CUDA_ERROR_CHECK(cudaMemsetAsync(keys, -1, keysSizeRaw, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(values, -1, valuesSizeRaw, stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(size, 0, sizeVarRaw, stream));
        }

        __host__ void insert(
            const uint32_t* device_keys, uint32_t data_num, uint value = 1, cudaStream_t stream = nullptr) {
            if (data_num == 0) return;
            uint gridDim = min(1024, CALC_GRID_DIM(data_num, THREADS_PER_BLOCK));
            cudaHashTable3_insert<<<gridDim, THREADS_PER_BLOCK, 0, stream>>>(keys, values, size, capacity, device_keys, value, data_num);
        }

        __host__ void insert(
            const uint32_t* device_keys, uint* device_values, uint32_t data_num, cudaStream_t stream = nullptr) {
            cudaHashTable3_insert<<<512, 256, 0, stream>>>(
                keys, values, size, capacity, device_keys, device_values, data_num);
        }

        __host__ void insert(const uint32_t* device_keys, uint32_t data_num, cudaStream_t stream = nullptr) {
            cudaHashTable3_insert<<<512, 256, 0, stream>>>(keys, values, size, capacity, device_keys, data_num);
        }


        __host__ void condition_insert(const uint32_t* device_keys, const uint32_t* device_conditions,
            uint32_t data_num, uint value = 1, cudaStream_t stream = nullptr) {
            cudaHashTable3_condition_insert<<<512, 256, 0, stream>>>(
                keys, values, size, capacity, device_keys, device_conditions, value, data_num);
        }
        __host__ void condition_insert(const uint32_t* device_keys, const bool* device_conditions, uint32_t data_num,
            uint value = 1, cudaStream_t stream = nullptr) {
            cudaHashTable3_condition_insert<<<512, 256, 0, stream>>>(
                keys, values, size, capacity, device_keys, device_conditions, value, data_num);
        }

        __host__ void insert_idx(const uint32_t* partitions, uint32_t idx, G_pointers device_graph, uint value = 1,
            cudaStream_t stream = nullptr) {
            cudaHashTable3_insert_idx<<<512, 256, 0, stream>>>(
                keys, values, size, capacity, partitions, device_graph, idx, value);
        }

        __host__ void clear_insert(cudaStream_t stream = nullptr) {
            CUDA_ERROR_CHECK(cudaMemsetAsync(keys, -1, capacity * sizeof(uint), stream));
            CUDA_ERROR_CHECK(cudaMemsetAsync(values, -1, capacity * sizeof(uint), stream));
        }

        __host__ cudaHashTable3_gpu_viewer<uint32_t> get_viewer() const {
            return {keys, values, size, capacity};
        }

        __host__ ~cudaHashTable3() {
            if (d_buffer) {
                CUDA_ERROR_CHECK(cudaFree(d_buffer));
            }
            
            
            
            
            
            
            
            
            
        }
    };

    __device__ __forceinline__ void initialize(Pair*& __restrict__ buckets, const uint laneID, const size_t capacity, const uint laneWidth,
        const uint32_t emptyKey, const uint32_t emptyValue) {
        if (buckets == nullptr) {
            return;
        }
        for (int i = laneID; i < capacity; i += laneWidth) {
            buckets[i].key   = emptyKey;
            buckets[i].value = emptyValue;
        }
    }

    __device__ __forceinline__ cuda::std::tuple<Pair*, uint*> initializeHashTableWithinGrid(const cg::grid_group& grid,
        const size_t capacity, uint64_t* ptr, const uint32_t emptyKey = -1, const uint32_t emptyValue = -1) {
        if (cooperative_groups::__v1::grid_group::thread_rank() == 0) {
            auto tmp1 = malloc(capacity * sizeof(Pair)), tmp2 = malloc(sizeof(uint));
            assert(tmp1 != nullptr);
            assert(tmp2 != nullptr);
            (*ptr)     = reinterpret_cast<uint64_t>(tmp1);
            *(ptr + 1) = reinterpret_cast<uint64_t>(tmp2);
        }
        grid.sync();
        auto bucketPtr = reinterpret_cast<Pair*>(*ptr);
        auto sizePtr   = reinterpret_cast<uint*>(*(ptr + 1));
        if (cooperative_groups::__v1::grid_group::thread_rank() == 0) {
            sizePtr[0] = 0;
        }
        initialize(bucketPtr, grid.thread_rank(), capacity, grid.size(), emptyKey, emptyValue);
        return {bucketPtr, sizePtr};
    }
} 

namespace Utils::MultiHash {
    struct cudaMultiHashTable_gpu_viewer {
    private:
        
        static const uint EMPTY_LOCK_SIG = 0xFFFFFFFF;
    public:
        uint32_t* keys; 
        int32_t* values; 
        uint* size; 
        uint* locks; 
        uint capacity_per_table; 
        uint num_tables; 
        uint emptyKey      = -1;
        int32_t emptyValue = 0; 

        
        __host__ cudaMultiHashTable_gpu_viewer(
            uint32_t* keys, int32_t* values, uint* size, uint* locks, uint capacity_per_table, uint num_tables) 
            : keys(keys), values(values), size(size), capacity_per_table(capacity_per_table), num_tables(num_tables), locks(locks) {
        }

        __device__ __forceinline__ uint hash_state_id(uint state_id) {
            
            state_id = ((state_id >> 16) ^ state_id) * 0x45d9f3b;
            state_id = ((state_id >> 16) ^ state_id) * 0x45d9f3b;
            state_id = (state_id >> 16) ^ state_id;
            return state_id % num_tables;
        }

        __device__ __forceinline__ int get_or_assign_table_id(uint state_id) {
            if (state_id == EMPTY_LOCK_SIG) {
                return -1; 
            }

            uint start_pos   = hash_state_id(state_id);
            uint current_pos = start_pos;

            do {
                uint current_state = locks[current_pos];

                
                if (current_state == state_id) {
                    return current_pos;
                }

                
                if (current_state == EMPTY_LOCK_SIG) {
                    if (atomicCAS(&locks[current_pos], EMPTY_LOCK_SIG, state_id) == EMPTY_LOCK_SIG) {
                        return current_pos;
                    }
                    
                    if (locks[current_pos] == state_id) {
                        return current_pos;
                    }
                }

                
                current_pos = (current_pos + 1) % num_tables;
            } while (current_pos != start_pos); 

            return -1; 
        }

        
        __device__ __forceinline__ int get_table_id(uint state_id) {
            if (state_id == EMPTY_LOCK_SIG) {
                return -1;
            }

            uint start_pos   = hash_state_id(state_id);
            uint current_pos = start_pos;

            do {
                if (locks[current_pos] == state_id) {
                    return current_pos;
                }

                
                if (locks[current_pos] == EMPTY_LOCK_SIG) {
                    return -1;
                }

                current_pos = (current_pos + 1) % num_tables;
            } while (current_pos != start_pos);

            return -1; 
        }

        
        __device__ __forceinline__ uint32_t get_idx(const uint32_t& key, uint state_id) {
            int table_id = get_table_id(state_id);
            if (table_id == -1) {
                return -1; 
            }

            uint32_t base_offset = table_id * capacity_per_table; 

            
            auto slot1        = HashTable::hash_func3a(key) % capacity_per_table;
            auto global_slot1 = base_offset + slot1;
            auto cur_k        = keys[global_slot1];
            if (cur_k == key) {
                return global_slot1;
            }
            if (cur_k == HashTable::emptyKey) {
                return -1;
            }

            
            auto slot2        = HashTable::hash_func3b(key) % capacity_per_table;
            auto global_slot2 = base_offset + slot2;
            cur_k             = keys[global_slot2];
            if (cur_k == key) {
                return global_slot2;
            }
            if (cur_k == HashTable::emptyKey) {
                return -1;
            }

            
            auto current_slot        = ((HashTable::hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity_per_table;
            auto global_current_slot = base_offset + current_slot;

            uint32_t probe_count = 0;
            while (probe_count < capacity_per_table) {
                
                cur_k = keys[global_current_slot];
                if (cur_k == key) {
                    return global_current_slot;
                }
                if (cur_k == HashTable::emptyKey) {
                    return -1;
                }
                current_slot        = (current_slot + 1) % capacity_per_table;
                global_current_slot = base_offset + current_slot;
                probe_count++;
            }
            return -1; 
        }

        
        __device__ __forceinline__ int32_t get_value(const uint32_t& key, uint state_id) {
            const uint32_t i = get_idx(key, state_id);
            if (i == static_cast<uint32_t>(-1)) {
                return emptyValue;
            }
            return values[i];
        }

        
        __device__ __forceinline__ bool find_key(const uint32_t& key, uint state_id) {
            return get_value(key, state_id) != emptyValue;
        }

        
        __device__ __forceinline__ bool insert(const uint32_t& key, const int32_t& value, uint state_id) { 
            if (key == emptyKey) {
                return false;
            }
            int table_id = get_or_assign_table_id(state_id);
            if (table_id == -1) {
                return false; 
            }
            uint32_t base_offset = table_id * capacity_per_table;

            
            uint32_t existing_idx = get_idx(key, state_id);
            if (existing_idx != UINT32_MAX) {
                
                atomicExch(&values[existing_idx], value);
                return true;
            }

            
            
            uint32_t candidate_slots[3];
            candidate_slots[0] = base_offset + (HashTable::hash_func3a(key) % capacity_per_table);
            candidate_slots[1] = base_offset + (HashTable::hash_func3b(key) % capacity_per_table);

            auto start_slot    = (HashTable::hash_func3c(key) == 0 ? HashTable::hash_func3a(key) % capacity_per_table
                                                                   : HashTable::hash_func3b(key) % capacity_per_table);
            candidate_slots[2] = base_offset + ((start_slot + 1) % capacity_per_table);

            
            uint32_t expected = emptyKey;
            for (int i = 0; i < 2; i++) {
                uint32_t slot = candidate_slots[i];
                
                if (atomicCAS(&keys[slot], expected, key) == expected) {
                    
                    values[slot] = value;
                    atomicAdd(&size[table_id], 1);
                    return true;
                }
                
                if (keys[slot] == key) {
                    atomicExch(&values[slot], value);
                    return true;
                }
            }
            
            uint32_t current_slot = candidate_slots[2];
            uint32_t probe_count  = 0;
            uint32_t max_probes   = capacity_per_table - 2; 
            while (probe_count < max_probes) {
                
                if (atomicCAS(&keys[current_slot], expected, key) == expected) {
                    values[current_slot] = value;
                    atomicAdd(&size[table_id], 1);
                    return true;
                }
                
                if (keys[current_slot] == key) {
                    atomicExch(&values[current_slot], value);
                    return true;
                }
                
                current_slot = base_offset + ((current_slot - base_offset + 1) % capacity_per_table);
                probe_count++;
            }
            return false; 
        }

        
        
        __device__ __forceinline__ bool try_add_at_slot(uint32_t slot, const uint32_t& key, uint table_id) {
            while (true) {
                uint32_t current_key = keys[slot];
                
                if (current_key == key) {
                    atomicAdd(&values[slot], 1);
                    return true;
                }
                
                if (current_key == emptyKey) {
                    if (atomicCAS(&keys[slot], emptyKey, key) == emptyKey) {
                        atomicAdd(&values[slot], 1);
                        atomicAdd(&size[table_id], 1);
                        return true;
                    }
                    continue;
                }
                
                return false;
            }
        }

        __device__ __forceinline__ bool add(const uint32_t& key, uint state_id) {
            if (key == emptyKey) {
                return false;
            }

            int table_id = get_or_assign_table_id(state_id);
            if (table_id == -1) {
                return false;
            }

            uint32_t base_offset = table_id * capacity_per_table;

            
            uint32_t candidate_slots[3];
            candidate_slots[0] = base_offset + (HashTable::hash_func3a(key) % capacity_per_table);
            candidate_slots[1] = base_offset + (HashTable::hash_func3b(key) % capacity_per_table);

            auto start_slot    = (HashTable::hash_func3c(key) == 0 ? HashTable::hash_func3a(key) % capacity_per_table
                                                                   : HashTable::hash_func3b(key) % capacity_per_table);
            candidate_slots[2] = base_offset + ((start_slot + 1) % capacity_per_table);
            
            for (int i = 0; i < 2; i++) {
                if (try_add_at_slot(candidate_slots[i], key, table_id)) {
                    return true;
                }
            }
            
            uint32_t current_slot = candidate_slots[2];
            uint32_t probe_count  = 0;
            uint32_t max_probes   = capacity_per_table - 2; 

            while (probe_count < max_probes) {
                if (try_add_at_slot(current_slot, key, table_id)) {
                    return true;
                }
                
                current_slot = base_offset + ((current_slot - base_offset + 1) % capacity_per_table);
                probe_count++;
            }
            return false; 
        }

        
        __device__ __forceinline__ bool add_count(const uint32_t& key, int32_t count, uint state_id) {
            if (key == emptyKey || key == HashTable::deletedKey) {
                return false;
            }

            int table_id = get_or_assign_table_id(state_id);
            if (table_id == -1) {
                return false;
            }

            uint32_t base_offset = table_id * capacity_per_table;

            
            uint32_t candidate_slots[3];
            candidate_slots[0] = base_offset + (HashTable::hash_func3a(key) % capacity_per_table);
            candidate_slots[1] = base_offset + (HashTable::hash_func3b(key) % capacity_per_table);

            auto start_slot    = (HashTable::hash_func3c(key) == 0 ? HashTable::hash_func3a(key) % capacity_per_table
                                                                   : HashTable::hash_func3b(key) % capacity_per_table);
            candidate_slots[2] = base_offset + ((start_slot + 1) % capacity_per_table);

            
            for (int i = 0; i < 2; i++) {
                if (try_add_count_at_slot(candidate_slots[i], key, count, table_id)) {
                    return true;
                }
            }
            
            uint32_t current_slot = candidate_slots[2];
            uint32_t probe_count  = 0;
            uint32_t max_probes   = capacity_per_table - 2;

            while (probe_count < max_probes) {
                if (try_add_count_at_slot(current_slot, key, count, table_id)) {
                    return true;
                }
                current_slot = base_offset + ((current_slot - base_offset + 1) % capacity_per_table);
                probe_count++;
            }
            return false;
        }

        
        __device__ __forceinline__ bool try_add_count_at_slot(uint32_t slot, const uint32_t& key, int32_t count, uint table_id) {
            while (true) {
                uint32_t current_key = keys[slot];
                
                if (current_key == key) {
                    atomicAdd(&values[slot], count);
                    return true;
                }
                
                if (current_key == emptyKey) {
                    if (atomicCAS(&keys[slot], emptyKey, key) == emptyKey) {
                        atomicAdd(&values[slot], count);
                        atomicAdd(&size[table_id], 1);
                        return true;
                    }
                    continue;
                }
                
                return false;
            }
        }

        
        __device__ __forceinline__ uint get_table_size(uint state_id) {
            int table_id = get_table_id(state_id);
            if (table_id == -1) {
                return 0;
            }
            return size[table_id];
        }

        
        __device__ __forceinline__ bool is_table_empty(uint state_id) {
            return get_table_size(state_id) == 0;
        }

        
        __device__ __forceinline__ bool is_table_full(uint state_id) {
            return get_table_size(state_id) >= capacity_per_table;
        }

        
        __device__ __forceinline__ float get_load_factor(uint state_id) {
            int table_id = get_table_id(state_id);
            if (table_id == -1) {
                return 0.0f;
            }
            return static_cast<float>(size[table_id]) / static_cast<float>(capacity_per_table);
        }
    };
} 

namespace Utils::cpu_hashTable {
    uint32_t hash_func3a(uint32_t x) {
        x = (x ^ (x >> 16)) * 0x85ebca6b;
        x = (x ^ (x >> 13)) * 0xc2b2ae35;
        x = x ^ (x >> 16);
        return x;
    }

    uint32_t hash_func3b(uint32_t x) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 13) ^ x) * 0x9E3779B1;
        x = (x >> 16) ^ x;
        return x;
    }

    uint32_t hash_func3c(uint32_t x) {
        x = (x * 0xabcdef) % 2;
        return x;
    }

    class cpu_hash_table {
    private:
        uint32_t capacity;
        uint32_t* keys;
        int32_t* values;
        uint32_t* size;

        int32_t _emptyValue = 0;
        uint32_t _emptyKey  = -1;
        uint32_t* gpu_keys  = nullptr;
        int32_t* gpu_values = nullptr;
        uint32_t* gpu_size  = nullptr;

    public:
        cpu_hash_table(uint32_t dataLength, float loadFactor, uint32_t emptyKey, int32_t emptyValue)
            : capacity(static_cast<uint>(dataLength / loadFactor)), _emptyKey(emptyKey), _emptyValue(emptyValue) {
            keys   = new uint32_t[capacity];
            values = new int32_t[capacity];
            std::ranges::fill(keys, keys + capacity, _emptyKey);
            std::ranges::fill(values, values + capacity, _emptyValue);
            size    = new uint32_t[1];
            size[0] = 0;
        }
        cpu_hash_table(uint32_t dataLength, float loadFactor) {
            capacity = static_cast<uint>(dataLength / loadFactor);
            keys     = new uint32_t[capacity];
            values   = new int32_t[capacity];
            std::ranges::fill(keys, keys + capacity, _emptyKey);
            std::ranges::fill(values, values + capacity, _emptyValue);
            size    = new uint32_t[1];
            size[0] = 0;
        }

        Utils::HashTable::cudaHashTable3_gpu_viewer<int32_t> get_viewer() {
            if (gpu_keys == nullptr) {
                CUDA_ERROR_CHECK(cudaMalloc((void**) &gpu_keys, sizeof(uint32_t) * capacity));
                CUDA_ERROR_CHECK(cudaMalloc((void**) &gpu_values, sizeof(uint32_t) * capacity));
                CUDA_ERROR_CHECK(cudaMalloc((void**) &gpu_size, sizeof(uint32_t)));
                CUDA_ERROR_CHECK(cudaMemcpy(gpu_keys, keys, sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
                CUDA_ERROR_CHECK(cudaMemcpy(gpu_values, values, sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
                CUDA_ERROR_CHECK(cudaMemcpy(gpu_size, size, sizeof(uint32_t), cudaMemcpyHostToDevice));
            }
            return {gpu_keys, gpu_values, gpu_size, capacity, _emptyKey, _emptyValue};
        }

        uint32_t cpu_get_idx(uint key) const{
            auto slot1 = hash_func3a(key) % capacity;
            auto cur_k = keys[slot1];
            if (cur_k == key) {
                return slot1;
            }
            if (cur_k == _emptyKey) {
                return capacity;
            }
            auto slot2 = hash_func3b(key) % capacity;
            cur_k      = keys[slot2];
            if (cur_k == key) {
                return slot2;
            }
            if (cur_k == _emptyKey) {
                return capacity;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                cur_k = keys[current_slot];
                if (cur_k == key) {
                    return current_slot;
                }
                if (cur_k == _emptyKey) {
                    return -1;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }

        uint32_t cpu_get_value(const uint32_t& key) {
            const uint32_t i = cpu_get_idx(key);
            if (i == capacity) {
                return _emptyValue;
            }
            return values[i];
        }

        uint32_t cpu_get_insert_idx(uint key) {
            auto slot1 = hash_func3a(key) % capacity;
            auto cur_k = keys[slot1];
            if (cur_k == _emptyKey) {
                return slot1;
            }
            auto slot2 = hash_func3b(key) % capacity;
            cur_k      = keys[slot2];
            if (cur_k == _emptyKey) {
                return slot2;
            }
            auto current_slot = ((hash_func3c(key) == 0 ? slot1 : slot2) + 1) % capacity;
            while (true) {
                cur_k = keys[current_slot];
                if (cur_k == _emptyKey) {
                    return current_slot;
                }
                current_slot = (current_slot + 1) % capacity;
            }
        }

        bool cpu_insert(uint key, uint value) {
            auto slot1 = hash_func3a(key) % capacity;
            auto cur_k = keys[slot1];
            if (cur_k == key || cur_k == _emptyKey) {
                keys[slot1]   = key;
                values[slot1] = value;
                return true;
            }
            auto slot2 = hash_func3b(key) % capacity;
            cur_k      = keys[slot2];
            if (cur_k == key || cur_k == _emptyKey) {
                keys[slot2]   = key;
                values[slot2] = value;
                return true;
            }
            auto pivot_slot   = hash_func3c(key) == 0 ? slot1 : slot2;
            auto current_slot = (pivot_slot + 1) % capacity;
            while (current_slot != pivot_slot) {
                cur_k = keys[current_slot];
                if (cur_k == key || cur_k == _emptyKey) {
                    keys[current_slot]   = key;
                    values[current_slot] = value;
                    return true;
                }
                current_slot = (current_slot + 1) % capacity;
            }
            return false;
        }

        ~cpu_hash_table() {
            if (gpu_keys) {
                CUDA_ERROR_CHECK(cudaFree(gpu_keys));
            }
            if (gpu_values) {
                CUDA_ERROR_CHECK(cudaFree(gpu_values));
            }
            if (gpu_size) {
                CUDA_ERROR_CHECK(cudaFree(gpu_size));
            }
            if (keys) {
                free(keys);
            }
            if (values) {
                free(values);
            }
            if (size) {
                free(size);
            }
            keys   = nullptr;
            values = nullptr;
            size   = nullptr;
        }
    };
} 

typedef Utils::HashTable::cudaArrayMap_gpu_viewer<int32_t> arrMapTable;
typedef Utils::HashTable::cudaHashTable3_gpu_viewer<int32_t> HashTable_viewer;

#endif 
