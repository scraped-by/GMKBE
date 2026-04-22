



#ifndef _PARTITION_CUH_
#define _PARTITION_CUH_

#include <numeric>
#include <thrust/system/cuda/detail/util.h>
#include <utils.h>

__constant__ inline uint degrees_bar[] = {32, 1024, 32768, 1048576, 33554432, 1073741824, MAX_UINT};



struct Partition_Device_viewer {
    uint* partition = nullptr;
    uint* size      = nullptr; 
    uint* idxs      = nullptr;

    Partition_Device_viewer(uint* partition, uint* size, uint* idxs) : partition(partition), size(size), idxs(idxs) {}

    static __device__ __forceinline__ uint get_partition_idx_from_degree(const uint node_degree) {
        
        
        
        
        

        if (node_degree < 32) {
            return 0;
        } else if (node_degree < 1024) {
            return 1;
        } else if (node_degree < 32768) {
            return 2;
        } else {
            return 3;
        }
    }
    __device__ __forceinline__ uint get_partition_idx_from_size(const uint idx) {
        uint prefix_sum = 0;
        prefix_sum += size[0];
        if (idx < prefix_sum) {
            return 0;
        }
        prefix_sum += size[1];
        if (idx < prefix_sum) {
            return 1;
        }
        prefix_sum += size[2];
        if (idx < prefix_sum) {
            return 2;
        }
        prefix_sum += size[3];
        if (idx < prefix_sum) {
            return 3;
        }
    }

    __device__ __forceinline__ uint get_global_offset(const uint partition_idx) const {
        uint global_offset = 0;
        for (uint i = 0; i < partition_idx; ++i) {
            global_offset += size[i];
        }
        return global_offset;
    }

    __device__ __forceinline__ void insert_node_to_partition_from_degree(
        const uint node, const uint node_degree) const {
        
        const uint partition_idx = get_partition_idx_from_degree(node_degree);
        insert_node_to_partition_from_partitionIdx(node, partition_idx);
    }

    __device__ __forceinline__ void insert_node_to_partition_from_partitionIdx(
        const uint node, const uint partition_idx) const {
        
        const uint insert_pos = atomicAdd(&idxs[partition_idx], 1);
        
        const uint global_offset = get_global_offset(partition_idx);
        
        partition[global_offset + insert_pos] = node;
    }


    






    static __device__ __forceinline__ uint get_partition_idx_from_global_index(
        const uint* size, const uint global_index) {
        
        if (global_index < size[0]) {
            return 0;
        }
        
        uint cumulative_size = size[0];
        if (global_index < cumulative_size + size[1]) {
            return 1;
        }
        
        cumulative_size += size[1];
        if (global_index < cumulative_size + size[2]) {
            return 2;
        }
        
        
        return 3;
    }
};


struct CudaDeviceDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            CUDA_ERROR_CHECK(cudaFree(ptr));
        }
    }
};


struct CudaHostDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            CUDA_ERROR_CHECK(cudaFreeHost(ptr));
        }
    }
};


template <typename T>
using CudaDevicePtr = std::unique_ptr<T, CudaDeviceDeleter>;

template <typename T>
using CudaHostPtr = std::unique_ptr<T, CudaHostDeleter>;

struct Partition {
    
    uint* partition              = nullptr; 
    uint* size                   = nullptr; 
    uint* idxs                   = nullptr; 
    uint* host_size              = nullptr; 
    uint* partition_original_ptr = nullptr; 
    uint totalSize               = 0; 
    uint partition_original_malloc_num = 0; 

    Partition() {
        CUDA_ERROR_CHECK(cudaMallocHost((void**) &host_size, 4 * sizeof(uint)));
        for (uint i = 0; i < 4; ++i) {
            host_size[i] = 0;
        }
        CUDA_ERROR_CHECK(cudaMalloc((void**) &size, 4 * sizeof(uint)));
        CUDA_ERROR_CHECK(cudaMemset(size, 0, 4 * sizeof(uint)));
        CUDA_ERROR_CHECK(cudaMalloc((void**) &idxs, 4 * sizeof(uint)));
        CUDA_ERROR_CHECK(cudaMemset(idxs, 0, 4 * sizeof(uint)));
    }

    Partition(cudaStream_t stream1, cudaStream_t stream2) {
        CUDA_ERROR_CHECK(cudaMallocAsync((void**) &size, 4 * sizeof(uint), stream1));
        CUDA_ERROR_CHECK(cudaMemsetAsync(size, 0, 4 * sizeof(uint), stream1));
        CUDA_ERROR_CHECK(cudaMallocAsync((void**) &idxs, 4 * sizeof(uint), stream2));
        CUDA_ERROR_CHECK(cudaMemsetAsync(idxs, 0, 4 * sizeof(uint), stream2));
    }
    __host__ ptrdiff_t get_bias() const {
        return partition - partition_original_ptr;
    }
    
    __host__ void malloc_partition(cudaStream_t& stream) {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(host_size, size, 4 * sizeof(uint), cudaMemcpyDefault, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        const uint length = accumulate(host_size, host_size + 4, uint());
        CUDA_ERROR_CHECK(cudaMalloc((void**) &partition_original_ptr, length * sizeof(uint)));
        partition_original_malloc_num = length;
        partition                     = partition_original_ptr;
        totalSize                     = partition_original_malloc_num;
    }

    __host__ void remalloc_partition(cudaStream_t& stream) {
        uint* new_partition_sizes = new uint[4];
        CUDA_ERROR_CHECK(cudaMemcpyAsync(new_partition_sizes, size, 4 * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        uint new_total_malloc_num = accumulate(new_partition_sizes, new_partition_sizes + 4, uint());
        uint* new_partition;
        CUDA_ERROR_CHECK(cudaMallocAsync(&new_partition, new_total_malloc_num * sizeof(uint), stream));

        assert(new_total_malloc_num > partition_original_malloc_num);

        
        uint old_offset = 0;
        uint new_offset = 0;
        
        for (int i = 0; i < 4; ++i) {
            uint elements_to_copy = host_size[i]; 
            if (elements_to_copy > 0) {
                
                uint* src_ptr = this->partition_original_ptr + old_offset;
                
                uint* dst_ptr = new_partition + new_offset;
                
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
            }
            
            old_offset += host_size[i];
            new_offset += new_partition_sizes[i];
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        CUDA_ERROR_CHECK(cudaFree(partition_original_ptr));

        ptrdiff_t offset_from_original      = this->partition - this->partition_original_ptr;
        this->partition_original_ptr        = new_partition;
        this->partition                     = this->partition_original_ptr + offset_from_original;
        this->partition_original_malloc_num = new_total_malloc_num;
        memcpy(host_size, new_partition_sizes, sizeof(uint) * 4);
    }

    __host__ void from_vector(
        const std::vector<uint>& vec, const std::vector<uint>& partition_size, cudaStream_t stream = nullptr) {
        assert(partition == nullptr);
        CUDA_ERROR_CHECK(cudaMallocAsync((void**) &partition, vec.size() * sizeof(uint), stream));
        partition_original_ptr = partition;
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(partition, vec.data(), vec.size() * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(size, partition_size.data(), 4 * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(idxs, partition_size.data(), 4 * sizeof(uint), cudaMemcpyHostToDevice, stream));
        memcpy(host_size, partition_size.data(), 4 * sizeof(uint));
        totalSize                     = std::accumulate(partition_size.begin(), partition_size.end(), 0);
        partition_original_malloc_num = totalSize;
    }

    __host__ void advance_and_update(cudaStream_t stream = nullptr) {
        assert(totalSize != 0);

        int target_partition_idx = -1;
        for (int i = 0; i < 4; ++i) {
            if (host_size[i] > 0) {
                target_partition_idx = i;
                break;
            }
        }
        if (target_partition_idx == -1) {
            fprintf(stderr, "Error: Inconsistent state. totalSize > 0 but all host_size partitions are empty.\n");
            assert(target_partition_idx == -1);
        }
        host_size[target_partition_idx]--;
        totalSize--; 
        uint* device_size_ptr = size + target_partition_idx;
        uint* device_idxs_ptr = idxs + target_partition_idx;
        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            device_size_ptr, &host_size[target_partition_idx], sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            device_idxs_ptr, &host_size[target_partition_idx], sizeof(uint), cudaMemcpyHostToDevice, stream));
        partition++;
    }
    Partition_Device_viewer get_gpu_viewer() {
        return {partition, size, idxs};
    }

    __host__ uint get_partition_idx_from_global_idx(const uint global_index) const {
        
        if (global_index < host_size[0]) {
            return 0;
        }
        
        uint cumulative_size = host_size[0];
        if (global_index < cumulative_size + host_size[1]) {
            return 1;
        }
        
        cumulative_size += host_size[1];
        if (global_index < cumulative_size + host_size[2]) {
            return 2;
        }
        
        
        return 3;
    }

    Partition& operator=(const Partition& other) {
        
        if (this == &other || other.partition == nullptr) {
            return *this;
        }

        if (this->partition_original_malloc_num != other.partition_original_malloc_num) {
            
            cleanup();
            
            totalSize                     = other.totalSize;
            partition_original_malloc_num = other.partition_original_malloc_num;
            
            if (other.host_size != nullptr) {
                memcpy(host_size, other.host_size, 4 * sizeof(uint));
            }
            
            if (other.size != nullptr) {
                CUDA_ERROR_CHECK(cudaMemcpy(size, other.size, 4 * sizeof(uint), cudaMemcpyDeviceToDevice));
            }
            
            if (other.idxs != nullptr) {
                CUDA_ERROR_CHECK(cudaMemcpy(idxs, other.idxs, 4 * sizeof(uint), cudaMemcpyDeviceToDevice));
            }
            
            if (other.partition != nullptr) {
                CUDA_ERROR_CHECK(
                    cudaMalloc((void**) &partition_original_ptr, other.partition_original_malloc_num * sizeof(uint)));
                CUDA_ERROR_CHECK(cudaMemcpy(partition_original_ptr, other.partition_original_ptr,
                    other.partition_original_malloc_num * sizeof(uint), cudaMemcpyDeviceToDevice));
                ptrdiff_t offset = other.partition - other.partition_original_ptr;
                partition        = partition_original_ptr + offset;
            }
        } else {
            totalSize                     = other.totalSize;
            partition_original_malloc_num = other.partition_original_malloc_num;
            memcpy(host_size, other.host_size, 4 * sizeof(uint));
            CUDA_ERROR_CHECK(cudaMemcpy(partition_original_ptr, other.partition_original_ptr,
                other.partition_original_malloc_num * sizeof(uint), cudaMemcpyDeviceToDevice));
            if (other.partition == other.partition_original_ptr) {
                partition = partition_original_ptr;
            } else {
                partition = partition_original_ptr + (other.partition - other.partition_original_ptr);
            }
            CUDA_ERROR_CHECK(cudaMemcpy(size, other.size, 4 * sizeof(uint), cudaMemcpyDeviceToDevice));
            CUDA_ERROR_CHECK(cudaMemcpy(idxs, other.idxs, 4 * sizeof(uint), cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    __host__ void CopyPartitionFromX(const Partition& other, uint partition_idx, cudaStream_t stream = nullptr) {
        
        assert(this->partition_original_ptr != nullptr && "Destination 'this' partition buffer is not allocated.");
        assert(other.partition_original_ptr != nullptr && "Source 'other' partition buffer is not allocated.");
        assert(this->partition_original_malloc_num >= other.partition_original_malloc_num
               && "Destination 'this' must be larger than or equal to source 'other'.");
        memcpy(this->host_size, other.host_size, 4 * sizeof(uint));
        this->host_size[partition_idx] += 1;
        
        uint offset_this  = 0; 
        uint offset_other = 0; 

        for (int i = 0; i < 4; ++i) {
            
            
            uint elements_to_copy = other.host_size[i];

            if (elements_to_copy > 0) {
                
                assert(this->host_size[i] >= elements_to_copy && "Destination partition is too small for source data.");

                
                uint* dst_ptr = this->partition_original_ptr + offset_this;
                uint* src_ptr = other.partition_original_ptr + offset_other;

                
                CUDA_ERROR_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, elements_to_copy * sizeof(uint),
                    cudaMemcpyDeviceToDevice, 
                    stream));
            }

            
            offset_this += this->host_size[i];
            offset_other += other.host_size[i];
        }

        

        
        this->totalSize = other.totalSize;

        
        
        ptrdiff_t view_offset = other.partition - other.partition_original_ptr;
        this->partition       = this->partition_original_ptr + view_offset;

        
        
        CUDA_ERROR_CHECK(cudaMemcpyAsync(this->idxs, other.idxs, 4 * sizeof(uint),
            cudaMemcpyDeviceToDevice, 
            stream));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(this->size, this->host_size, 4 * sizeof(uint),
            cudaMemcpyHostToDevice, 
            stream));
    }

    
    Partition& operator=(Partition&& other) noexcept {
        
        if (this == &other) {
            return *this;
        }
        
        
        if (host_size != nullptr) {
            cudaFreeHost(host_size);
        }
        if (partition_original_ptr != nullptr) {
            cudaFree(partition_original_ptr);
        }
        if (idxs != nullptr) {
            cudaFree(idxs);
        }
        if (size != nullptr) {
            cudaFree(size);
        }
        
        
        partition                     = other.partition;
        size                          = other.size;
        idxs                          = other.idxs;
        host_size                     = other.host_size;
        partition_original_ptr        = other.partition_original_ptr;
        totalSize                     = other.totalSize;
        partition_original_malloc_num = other.partition_original_malloc_num;
        
        
        other.partition                     = nullptr;
        other.size                          = nullptr;
        other.idxs                          = nullptr;
        other.host_size                     = nullptr;
        other.partition_original_ptr        = nullptr;
        other.totalSize                     = 0;
        other.partition_original_malloc_num = 0;
        
        return *this;
    }

    bool empty() const {
        return totalSize == 0;
    }
    bool is_valid() const {
        return partition_original_malloc_num == 0;
    }
    ~Partition() {
        if (host_size != nullptr) {
            CUDA_ERROR_CHECK(cudaFreeHost(host_size));
        }
        if (partition_original_ptr != nullptr) {
            CUDA_ERROR_CHECK(cudaFree(partition_original_ptr));
            partition_original_ptr = nullptr;
            partition              = nullptr;
        }
        if (idxs != nullptr) {
            CUDA_ERROR_CHECK(cudaFree(idxs));
        }
        if (size != nullptr) {
            CUDA_ERROR_CHECK(cudaFree(size));
        }
    }

private:
    
    void cleanup() {
        if (partition_original_ptr != nullptr) {
            cudaFree(partition_original_ptr);
            partition              = nullptr;
            partition_original_ptr = nullptr;
        }
    }
};

class Partition2 {
private:
    inline void initialize() {
        uint* p_size      = nullptr;
        uint* p_idxs      = nullptr;
        uint* p_host_size = nullptr;

        
        CUDA_ERROR_CHECK(cudaMalloc((void**) &p_size, 4 * sizeof(uint)));
        size_.reset(p_size);
        CUDA_ERROR_CHECK(cudaMemset(size_.get(), 0, 4 * sizeof(uint)));

        CUDA_ERROR_CHECK(cudaMalloc((void**) &p_idxs, 4 * sizeof(uint)));
        idxs_.reset(p_idxs);
        CUDA_ERROR_CHECK(cudaMemset(idxs_.get(), 0, 4 * sizeof(uint)));

        CUDA_ERROR_CHECK(cudaMallocHost((void**) &p_host_size, 4 * sizeof(uint)));
        host_size_.reset(p_host_size);
        std::fill(host_size_.get(), host_size_.get() + 4, 0);
    }
    [[deprecated("This function is inefficient and will be removed in a future version.")]]
    void malloc_partition_(cudaStream_t stream) {
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(host_size_.get(), size_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        const uint length = std::accumulate(host_size_.get(), host_size_.get() + 4, 0u);
        if (length == 0) {
            partition_original_ptr_.reset(nullptr);
            partition_original_malloc_num = 0;
            partition                     = nullptr;
            totalSize                     = 0;
            return;
        }
        uint* p_partition = nullptr;
        CUDA_ERROR_CHECK(cudaMalloc((void**) &p_partition, length * sizeof(uint)));
        partition_original_ptr_.reset(p_partition); 

        partition_original_malloc_num = length;
        partition                     = partition_original_ptr_.get();
        totalSize                     = length;
    }

    [[deprecated("This function is inefficient and will be removed in a future version.")]]
    __host__ void try_to_remalloc_partition_shrink(cudaStream_t stream) {
        uint new_partition_sizes[4];
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(new_partition_sizes, size_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        uint new_total_malloc_num = std::accumulate(new_partition_sizes, new_partition_sizes + 4, 0u);

        
        if (new_total_malloc_num == 0) {
            clean(stream);
            return;
        }

        auto migrate_data_logic = [&](uint* dst_base_ptr) {
            uint old_offset = 0;
            uint new_offset = 0;
            for (int i = 0; i < 4; ++i) {
                uint elements_to_copy = host_size_.get()[i];
                if (elements_to_copy > 0) {
                    uint* src_ptr = partition_original_ptr_.get() + old_offset;
                    uint* dst_ptr = dst_base_ptr + new_offset;
                    if (src_ptr != dst_ptr) {
                        CUDA_ERROR_CHECK(cudaMemcpyAsync(
                            dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
                    }
                }
                old_offset += host_size_.get()[i];
                new_offset += new_partition_sizes[i];
            }
        };

        
        const bool must_realloc_due_to_growth   = new_total_malloc_num > partition_original_malloc_num;
        const bool should_realloc_due_to_shrink = (partition_original_malloc_num > 0) && (new_total_malloc_num > 0)
                                               && (partition_original_malloc_num > new_total_malloc_num * 4);

        if (must_realloc_due_to_growth || should_realloc_due_to_shrink) {
            
            uint* new_partition_data = nullptr;
            CUDA_ERROR_CHECK(cudaMallocAsync(&new_partition_data, new_total_malloc_num * sizeof(uint), stream));

            
            CudaDevicePtr<uint> new_partition_ptr(new_partition_data);

            migrate_data_logic(new_partition_data);

            
            ptrdiff_t offset_from_original = partition != nullptr ? partition - partition_original_ptr_.get() : 0;

            
            partition_original_ptr_.swap(new_partition_ptr);

            partition_original_malloc_num = new_total_malloc_num;
            partition                     = partition_original_ptr_.get() + offset_from_original;

        } else {
            
            partition = partition_original_ptr_.get(); 
            migrate_data_logic(partition_original_ptr_.get());
        }

        
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        
        totalSize = new_total_malloc_num;
        std::copy(new_partition_sizes, new_partition_sizes + 4, host_size_.get());
    }

public:
    
    
    CudaDevicePtr<uint> partition_original_ptr_ = nullptr;
    CudaDevicePtr<uint> size_                   = nullptr;
    CudaDevicePtr<uint> idxs_                   = nullptr;
    CudaHostPtr<uint> host_size_                = nullptr;

    
    uint* partition                    = nullptr; 
    uint totalSize                     = 0;
    uint partition_original_malloc_num = 0;

    

    
    Partition2() {
        initialize();
    }
    __host__ Partition2(const uint size) { 
        initialize();
        uint* d_ptr       = nullptr;
        const uint length = size + 1; 
        cudaError_t err   = cudaMalloc((void**) &d_ptr, length * sizeof(uint));
        if (err != cudaSuccess) {
            
            throw std::runtime_error("Failed to allocate CUDA memory in Partition.");
        }

        
        partition_original_ptr_.reset(d_ptr);
        partition                     = d_ptr; 
        partition_original_malloc_num = length;
    }
    __host__ ptrdiff_t get_bias() const {
        return partition - partition_original_ptr_.get();
    }
    __host__ void CopyPartitionFromX(const Partition2& other, uint partition_idx, cudaStream_t stream = nullptr) {
        

        
        assert(this->partition_original_ptr_ && "Destination 'this' partition buffer is not allocated.");
        assert(other.partition_original_ptr_ && "Source 'other' partition buffer is not allocated.");

        
        
        assert(this->partition_original_malloc_num >= other.totalSize + 1
               && "Destination 'this' buffer must be large enough to hold 'other' + 1 element.");

        

        
        
        std::copy(other.host_size_.get(), other.host_size_.get() + 4, this->host_size_.get());
        uint tmp = accumulate(other.host_size_.get(), other.host_size_.get() + 4, 0);

        
        
        this->host_size_.get()[partition_idx] += 1;


        
        uint offset_this  = 0; 
        uint offset_other = 0; 

        for (int i = 0; i < 4; ++i) {
            
            uint elements_to_copy = other.host_size_.get()[i];

            if (elements_to_copy > 0) {
                
                
                
                uint* dst_ptr       = this->partition_original_ptr_.get() + offset_this;
                const uint* src_ptr = other.partition_original_ptr_.get() + offset_other; 

                
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
            }

            
            
            
            
            offset_this += this->host_size_.get()[i];
            offset_other += other.host_size_.get()[i];
        }

        

        
        
        this->totalSize = other.totalSize + 1;

        
        
        
        
        this->partition = this->partition_original_ptr_.get() + other.get_bias();

        

        
        
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(this->idxs_.get(), other.idxs_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToDevice, stream));

        
        
        
        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            this->size_.get(), this->host_size_.get(), 4 * sizeof(uint), cudaMemcpyHostToDevice, stream));
    }


    
    ~Partition2() = default;

    

    
    Partition2(Partition2&& other) noexcept = default;

    
    
    





    Partition2& operator=(Partition2&& other) noexcept {
        
        if (this != &other) {
            
            
            partition_original_ptr_       = std::move(other.partition_original_ptr_);
            size_                         = std::move(other.size_);
            idxs_                         = std::move(other.idxs_);
            host_size_                    = std::move(other.host_size_);
            partition                     = other.partition;
            totalSize                     = other.totalSize;
            partition_original_malloc_num = other.partition_original_malloc_num;

            
            other.partition                     = nullptr;
            other.totalSize                     = 0;
            other.partition_original_malloc_num = 0;
        }
        return *this;
    }

    
    

    
    
    Partition2(const Partition2& other) = delete;

    
    Partition2& operator=(const Partition2& other) {
        if (this == &other) {
            return *this;
        }

        
        Partition2 temp;

        
        temp.totalSize                     = other.totalSize;
        temp.partition_original_malloc_num = other.partition_original_malloc_num;

        
        if (other.host_size_) {
            std::copy(other.host_size_.get(), other.host_size_.get() + 4, temp.host_size_.get());
        }

        
        if (other.size_) {
            CUDA_ERROR_CHECK(
                cudaMemcpy(temp.size_.get(), other.size_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToDevice));
        }

        
        if (other.idxs_) {
            CUDA_ERROR_CHECK(
                cudaMemcpy(temp.idxs_.get(), other.idxs_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToDevice));
        }

        
        if (other.partition_original_ptr_) {
            uint* new_partition_data = nullptr;
            CUDA_ERROR_CHECK(
                cudaMalloc((void**) &new_partition_data, other.partition_original_malloc_num * sizeof(uint)));
            temp.partition_original_ptr_.reset(new_partition_data);

            CUDA_ERROR_CHECK(cudaMemcpy(temp.partition_original_ptr_.get(), other.partition_original_ptr_.get(),
                other.partition_original_malloc_num * sizeof(uint), cudaMemcpyDeviceToDevice));

            
            ptrdiff_t offset = other.partition - other.partition_original_ptr_.get();
            temp.partition   = temp.partition_original_ptr_.get() + offset;
        }

        
        swap(*this, temp);

        return *this;
    }

    
    friend void swap(Partition2& first, Partition2& second) noexcept {
        using std::swap;
        swap(first.partition_original_ptr_, second.partition_original_ptr_);
        swap(first.size_, second.size_);
        swap(first.idxs_, second.idxs_);
        swap(first.host_size_, second.host_size_);
        swap(first.partition, second.partition);
        swap(first.totalSize, second.totalSize);
        swap(first.partition_original_malloc_num, second.partition_original_malloc_num);
    }


    
    __host__ void malloc_partition(cudaStream_t stream) {
        
        uint new_host_sizes[4];
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(new_host_sizes, size_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToHost, stream));
        
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        
        const uint length = std::accumulate(new_host_sizes, new_host_sizes + 4, 0u);

        
        if (length == 0) {
            
            partition_original_ptr_.reset(nullptr);
            partition_original_malloc_num = 0;
            partition                     = nullptr;
            totalSize                     = 0;
            
            std::fill(host_size_.get(), host_size_.get() + 4, 0);
            return;
        }

        
        const bool must_realloc_due_to_growth = length > partition_original_malloc_num;
        
        const bool should_realloc_due_to_shrink =
            (partition_original_malloc_num > 0) && (length < static_cast<uint>(partition_original_malloc_num * 0.25));

        if (must_realloc_due_to_growth || should_realloc_due_to_shrink) {
            
            
            uint* p_partition = nullptr;
            CUDA_ERROR_CHECK(cudaMalloc((void**) &p_partition, length * sizeof(uint)));

            
            partition_original_ptr_.reset(p_partition);
            partition_original_malloc_num = length;
        } else {
            
            
        }

        
        partition = partition_original_ptr_.get(); 
        totalSize = length;
        
        std::copy(new_host_sizes, new_host_sizes + 4, host_size_.get());
    }

    __host__ void try_to_remalloc_partition(cudaStream_t stream) {


        
        uint new_partition_sizes[4];
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(new_partition_sizes, size_.get(), 4 * sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        uint new_total_malloc_num = std::accumulate(new_partition_sizes, new_partition_sizes + 4, 0u);
        if (new_partition_sizes == 0) {
            clean(stream);
        }
        auto migrate_data_logic = [&](uint* dst_base_ptr) {
            uint old_offset = 0;
            uint new_offset = 0;
            
            for (int i = 0; i < 4; ++i) {
                uint elements_to_copy = host_size_.get()[i]; 
                if (elements_to_copy > 0) {
                    
                    uint* src_ptr = partition_original_ptr_.get() + old_offset;
                    
                    uint* dst_ptr = dst_base_ptr + new_offset;

                    
                    if (src_ptr != dst_ptr) {
                        CUDA_ERROR_CHECK(cudaMemcpyAsync(
                            dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
                    }
                }
                old_offset += host_size_.get()[i]; 
                new_offset += new_partition_sizes[i]; 
            }
        };

        if (new_total_malloc_num > partition_original_malloc_num) {
            uint* new_partition_data = nullptr;
            CUDA_ERROR_CHECK(cudaMallocAsync(&new_partition_data, new_total_malloc_num * sizeof(uint), stream));
            CudaDevicePtr<uint> new_partition_ptr(new_partition_data); 
            migrate_data_logic(new_partition_data);
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            
            ptrdiff_t offset_from_original = partition != nullptr ? partition - partition_original_ptr_.get() : 0;
            
            partition_original_ptr_.swap(new_partition_ptr);
            
            partition_original_malloc_num = new_total_malloc_num;
            partition                     = partition_original_ptr_.get() + offset_from_original;
        } else {
            partition = partition_original_ptr_.get();
            migrate_data_logic(partition_original_ptr_.get());
        }
        totalSize = new_total_malloc_num;
        std::copy(new_partition_sizes, new_partition_sizes + 4, host_size_.get());
    }


    __host__ uint get_partition_idx_from_global_idx(const uint global_index) const {
        
        const uint* host_size_ptr = this->host_size();
        if (global_index < host_size_ptr[0]) {
            return 0;
        }
        
        uint cumulative_size = host_size_ptr[0];
        if (global_index < cumulative_size + host_size_ptr[1]) {
            return 1;
        }
        
        cumulative_size += host_size_ptr[1];
        if (global_index < cumulative_size + host_size_ptr[2]) {
            return 2;
        }
        
        
        return 3;
    }

    __host__ void from_vector(
        const std::vector<uint>& vec, const std::vector<uint>& partition_size, cudaStream_t stream = nullptr) {
        assert(partition_original_ptr_ == nullptr);
        totalSize                     = vec.size();
        partition_original_malloc_num = totalSize;
        if (totalSize == 0) {
            return;
        }

        uint* p_partition = nullptr;
        CUDA_ERROR_CHECK(cudaMallocAsync((void**) &p_partition, totalSize * sizeof(uint), stream));
        partition_original_ptr_.reset(p_partition);
        partition = partition_original_ptr_.get();

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(partition, vec.data(), totalSize * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(size_.get(), partition_size.data(), 4 * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(idxs_.get(), partition_size.data(), 4 * sizeof(uint), cudaMemcpyHostToDevice, stream));

        std::copy(partition_size.begin(), partition_size.end(), host_size_.get());
    }

    __host__ void advance_and_update(cudaStream_t stream = nullptr) {
        if (totalSize == 0) {
            return;
        }

        int target_partition_idx = -1;
        for (int i = 0; i < 4; ++i) {
            if (host_size_.get()[i] > 0) {
                target_partition_idx = i;
                break;
            }
        }
        assert(target_partition_idx != -1 && "Inconsistent state.");

        host_size_.get()[target_partition_idx]--;
        totalSize--;
        partition++;

        uint* device_size_ptr = size_.get() + target_partition_idx;
        uint* device_idxs_ptr = idxs_.get() + target_partition_idx;
        uint new_val          = host_size_.get()[target_partition_idx];
        CUDA_ERROR_CHECK(cudaMemcpyAsync(device_size_ptr, &new_val, sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(device_idxs_ptr, &new_val, sizeof(uint), cudaMemcpyHostToDevice, stream));
    }

    









    void clean(cudaStream_t stream = nullptr) {
        
        CUDA_ERROR_CHECK(cudaMemsetAsync(size_.get(), 0, 4 * sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(idxs_.get(), 0, 4 * sizeof(uint), stream));

        
        std::fill(host_size_.get(), host_size_.get() + 4, 0);

        
        totalSize = 0;
        partition = nullptr;
        
        
    }

    

    bool empty() const {
        return totalSize == 0;
    }

    
    
    bool is_partition_allocated() const {
        return partition_original_ptr_ != nullptr;
    }

    Partition_Device_viewer get_gpu_viewer() {
        return {partition, size_.get(), idxs_.get()};
    }

    __host__ void reset() {
        assert(partition_original_ptr_ == nullptr && partition != nullptr);
        partition_original_ptr_.reset(partition);
        partition_original_malloc_num = totalSize;
    }

    
    uint total_size() const {
        return totalSize;
    }
    const uint* host_size() const {
        return host_size_.get();
    }
};

#define LAUNCH_PARTITION_KERNEL(PARTITION_PTR, KERNEL_NAME, STREAM, ...) \
    do { \
         \
        if (!(PARTITION_PTR) || !(PARTITION_PTR)->host_size_.get()) { \
             \
            std::cerr << "Launch error: PARTITION_PTR or its host_size is null in " << __FILE__ << ":" << __LINE__ << std::endl; \
            break; \
        } \
        \
         \
        const uint* partition_sizes = (PARTITION_PTR)->host_size_.get(); \
        \
        for (uint start = 0; start < 3; start++) { \
            const uint current_partition_size = partition_sizes[start]; \
            \
             \
             \
            if (current_partition_size == 0) { \
                continue; \
            } \
            \
             \
             \
            const uint threads_per_block = std::max(32u, std::min(256u, next_power_of_2(current_partition_size))); \
            \
             \
            const uint group_num = std::max(32u, static_cast<uint>(pow(32, start))); \
            \
             \
            const uint total_threads_needed = group_num * current_partition_size; \
            const uint grid_num = std::min(CALC_GRID_DIM(total_threads_needed, threads_per_block), static_cast<uint>(MAX_GRID_DIM_X)); \
            \
             \
            if (grid_num == 0) { \
                continue; \
            } \
            \
             \
            KERNEL_NAME<<<grid_num, threads_per_block, 0, STREAM>>>(__VA_ARGS__, start, group_num); \
        } \
    } while (0)


inline uint safe_power_of_32(uint n) {
    if (n == 0) return 1;
    if (n >= 7) return UINT_MAX / 2; 
    return (uint)pow(32, n);
}

















































#ifdef ABLATION_FIXED_GROUP_SIZE
    
    
    #define _INTERNAL_CALC_GROUP_SIZE(partition_idx) (ABLATION_FIXED_GROUP_SIZE)
#else
    
    
    #define _INTERNAL_CALC_GROUP_SIZE(partition_idx) std::max(32u, safe_power_of_32(partition_idx))
#endif
#define _VAL_TO_STR_HELPER(x) #x
#define _VAL_TO_STR(x) _VAL_TO_STR_HELPER(x)



#define LAUNCH_PARTITION_KERNEL_V3(PARTITION_PTR, KERNEL_NAME, STREAM, ...) \
    do { \
         \
        if (!(PARTITION_PTR) || !(PARTITION_PTR)->host_size_.get()) { \
            std::cerr << "Launch error: PARTITION_PTR or its host_size is null in " << __FILE__ << ":" << __LINE__ << std::endl; \
            break; \
        } \
        \
         \
        const uint* partition_sizes = (PARTITION_PTR)->host_size_.get(); \
        const uint max_partitions = (PARTITION_PTR)->max_partition_num_; \
        \
        for (uint start = 0; start < max_partitions; start++) { \
            const uint current_partition_size = partition_sizes[start]; \
            \
             \
            if (current_partition_size == 0) { \
                continue; \
            } \
            \
             \
            const uint group_size = _INTERNAL_CALC_GROUP_SIZE(start); \
            \
             \
            uint total_threads_needed; \
            if (__builtin_umul_overflow(group_size, current_partition_size, &total_threads_needed)) { \
                 \
                total_threads_needed = UINT_MAX / 2; \
            } \
            \
            const uint grid_num = std::min(CALC_GRID_DIM(total_threads_needed, THREADS_PER_BLOCK), static_cast<uint>(MAX_GRID_DIM_X)); \
            \
             \
            if (grid_num == 0) { \
                continue; \
            } \
            \
             \
            KERNEL_NAME<<<grid_num, THREADS_PER_BLOCK, 0, STREAM>>>(__VA_ARGS__, start, group_size); \
        } \
    } while (0)

__constant__ uint d_max_partition_num;

struct Partition_Device_viewer_v3 {
    uint* partition = nullptr;
    uint* size      = nullptr;
    uint* idxs      = nullptr;

    Partition_Device_viewer_v3(uint* partition, uint* size, uint* idxs)
        : partition(partition), size(size), idxs(idxs) {}

    static __device__ __forceinline__ uint get_partition_idx_from_degree(const uint node_degree) {
        
        
        if (node_degree == 0) return 0;

        
        uint partition_idx = 0;

        while (partition_idx < d_max_partition_num - 1 && node_degree >= degrees_bar[partition_idx]) {
            partition_idx++;
        }

        return partition_idx;
    }

    __device__ __forceinline__ uint get_partition_idx_from_size(const uint idx) {
        uint prefix_sum = 0;
        for (uint i = 0; i < d_max_partition_num; ++i) {
            prefix_sum += size[i];
            if (idx < prefix_sum) {
                return i;
            }
        }
        return d_max_partition_num - 1;
    }

    __device__ __forceinline__ uint get_global_offset(const uint partition_idx) const {
        uint global_offset = 0;
        for (uint i = 0; i < partition_idx; ++i) {
            global_offset += size[i];
        }
        return global_offset;
    }

    __device__ __forceinline__ void insert_node_to_partition_from_degree(
        const uint node, const uint node_degree) const {
        const uint partition_idx = get_partition_idx_from_degree(node_degree);
        insert_node_to_partition_from_partitionIdx(node, partition_idx);
    }

    __device__ __forceinline__ void insert_node_to_partition_from_partitionIdx(
        const uint node, const uint partition_idx) const {
        const uint insert_pos = atomicAdd(&idxs[partition_idx], 1);
        const uint global_offset = get_global_offset(partition_idx);
        partition[global_offset + insert_pos] = node;
    }

    static __device__ __forceinline__ uint get_partition_idx_from_global_index(
        const uint* size, const uint global_index) {
        uint cumulative_size = 0;
        for (uint i = 0; i < d_max_partition_num; ++i) {
            cumulative_size += size[i];
            if (global_index < cumulative_size) {
                return i;
            }
        }
        return d_max_partition_num - 1;
    }
};

class CandidateQueue_v3 {
public:
    
    uint* partition_shared_ptr = nullptr;

    
    CudaDevicePtr<uint> size_ = nullptr;
    
    CudaHostPtr<uint> host_size_ = nullptr;

    uint max_partition_num = 0;

    
    
    CandidateQueue_v3(uint* shared_gpu_mem, uint max_parts)
        : partition_shared_ptr(shared_gpu_mem), max_partition_num(max_parts) {

        uint* p_size = nullptr;
        uint* p_host_size = nullptr;

        
        CUDA_ERROR_CHECK(cudaMalloc((void**) &p_size, max_partition_num * sizeof(uint)));
        size_.reset(p_size);
        CUDA_ERROR_CHECK(cudaMemset(size_.get(), 0, max_partition_num * sizeof(uint)));

        
        CUDA_ERROR_CHECK(cudaMallocHost((void**) &p_host_size, max_partition_num * sizeof(uint)));
        host_size_.reset(p_host_size);
        std::fill(host_size_.get(), host_size_.get() + max_partition_num, 0);
    }

    ~CandidateQueue_v3() = default; 
    CandidateQueue_v3(const CandidateQueue_v3&) = delete;
    CandidateQueue_v3& operator=(const CandidateQueue_v3&) = delete;
    CandidateQueue_v3(CandidateQueue_v3&& other) noexcept
        : partition_shared_ptr(other.partition_shared_ptr), 
          size_(std::move(other.size_)),                    
          host_size_(std::move(other.host_size_)),          
          max_partition_num(other.max_partition_num)        
    {
        
        other.partition_shared_ptr = nullptr;
        other.max_partition_num = 0;
    }

    
    
    
    
    __host__ void push(uint partition_idx, cudaStream_t stream = nullptr) {
        
        host_size_.get()[partition_idx]++;

        
        
        
        uint new_val = host_size_.get()[partition_idx];
        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            size_.get() + partition_idx, 
            &new_val,                    
            sizeof(uint),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    
    
    Partition_Device_viewer_v3 get_gpu_viewer() {
        
        
        return {partition_shared_ptr, size_.get(), nullptr};
    }

    
    void clear(cudaStream_t stream = 0) {
        std::fill(host_size_.get(), host_size_.get() + max_partition_num, 0);
        CUDA_ERROR_CHECK(cudaMemsetAsync(size_.get(), 0, max_partition_num * sizeof(uint), stream));
    }
};

class Partition_v3 {
private:
    


    
    static uint calculate_partition_num(uint max_degree) {
        if (max_degree == 0) return 1;

        uint num_partitions = 1;
        uint threshold = 32;

        while (max_degree >= threshold) {
            num_partitions++;
            threshold *= 32;
            
            if (threshold < 32 || num_partitions >= 10) break;  
        }

        return num_partitions;
    }

    inline void initialize() {
        uint* p_size      = nullptr;
        uint* p_idxs      = nullptr;
        uint* p_host_size = nullptr;

        
        CUDA_ERROR_CHECK(cudaMalloc((void**) &p_size, max_partition_num_ * sizeof(uint)));
        size_.reset(p_size);
        CUDA_ERROR_CHECK(cudaMemset(size_.get(), 0, max_partition_num_ * sizeof(uint)));

        CUDA_ERROR_CHECK(cudaMalloc((void**) &p_idxs, max_partition_num_ * sizeof(uint)));
        idxs_.reset(p_idxs);
        CUDA_ERROR_CHECK(cudaMemset(idxs_.get(), 0, max_partition_num_ * sizeof(uint)));

        CUDA_ERROR_CHECK(cudaMallocHost((void**) &p_host_size, max_partition_num_ * sizeof(uint)));
        host_size_.reset(p_host_size);
        std::fill(host_size_.get(), host_size_.get() + max_partition_num_, 0);
    }

public:
    static uint max_partition_num_;
    
    CudaDevicePtr<uint> partition_original_ptr_ = nullptr;
    CudaDevicePtr<uint> size_                   = nullptr;
    CudaDevicePtr<uint> idxs_                   = nullptr;
    CudaHostPtr<uint> host_size_                = nullptr;

    
    uint* partition                    = nullptr;
    uint totalSize                     = 0;
    uint partition_original_malloc_num = 0;

    
    static void set_max_partition_num_from_degree(uint max_degree) {
        max_partition_num_ = calculate_partition_num(max_degree);
        CUDA_ERROR_CHECK(cudaMemcpyToSymbol(d_max_partition_num, &max_partition_num_, sizeof(uint)));
        assert(max_partition_num_ < 7);
    }

    static uint get_max_partition_num() {
        return max_partition_num_;
    }

    
    Partition_v3() {
        initialize();
    }

    __host__ Partition_v3(const uint size) {
        initialize();
        uint* d_ptr       = nullptr;
        const uint length = size + 1;
        cudaError_t err   = cudaMalloc((void**) &d_ptr, length * sizeof(uint));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory in Partition_v3.");
        }

        partition_original_ptr_.reset(d_ptr);
        partition                     = d_ptr;
        partition_original_malloc_num = length;
    }

    ~Partition_v3() = default;

    
    Partition_v3(Partition_v3&& other) noexcept = default;

    Partition_v3& operator=(Partition_v3&& other) noexcept {
        if (this != &other) {
            partition_original_ptr_       = std::move(other.partition_original_ptr_);
            size_                         = std::move(other.size_);
            idxs_                         = std::move(other.idxs_);
            host_size_                    = std::move(other.host_size_);
            partition                     = other.partition;
            totalSize                     = other.totalSize;
            partition_original_malloc_num = other.partition_original_malloc_num;

            other.partition                     = nullptr;
            other.totalSize                     = 0;
            other.partition_original_malloc_num = 0;
        }
        return *this;
    }

    
    Partition_v3(const Partition_v3& other) = delete;
    static void operator_equal(){

    }
    Partition_v3& operator=(const Partition_v3& other) {
        if (this == &other) {
            return *this;
        }

        Partition_v3 temp;
        temp.totalSize                     = other.totalSize;
        temp.partition_original_malloc_num = other.partition_original_malloc_num;

        if (other.host_size_) {
            std::copy(other.host_size_.get(), other.host_size_.get() + max_partition_num_, temp.host_size_.get());
        }

        if (other.size_) {
            CUDA_ERROR_CHECK(
                cudaMemcpy(temp.size_.get(), other.size_.get(), max_partition_num_ * sizeof(uint), cudaMemcpyDeviceToDevice));
        }

        if (other.idxs_) {
            CUDA_ERROR_CHECK(
                cudaMemcpy(temp.idxs_.get(), other.idxs_.get(), max_partition_num_ * sizeof(uint), cudaMemcpyDeviceToDevice));
        }

        if (other.partition_original_ptr_) {
            uint* new_partition_data = nullptr;
            CUDA_ERROR_CHECK(
                cudaMalloc((void**) &new_partition_data, other.partition_original_malloc_num * sizeof(uint)));
            temp.partition_original_ptr_.reset(new_partition_data);

            CUDA_ERROR_CHECK(cudaMemcpy(temp.partition_original_ptr_.get(), other.partition_original_ptr_.get(),
                other.partition_original_malloc_num * sizeof(uint), cudaMemcpyDeviceToDevice));

            ptrdiff_t offset = other.partition - other.partition_original_ptr_.get();
            temp.partition   = temp.partition_original_ptr_.get() + offset;
        }

        swap(*this, temp);
        return *this;
    }

    friend void swap(Partition_v3& first, Partition_v3& second) noexcept {
        using std::swap;
        swap(first.partition_original_ptr_, second.partition_original_ptr_);
        swap(first.size_, second.size_);
        swap(first.idxs_, second.idxs_);
        swap(first.host_size_, second.host_size_);
        swap(first.partition, second.partition);
        swap(first.totalSize, second.totalSize);
        swap(first.partition_original_malloc_num, second.partition_original_malloc_num);
    }

    
    __host__ ptrdiff_t get_bias() const {
        return partition - partition_original_ptr_.get();
    }

    __host__ void CopyPartitionFromX(const Partition_v3& other, uint partition_idx, cudaStream_t stream = nullptr) {
        assert(this->partition_original_ptr_ && "Destination 'this' partition buffer is not allocated.");
        assert(other.partition_original_ptr_ && "Source 'other' partition buffer is not allocated.");
        assert(this->partition_original_malloc_num >= other.totalSize + 1
               && "Destination 'this' buffer must be large enough to hold 'other' + 1 element.");

        std::copy(other.host_size_.get(), other.host_size_.get() + max_partition_num_, this->host_size_.get());
        this->host_size_.get()[partition_idx] += 1;

        uint offset_this  = 0;
        uint offset_other = 0;

        for (uint i = 0; i < max_partition_num_; ++i) {
            uint elements_to_copy = other.host_size_.get()[i];

            if (elements_to_copy > 0) {
                uint* dst_ptr       = this->partition_original_ptr_.get() + offset_this;
                const uint* src_ptr = other.partition_original_ptr_.get() + offset_other;

                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
            }

            offset_this += this->host_size_.get()[i];
            offset_other += other.host_size_.get()[i];
        }

        this->totalSize = other.totalSize + 1;
        this->partition = this->partition_original_ptr_.get() + other.get_bias();

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(this->idxs_.get(), other.idxs_.get(), max_partition_num_ * sizeof(uint), cudaMemcpyDeviceToDevice, stream));

        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            this->size_.get(), this->host_size_.get(), max_partition_num_ * sizeof(uint), cudaMemcpyHostToDevice, stream));
    }
    __host__ tuple<uint, uint*> cal_new_size(cudaStream_t stream) {
        uint* new_host_sizes = new uint[max_partition_num_];
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(new_host_sizes, size_.get(), max_partition_num_ * sizeof(uint), cudaMemcpyDeviceToHost, stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        const uint length = std::accumulate(new_host_sizes, new_host_sizes + max_partition_num_, 0u);
        return {length, new_host_sizes};
    }

    __host__ void malloc_partition(cudaStream_t stream) {
        auto [length, new_host_sizes] = cal_new_size(stream);
        malloc_partition(length, new_host_sizes, stream);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    }
    __host__ void malloc_partition(uint length, uint* new_host_sizes, cudaStream_t stream) {
        if (length == 0) {
            partition_original_ptr_.reset(nullptr);
            partition_original_malloc_num = 0;
            partition                     = nullptr;
            totalSize                     = 0;
            std::fill(host_size_.get(), host_size_.get() + max_partition_num_, 0);
            delete[] new_host_sizes;
            return;
        }

        const bool must_realloc_due_to_growth = length > partition_original_malloc_num;
        const bool should_realloc_due_to_shrink =
            (partition_original_malloc_num > 0) && (length < static_cast<uint>(partition_original_malloc_num * 0.25));

        if (must_realloc_due_to_growth || should_realloc_due_to_shrink) {
            uint* p_partition = nullptr;
            CUDA_ERROR_CHECK(cudaMalloc((void**) &p_partition, length * sizeof(uint)));
            partition_original_ptr_.reset(p_partition);
            partition_original_malloc_num = length;
        }

        partition = partition_original_ptr_.get();
        totalSize = length;
        std::copy(new_host_sizes, new_host_sizes + max_partition_num_, host_size_.get());
        delete[] new_host_sizes;
    }

    __host__ void try_to_remalloc_partition(cudaStream_t stream) {
        auto [new_total_malloc_num, new_partition_sizes] = cal_new_size(stream);
        
        
        
        
        
        
        if (new_total_malloc_num == 0) {
            clean(stream);
            delete[] new_partition_sizes;
            return;
        }

        auto migrate_data_logic = [&](uint* dst_base_ptr) {
            uint old_offset = 0;
            uint new_offset = 0;
            for (uint i = 0; i < max_partition_num_; ++i) {
                uint elements_to_copy = host_size_.get()[i];
                if (elements_to_copy > 0) {
                    uint* src_ptr = partition_original_ptr_.get() + old_offset;
                    uint* dst_ptr = dst_base_ptr + new_offset;

                    if (src_ptr != dst_ptr) {
                        CUDA_ERROR_CHECK(cudaMemcpyAsync(
                            dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
                    }
                }
                old_offset += host_size_.get()[i];
                new_offset += new_partition_sizes[i];
            }
        };

        if (new_total_malloc_num > partition_original_malloc_num) {
            uint* new_partition_data = nullptr;
            CUDA_ERROR_CHECK(cudaMallocAsync(&new_partition_data, new_total_malloc_num * sizeof(uint), stream));
            CudaDevicePtr<uint> new_partition_ptr(new_partition_data);
            migrate_data_logic(new_partition_data);
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

            ptrdiff_t offset_from_original = partition != nullptr ? partition - partition_original_ptr_.get() : 0;
            partition_original_ptr_.swap(new_partition_ptr);
            partition_original_malloc_num = new_total_malloc_num;
            partition                     = partition_original_ptr_.get() + offset_from_original;
        } else {
            partition = partition_original_ptr_.get();
            migrate_data_logic(partition_original_ptr_.get());
        }

        totalSize = new_total_malloc_num;
        std::copy(new_partition_sizes, new_partition_sizes + max_partition_num_, host_size_.get());
        delete[] new_partition_sizes;
    }

    __host__ void try_to_remalloc_partition_shrink(cudaStream_t stream) {
        auto [new_total_malloc_num, new_partition_sizes] = cal_new_size(stream);
        
        
        
        
        
        

        if (new_total_malloc_num == 0) {
            clean(stream);
            delete[] new_partition_sizes;
            return;
        }

        auto migrate_data_logic = [&](uint* dst_base_ptr) {
            uint old_offset = 0;
            uint new_offset = 0;
            for (uint i = 0; i < max_partition_num_; ++i) {
                uint elements_to_copy = host_size_.get()[i];
                if (elements_to_copy > 0) {
                    uint* src_ptr = partition_original_ptr_.get() + old_offset;
                    uint* dst_ptr = dst_base_ptr + new_offset;
                    if (src_ptr != dst_ptr) {
                        CUDA_ERROR_CHECK(cudaMemcpyAsync(
                            dst_ptr, src_ptr, elements_to_copy * sizeof(uint), cudaMemcpyDeviceToDevice, stream));
                    }
                }
                old_offset += host_size_.get()[i];
                new_offset += new_partition_sizes[i];
            }
        };

        const bool must_realloc_due_to_growth   = new_total_malloc_num > partition_original_malloc_num;
        const bool should_realloc_due_to_shrink = (partition_original_malloc_num > 0) && (new_total_malloc_num > 0)
                                               && (partition_original_malloc_num > new_total_malloc_num * 4);

        if (must_realloc_due_to_growth || should_realloc_due_to_shrink) {
            uint* new_partition_data = nullptr;
            CUDA_ERROR_CHECK(cudaMallocAsync(&new_partition_data, new_total_malloc_num * sizeof(uint), stream));
            CudaDevicePtr<uint> new_partition_ptr(new_partition_data);
            migrate_data_logic(new_partition_data);

            ptrdiff_t offset_from_original = partition != nullptr ? partition - partition_original_ptr_.get() : 0;
            partition_original_ptr_.swap(new_partition_ptr);
            partition_original_malloc_num = new_total_malloc_num;
            partition                     = partition_original_ptr_.get() + offset_from_original;
        } else {
            partition = partition_original_ptr_.get();
            migrate_data_logic(partition_original_ptr_.get());
        }

        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        totalSize = new_total_malloc_num;
        std::copy(new_partition_sizes, new_partition_sizes + max_partition_num_, host_size_.get());
        delete[] new_partition_sizes;
    }

    __host__ uint get_partition_idx_from_global_idx(const uint global_index) const {
        const uint* host_size_ptr = this->host_size();
        uint cumulative_size = 0;

        for (uint i = 0; i < max_partition_num_; ++i) {
            cumulative_size += host_size_ptr[i];
            if (global_index < cumulative_size) {
                return i;
            }
        }
        return max_partition_num_ - 1;
    }

    __host__ void from_vector(
        const std::vector<uint>& vec, const std::vector<uint>& partition_size, cudaStream_t stream = nullptr) {
        assert(partition_original_ptr_ == nullptr);
        assert(partition_size.size() == max_partition_num_);

        totalSize                     = vec.size();
        partition_original_malloc_num = totalSize;
        if (totalSize == 0) {
            return;
        }

        uint* p_partition = nullptr;
        CUDA_ERROR_CHECK(cudaMallocAsync((void**) &p_partition, totalSize * sizeof(uint), stream));
        partition_original_ptr_.reset(p_partition);
        partition = partition_original_ptr_.get();

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(partition, vec.data(), totalSize * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(size_.get(), partition_size.data(), max_partition_num_ * sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(idxs_.get(), partition_size.data(), max_partition_num_ * sizeof(uint), cudaMemcpyHostToDevice, stream));

        std::copy(partition_size.begin(), partition_size.end(), host_size_.get());
    }

    __host__ void advance_and_update(cudaStream_t stream = nullptr) {
        if (totalSize == 0) {
            return;
        }

        int target_partition_idx = -1;
        for (uint i = 0; i < max_partition_num_; ++i) {
            if (host_size_.get()[i] > 0) {
                target_partition_idx = i;
                break;
            }
        }
        assert(target_partition_idx != -1 && "Inconsistent state.");

        host_size_.get()[target_partition_idx]--;
        totalSize--;
        partition++;

        uint* device_size_ptr = size_.get() + target_partition_idx;
        uint* device_idxs_ptr = idxs_.get() + target_partition_idx;
        uint new_val          = host_size_.get()[target_partition_idx];
        CUDA_ERROR_CHECK(cudaMemcpyAsync(device_size_ptr, &new_val, sizeof(uint), cudaMemcpyHostToDevice, stream));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(device_idxs_ptr, &new_val, sizeof(uint), cudaMemcpyHostToDevice, stream));
    }

    void clean(cudaStream_t stream = nullptr) {
        CUDA_ERROR_CHECK(cudaMemsetAsync(size_.get(), 0, max_partition_num_ * sizeof(uint), stream));
        CUDA_ERROR_CHECK(cudaMemsetAsync(idxs_.get(), 0, max_partition_num_ * sizeof(uint), stream));
        std::fill(host_size_.get(), host_size_.get() + max_partition_num_, 0);
        totalSize = 0;
        partition = nullptr;
    }

    bool empty() const {
        return totalSize == 0;
    }

    bool is_partition_allocated() const {
        return partition_original_ptr_ != nullptr;
    }

    Partition_Device_viewer_v3 get_gpu_viewer() {
        return {partition, size_.get(), idxs_.get()};
    }

    __host__ void reset() {
        assert(partition_original_ptr_ == nullptr && partition != nullptr);
        partition_original_ptr_.reset(partition);
        partition_original_malloc_num = totalSize;
    }

    uint total_size() const {
        return totalSize;
    }

    const uint* host_size() const {
        return host_size_.get();
    }


    
    static Partition_v3 MergeAndCreate(const Partition_v3& p_obj, const CandidateQueue_v3& q_obj, cudaStream_t stream = 0) {
        
        Partition_v3 new_partition;

        
        assert(p_obj.max_partition_num_ == q_obj.max_partition_num);

        uint total_new_size = 0;

        
        for (uint i = 0; i < max_partition_num_; ++i) {
            uint p_sz = p_obj.host_size_.get()[i];
            uint q_sz = q_obj.host_size_.get()[i];
            uint new_sz = p_sz + q_sz;

            new_partition.host_size_.get()[i] = new_sz;
            total_new_size += new_sz;
        }

        new_partition.totalSize = total_new_size;
        new_partition.partition_original_malloc_num = total_new_size;

        
        if (total_new_size == 0) {
            return new_partition;
        }

        
        uint* d_ptr = nullptr;
        CUDA_ERROR_CHECK(cudaMallocAsync((void**)&d_ptr, total_new_size * sizeof(uint), stream));
        new_partition.partition_original_ptr_.reset(d_ptr);
        new_partition.partition = d_ptr; 

        
        uint offset_p = 0;
        uint offset_q = 0;
        uint offset_new = 0;

        
        
        
        
        
        const uint* src_p_base = p_obj.partition_original_ptr_.get();
        const uint* src_q_base = q_obj.partition_shared_ptr;
        uint* dst_base = new_partition.partition_original_ptr_.get();

        for (uint i = 0; i < max_partition_num_; ++i) {
            uint p_sz = p_obj.host_size_.get()[i];
            uint q_sz = q_obj.host_size_.get()[i];

            
            if (p_sz > 0) {
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    dst_base + offset_new,          
                    src_p_base + offset_p,          
                    p_sz * sizeof(uint),
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
            }

            
            if (q_sz > 0) {
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    dst_base + offset_new + p_sz,   
                    src_q_base + offset_q,          
                    q_sz * sizeof(uint),
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
            }

            
            offset_p += p_sz;
            offset_q += q_sz;
            offset_new += (p_sz + q_sz);
        }

        
        
        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            new_partition.size_.get(),
            new_partition.host_size_.get(),
            max_partition_num_ * sizeof(uint),
            cudaMemcpyHostToDevice,
            stream
        ));

        CUDA_ERROR_CHECK(cudaMemcpyAsync(
            new_partition.idxs_.get(),
            new_partition.size_.get(),
            max_partition_num_ * sizeof(uint),
            cudaMemcpyHostToDevice,
            stream
        ));

        return new_partition;
    }
};


uint Partition_v3::max_partition_num_ = 4;  




#endif 
