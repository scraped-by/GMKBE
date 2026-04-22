



#ifndef TMP_LOAD_H
#define TMP_LOAD_H
#include <partition.cuh>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/color.h>

template<typename T>
void load_graph_from_csr(const std::string& filename,
                         std::vector<T>& neighbors,
                         std::vector<T>& degrees,
                         std::vector<T>& neighbors_offset,
                         T& size,
                         T& bi_partition_idx, T& k, T& theta) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        if (not filesystem::exists(filename))
            fmt::print(stderr, "Error: File does not exist.\n");
        exit(-1);
    }

    
    neighbors.clear();
    degrees.clear();
    neighbors_offset.clear();

    
    infile >> size >> bi_partition_idx >> k >> theta;

    
    if (size == 0) {
        neighbors_offset.push_back(0); 
        std::cout << "Loaded an empty graph." << std::endl;
        return;
    }

    
    degrees.resize(size);
    neighbors_offset.resize(size + 1);
    neighbors_offset[0] = 0;

    std::string line;
    
    std::getline(infile, line);

    
    for (int i = 0; i < size; ++i) {
        std::getline(infile, line);
        std::stringstream ss(line);

        
        T deg;
        ss >> deg;
        degrees[i] = deg;

        
        neighbors_offset[i + 1] = neighbors_offset[i] + deg;

        
        T neighbor_id;
        for (int j = 0; j < deg; ++j) {
            ss >> neighbor_id;
            neighbors.push_back(neighbor_id);
        }
    }

    infile.close();
    std::cout << "Graph successfully loaded from " << filename << std::endl;
}


template<typename T>
void load_graph_from_csr(const std::string& filename,
                         std::vector<T>& neighbors,
                         std::vector<T>& degrees,
                         std::vector<T>& neighbors_offset,
                         T& size,
                         T& bi_partition_idx, T& core_num) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        if (not filesystem::exists(filename))
            fmt::print(stderr, "Error: File does not exist.\n");
        exit(-1);
    }

    
    neighbors.clear();
    degrees.clear();
    neighbors_offset.clear();

    
    infile >> size >> bi_partition_idx >> core_num;

    
    if (size == 0) {
        neighbors_offset.push_back(0); 
        std::cout << "Loaded an empty graph." << std::endl;
        return;
    }

    
    degrees.resize(size);
    neighbors_offset.resize(size + 1);
    neighbors_offset[0] = 0;

    std::string line;
    
    std::getline(infile, line);

    
    for (int i = 0; i < size; ++i) {
        std::getline(infile, line);
        std::stringstream ss(line);

        
        T deg;
        ss >> deg;
        degrees[i] = deg;

        
        neighbors_offset[i + 1] = neighbors_offset[i] + deg;

        
        T neighbor_id;
        for (int j = 0; j < deg; ++j) {
            ss >> neighbor_id;
            neighbors.push_back(neighbor_id);
        }
    }

    infile.close();
    std::cout << "Graph successfully loaded from " << filename << std::endl;
}


template<typename T>
void load_graph_from_csr(const std::string& filename,
                         std::vector<T>& neighbors,
                         std::vector<T>& degrees,
                         std::vector<T>& neighbors_offset,
                         T& size,
                         T& bi_partition_idx) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        if (!std::filesystem::exists(filename))
            std::cerr << "Error: File does not exist.\n";
        std::exit(-1);
    }

    neighbors.clear();
    degrees.clear();
    neighbors_offset.clear();

    
    
    
    
    T nEdge;
    infile >> nEdge >> size >> bi_partition_idx;

    if (size == 0) {
        neighbors_offset.push_back(0);
        std::cout << "Loaded an empty graph." << std::endl;
        return;
    }

    degrees.resize(size);
    neighbors_offset.resize(size + 1);
    neighbors_offset[0] = 0;
    neighbors.reserve(static_cast<size_t>(nEdge));

    
    std::string line;
    std::getline(infile, line);

    
    for (T i = 0; i < size; ++i) {
        if (!std::getline(infile, line)) {
            std::cerr << "Error: Unexpected EOF at vertex " << i << std::endl;
            std::exit(-1);
        }
        std::stringstream ss(line);

        T node_id;
        ss >> node_id;            
        
        

        T deg;
        ss >> deg;
        degrees[i] = deg;
        neighbors_offset[i + 1] = neighbors_offset[i] + deg;

        T nbr;
        for (T j = 0; j < deg; ++j) {
            ss >> nbr;
            neighbors.push_back(nbr);
        }
    }

    infile.close();
    std::cout << "Graph successfully loaded from " << filename << std::endl;
}


template<typename T>
void load_graph_from_bin(const std::string& filename,
                         std::vector<T>& neighbors,
                         std::vector<T>& degrees,
                         std::vector<T>& neighbors_offset,
                         T& size,
                         T& bi_partition_idx) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        if (!std::filesystem::exists(filename))
            std::cerr << "Error: File does not exist.\n";
        std::exit(-1);
    }

    neighbors.clear();
    degrees.clear();
    neighbors_offset.clear();

    
    uint32_t nEdge = 0, nodeCount = 0, bipartite_node = 0;
    if (std::fread(&nEdge,          4, 1, fp) != 1 ||
        std::fread(&nodeCount,      4, 1, fp) != 1 ||
        std::fread(&bipartite_node, 4, 1, fp) != 1) {
        std::cerr << "Error: Failed to read header.\n";
        std::fclose(fp);
        std::exit(-1);
    }

    size             = static_cast<T>(nodeCount);
    bi_partition_idx = static_cast<T>(bipartite_node);

    if (size == 0) {
        neighbors_offset.push_back(0);
        std::fclose(fp);
        std::cout << "Loaded an empty graph." << std::endl;
        return;
    }

    degrees.resize(size);
    neighbors_offset.resize(size + 1);
    neighbors_offset[0] = 0;
    neighbors.resize(static_cast<size_t>(nEdge));

    size_t write_pos = 0;
    for (T i = 0; i < size; ++i) {
        uint32_t id = 0, deg = 0;
        if (std::fread(&id,  4, 1, fp) != 1 ||
            std::fread(&deg, 4, 1, fp) != 1) {
            std::cerr << "Error: Failed to read vertex record at " << i << std::endl;
            std::fclose(fp);
            std::exit(-1);
        }

        degrees[i] = static_cast<T>(deg);
        neighbors_offset[i + 1] = neighbors_offset[i] + static_cast<T>(deg);

        if (deg > 0) {
            if constexpr (sizeof(T) == 4) {
                
                if (std::fread(neighbors.data() + write_pos, 4, deg, fp) != deg) {
                    std::cerr << "Error: Failed to read neighbors at " << i << std::endl;
                    std::fclose(fp);
                    std::exit(-1);
                }
                write_pos += deg;
            } else {
                
                std::vector<uint32_t> buf(deg);
                if (std::fread(buf.data(), 4, deg, fp) != deg) {
                    std::cerr << "Error: Failed to read neighbors at " << i << std::endl;
                    std::fclose(fp);
                    std::exit(-1);
                }
                for (uint32_t k = 0; k < deg; ++k)
                    neighbors[write_pos++] = static_cast<T>(buf[k]);
            }
        }
    }

    std::fclose(fp);
    std::cout << "Graph successfully loaded from " << filename << std::endl;
}

template<class T>
inline void output(const string name, vector<T>datas) {
    for (size_t i = 0; i < datas.size(); ++i) {
        if (i != 0) fmt::print(", ");
        if (i == 50) {
            fmt::print(fg(fmt::color::red), "...");
            break;
        }
        fmt::print("{}", datas[i]);
    }
    fmt::println("]");
}

template<class T>
inline void output(string name, shared_ptr<T> partition, fmt::text_style style = fmt::text_style()) {
    fmt::print(style, "{}(size = {}): [", name, partition->totalSize);
    if (partition->totalSize > 0) {
        uint *tmp_newPartition = new uint[partition->totalSize];
        CUDA_ERROR_CHECK(cudaMemcpyAsync(tmp_newPartition, partition->partition, partition->totalSize * sizeof(uint), cudaMemcpyDeviceToHost));
        vector<uint> host_partition (tmp_newPartition, tmp_newPartition + partition->totalSize);
        output(name, host_partition);
    }
    else fmt::println("]");
}

template<class T>
inline void output(string name, T& partition, fmt::text_style style = fmt::text_style()) {
    fmt::print(style, "{}(size = {}): [", name, partition.totalSize);
    if (partition.totalSize > 0) {
        uint *tmp_newPartition = new uint[partition.totalSize];
        CUDA_ERROR_CHECK(cudaMemcpyAsync(tmp_newPartition, partition.partition, partition.totalSize * sizeof(uint), cudaMemcpyDeviceToHost));
        vector<uint> host_partition (tmp_newPartition, tmp_newPartition + partition.totalSize);
        output(name, host_partition);
    }
    else fmt::println("]");
}

#endif 
