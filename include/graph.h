#ifndef CUTS_GRAPH_H
#define CUTS_GRAPH_H
#include <kcore_24_10_08/common.h>
#include <filesystem>
#include <fmt/core.h>
#include <sys/mman.h>  
#include <fcntl.h>     
#include <unistd.h>    
#include <sys/stat.h>  
#include <sstream>

class Graph {
public:
	uint V;
	uint E;
	uint bipartitePoint;
	uint *neighbors, *neighbors_map;
	uint *neighbors_offset;
	uint *degrees;
	uint *core_number;
	uint thresh;

	Graph(const std::filesystem::path &input_file, int thresh): thresh(thresh) {
		if (input_file.has_extension()) {
			const auto extension = input_file.extension();
			if (extension == ".bin" or extension == ".graph")
				readGraphFile(input_file);
			else ERROR_CALL("Extension is not right");
		}
	}

	Graph() = default;

	bool readGraphFile(const std::filesystem::path &input_file) {
		
		int fd = open(input_file.c_str(), O_RDONLY);
		if (fd == -1) {
			ERROR_CALL("path: " + input_file.string() + " - File couldn't open");
			return false;
		}

		
		struct stat sb;
		if (fstat(fd, &sb) == -1) {
			close(fd);
			ERROR_CALL("Failed to get file size for: " + input_file.string());
			return false;
		}

		
		char *fileData = static_cast<char *>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
		if (fileData == MAP_FAILED) {
			close(fd);
			ERROR_CALL("Memory mapping failed for file: " + input_file.string());
			return false;
		}

		
		size_t index = 0;

		
        std::memcpy(&E, fileData + index, sizeof(uint));
        index += sizeof(uint);
		std::memcpy(&V, fileData + index, sizeof(uint));
		index += sizeof(uint);
        std::memcpy(&bipartitePoint, fileData + index, sizeof(uint));
        index += sizeof(uint);
		bipartitePoint --;

		
		neighbors = new uint[E];
		neighbors_map = new uint[E];
		neighbors_offset = new uint[V + 1];
		degrees = new uint[V + 1];
		core_number = new uint[V];

		neighbors_offset[0] = 0;
		
		uint edge = 0;
		for (uint i = 0; i < V; ++i) {
			uint node, degree;
			
			std::memcpy(&node, fileData + index, sizeof(uint));
			index += sizeof(uint);
			std::memcpy(&degree, fileData + index, sizeof(uint));
			index += sizeof(uint);
			
			uint next_node = node + 1;

			degrees[node] = degree;
			neighbors_offset[next_node] = neighbors_offset[node] + degree;

			
			for (uint j = 0; j < degree; ++j) {
				uint neighbor;
				std::memcpy(&neighbor, fileData + index, sizeof(uint));
				index += sizeof(uint);
				
				
				
				neighbors[edge] = neighbor;
				neighbors_map[edge] = node;
				edge ++;
			}
		}
		if (edge != E)
			ERROR_CALL("total degree is not equal to edge number");
		degrees[V] = std::numeric_limits<uint32_t>::max();

		
		munmap(fileData, sb.st_size);
		close(fd);

		return true;
	}

	~Graph() {
		fmt::print("Deallocated...\n");
		delete[] core_number;
		delete[] neighbors;
		delete[] neighbors_map;
		delete[] neighbors_offset;
		delete[] degrees;
	}
};

std::string readline(const char* &data, size_t &remaining_size) {
    if (remaining_size == 0) {
        return ""; 
    }

    
    const char* newline = reinterpret_cast<const char*>(memchr(data, '\n', remaining_size));
    std::string line;

    if (newline) {
        
        size_t line_length = newline - data;
        line.assign(data, line_length);
        
        data = newline + 1;
        remaining_size -= (line_length + 1);
    } else {
        
        line.assign(data, remaining_size);
        remaining_size = 0; 
    }

    return line;
}

uint ReadGraph(string dataset_path, int **&Graph, int *&degrees, bool bipartite = false) {
	
	int fd = open(dataset_path.c_str(), O_RDONLY);
	if (fd == -1) {
		ERROR_CALL("path: " + dataset_path + " - File couldn't open");
		return false;
	}

	
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		close(fd);
		ERROR_CALL("Failed to get file size for: " + dataset_path);
		return false;
	}

	
	char *fileData = static_cast<char *>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
	if (fileData == MAP_FAILED) {
		close(fd);
		ERROR_CALL("Memory mapping failed for file: " + dataset_path);
		return false;
	}

	
	size_t index = 0;
	uint V, E;
	
    std::memcpy(&E, fileData + index, sizeof(uint));
    index += sizeof(uint);
	std::memcpy(&V, fileData + index, sizeof(uint));
	index += sizeof(uint);
	if(bipartite){
        uint tmp;
        std::memcpy(&tmp, fileData + index, sizeof(uint));
        index += sizeof(uint);
    }
	Graph = new int *[V];

	degrees = new int[V];
	uint *neg = new uint[V];
	for (uint i = 0; i < V; ++i) {
		uint node, degree;
		
		std::memcpy(&node, fileData + index, sizeof(uint));
		index += sizeof(uint);
		std::memcpy(&degree, fileData + index, sizeof(uint));
		index += sizeof(uint);

		degrees[node] = degree;
		
		for (uint j = 0; j < degree; ++j) {
			uint neighbor;
			std::memcpy(&neighbor, fileData + index, sizeof(uint));
			index += sizeof(uint);
			neg[j] = neighbor;
		}
		int *temp_array = new int[degree];
		for (int _i = 0; _i < degree; ++_i) {
			temp_array[_i] = neg[_i];
		}
		Graph[node] = temp_array;
	}
	return V;
}

uint ReadGraph(string dataset_path, int **&Graph, int *&degrees, int &rightFirstNode)  {
	
	int fd = open(dataset_path.c_str(), O_RDONLY);
	if (fd == -1) {
		ERROR_CALL("path: " + dataset_path + " - File couldn't open");
		return false;
	}

	
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		close(fd);
		ERROR_CALL("Failed to get file size for: " + dataset_path);
		return false;
	}

	
	char *fileData = static_cast<char *>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
	if (fileData == MAP_FAILED) {
		close(fd);
		ERROR_CALL("Memory mapping failed for file: " + dataset_path);
		return false;
	}

	
	size_t index = 0;
	uint V, E;
	
	std::memcpy(&E, fileData + index, sizeof(uint));
	index += sizeof(uint);
	std::memcpy(&V, fileData + index, sizeof(uint));
	index += sizeof(uint);
	std::memcpy(&rightFirstNode, fileData + index, sizeof(uint));
	index += sizeof(uint);
	Graph = new int *[V];

	degrees = new int[V];
	uint *neg = new uint[V];
	for (uint i = 0; i < V; ++i) {
		uint node, degree;
		
		std::memcpy(&node, fileData + index, sizeof(uint));
		index += sizeof(uint);
		std::memcpy(&degree, fileData + index, sizeof(uint));
		index += sizeof(uint);

		degrees[node - 1] = degree;
		
		for (uint j = 0; j < degree; ++j) {
			uint neighbor;
			std::memcpy(&neighbor, fileData + index, sizeof(uint));
			index += sizeof(uint);
			neg[j] = neighbor;
		}
		int *temp_array = new int[degree];
		for (int _i = 0; _i < degree; ++_i) {
			temp_array[_i] = neg[_i];
		}
		Graph[node - 1] = temp_array;
	}
	return V;
}

uint ReadGraph2(const filesystem::path dataset_path, int **& Graph, int *&degrees, int &rightFirstNode){
    int fd = open(dataset_path.c_str(), O_RDONLY);
    if (fd == -1) {
        ERROR_CALL("path: " + dataset_path.string() + " - File couldn't open");
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        ERROR_CALL("Failed to get file size for: " + dataset_path.string());
        return false;
    }

    
    const char *fileData = static_cast<char *>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (fileData == MAP_FAILED) {
        close(fd);
        ERROR_CALL("Memory mapping failed for file: " + dataset_path.string());
        return false;
    }
    size_t file_size = sb.st_size;
    string FirstLine = readline(fileData, file_size);
    stringstream ss_(FirstLine);
    
    size_t index = 0;
    uint V, E;
    ss_ >> V >> rightFirstNode >> E;
    Graph = new int *[V];
    degrees = new int[V];
    uint *neg = new uint[V];
    for (uint i = 0; i < V; ++i) {
        string line = readline(fileData, file_size);
        stringstream ss(line);
        uint node, degree = 0;
        ss >> node;
        
        while(ss >> neg[degree ++]);
        degree -= 1;
        degrees[node] = degree;
        int *temp_array = new int[degree];
        for (int _i = 0; _i < degree; ++_i) {
            temp_array[_i] = neg[_i];
        }
        Graph[node] = temp_array;
    }
    return V;
}


#endif 
