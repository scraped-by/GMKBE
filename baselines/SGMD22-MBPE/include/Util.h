#ifndef SGMOD22_MBPE_UTILS_H
#define SGMOD22_MBPE_UTILS_H
#include<fstream>
#include<iostream>
#include<string>
#include<list>
#include <filesystem>
#include <sys/mman.h>  // For mmap
#include <fcntl.h>     // For open
#include <unistd.h>    // For close
#include <sys/stat.h>  // For file size
#include <set>
#include <vector>
#include <algorithm>
#include <cstring>
using namespace std;
#define ERROR_CALL(msg){                                                \
		std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << " "    \
			<< msg << std::endl;                                        \
		exit(1);                                                        \
	}                                                                   \

char* itoa(int num,char* str,int radix)
{/*索引表*/
	char index[]="0123456789ABCDEF";
	unsigned unum;/*中间变量*/
	int i=0,j,k;
	/*确定unum的值*/
	if(radix==10&&num<0)/*十进制负数*/
	{
		unum=(unsigned)-num;
		str[i++]='-';
	}
	else unum=(unsigned)num;/*其他情况*/
	/*转换*/
	do{
		str[i++]=index[unum%(unsigned)radix];
		unum/=radix;
	}while(unum);
	str[i]='\0';
	/*逆序*/
	if(str[0]=='-')
		k=1;/*十进制负数*/
	else
		k=0;

	for(j=k;j<=(i-1)/2;j++)
	{       char temp;
		temp=str[j];
		str[j]=str[i-1+k-j];
		str[i-1+k-j]=temp;
	}
	return str;
}

class Util{
public:
	int ReadGraph(string dataset_path,int **&Graph, int *&degree, int &bipartite);
	int ReadGraph(string dataset_path,vector<vector<int>> &Graph, vector<int> &degree, int &bipartite);
	int ReadGraph(string dataset_path,int **&Graph, int *&degree);
};

int Util::ReadGraph(string dataset_path,int **&Graph, int *&degree, int &bipartite){
	ifstream read;
	read.open(dataset_path);
	cout << std::boolalpha << read.is_open() << endl;
	if(not read.is_open()){
		cerr << "graph path not found " << __FILE__ << " " << __LINE__ << endl;
		exit(-1);
	}

	string temp;
	read>>temp;
	int graph_size=stoi(temp);
	Graph=new int*[graph_size];
	delete []degree;
	degree=new int[graph_size];
	read>>temp;
	int B_index=stoi(temp);
	bipartite=B_index;
	read>>temp;
	int index=0;
	int *neg=new int[graph_size];
	char a;
	int temp_count=0;
	bool first=true;
	while(!read.eof()){
		if(first){
			read>>temp;
			first=false;
		}
		read.get(a);
		if(a=='\r')
			continue;
		if(a=='\n'){
			if(index>=graph_size)
				break;
			degree[index]=temp_count;
			int *temp_array=new int[temp_count];
			for(int i=0;i<temp_count;++i){
				temp_array[i] = neg[i];
			}
			Graph[index] = temp_array;
			temp_count=0;
			index++;
			first=true;
			continue;
		}
		read>>temp;
		neg[temp_count]=stoi(temp);
		temp_count++;

	}
	delete []neg;
	return graph_size;
}

int Util::ReadGraph(string dataset_path,vector<vector<int>> &Graph, vector<int> &degree, int &bipartite){
	ifstream read;
	read.open(dataset_path);
	if(not read.is_open()){
		cerr << "graph path not found " << __FILE__ << " " << __LINE__ << endl;
		exit(-1);
	}

	string temp;
	read>>temp;
	int graph_size=stoi(temp);
	Graph.assign(graph_size, vector<int>());
	degree.assign(graph_size, 0);
	read>>temp;
	int B_index=stoi(temp);
	bipartite=B_index;
	read>>temp;
	int index=0;
	int *neg=new int[graph_size];
	char a;
	int temp_count=0;
	bool first=true;
	while(!read.eof()){
		if(first){
			read>>temp;
			first=false;
		}
		read.get(a);
		if(a=='\r')
			continue;
		if(a=='\n'){
			if(index>=graph_size)
				break;
			degree[index]=temp_count;
			int *temp_array=new int[temp_count];
			for(int i=0;i<temp_count;++i)
				temp_array[i] = neg[i];

			Graph[index].assign(temp_array, temp_array + temp_count);
			temp_count=0;
			index++;
			first=true;
			continue;
		}
		read>>temp;
		neg[temp_count]=stoi(temp);
		temp_count++;

	}
	delete []neg;
	return graph_size;
}

tuple<bool, uint, uint> ReadBinGraph(
    const std::filesystem::path& dataset_path,
    std::vector<std::vector<int>>& Graph,
    std::vector<int>& degrees,
    int& rightFirstNode)
{
    int fd = open(dataset_path.c_str(), O_RDONLY);
    if (fd == -1) {
        ERROR_CALL("path: " + dataset_path.string() + " - File couldn't open");
        return {false, 0, 0};
    }

    struct stat sb{};
    if (fstat(fd, &sb) == -1) {
        close(fd);
        ERROR_CALL("Failed to get file size for: " + dataset_path.string());
        return {false, 0, 0};
    }

    if (sb.st_size < 12) {
        close(fd);
        ERROR_CALL("File too small: " + dataset_path.string());
        return {false, 0, 0};
    }

    char* fileData = static_cast<char*>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (fileData == MAP_FAILED) {
        close(fd);
        ERROR_CALL("Memory mapping failed for file: " + dataset_path.string());
        return {false, 0, 0};
    }

    size_t index = 0;
    uint E, V, firstRight;

    std::memcpy(&E, fileData + index, 4); index += 4;
    std::memcpy(&V, fileData + index, 4); index += 4;
    std::memcpy(&firstRight, fileData + index, 4); index += 4;

    rightFirstNode = static_cast<int>(firstRight); // 不要减 1

    uint64_t expectedSize = 12ull + 8ull * V + 4ull * E;
    if (expectedSize != (uint64_t)sb.st_size) {
        munmap(fileData, sb.st_size);
        close(fd);
        ERROR_CALL("Binary format mismatch: file size check failed for " + dataset_path.string());
        return {false, 0, 0};
    }

    degrees.assign(V, 0);
    Graph.assign(V, std::vector<int>());

    uint sumDeg = 0;

    for (uint i = 0; i < V; ++i) {
        uint node, degree;

        std::memcpy(&node, fileData + index, 4); index += 4;
        std::memcpy(&degree, fileData + index, 4); index += 4;

        if (node >= V || index + 4ull * degree > (size_t)sb.st_size) {
            munmap(fileData, sb.st_size);
            close(fd);
            ERROR_CALL("Corrupted binary graph: out-of-range node/degree");
            return {false, 0, 0};
        }

        degrees[node] = static_cast<int>(degree);
        Graph[node].reserve(degree);

        for (uint j = 0; j < degree; ++j) {
            uint neighbor;
            std::memcpy(&neighbor, fileData + index, 4);
            index += 4;
            Graph[node].push_back(static_cast<int>(neighbor));
        }

        sumDeg += degree;
    }

    if (sumDeg != E || index != (size_t)sb.st_size) {
        munmap(fileData, sb.st_size);
        close(fd);
        ERROR_CALL("Binary format mismatch: degree sum or final offset check failed");
        return {false, 0, 0};
    }

    munmap(fileData, sb.st_size);
    close(fd);

    return {true, V, E};
}

std::tuple<bool, uint, uint> ReadTextGraph(const std::filesystem::path dataset_path, std::vector<std::vector<int>>& Graph, std::vector<int>& degrees, int &rightFirstNode) {
	// 打开文本文件
	std::ifstream in(dataset_path);
	if (!in.is_open()) {
		// 假设 ERROR_CALL 是你定义好的宏或函数
		std::cerr << "[ERROR] path: " << dataset_path.string() << " - File couldn't open" << std::endl;
		return {false, 0, 0};
	}

	uint V, bipartite_idx, core_num;
	// 读取头部信息：V bipartite_idx core_num
	if (!(in >> V >> bipartite_idx >> core_num)) {
		std::cerr << "[ERROR] Failed to read graph header for: " << dataset_path.string() << std::endl;
		return {false, 0, 0};
	}

	// 写入代码中 bipartite_idx 就是右侧顶点的起始索引，无需像原代码那样 -1
	rightFirstNode = bipartite_idx;

	// 初始化数据结构
	degrees.assign(V, 0);       // 原代码是 V+1，这里改为 V 更符合 0-indexed 标准
	Graph.assign(V, std::vector<int>());

	uint total_degrees = 0;

	// 逐行读取每个顶点的度数和邻居
	// 注意：由于写入文件没有保存节点原始 ID，这里只能按顺序 0 到 V-1 分配 ID
	for (uint i = 0; i < V; ++i) {
		uint degree;
		if (!(in >> degree)) break;

		degrees[i] = degree;
		Graph[i].reserve(degree);
		total_degrees += degree;

		// 读取邻接节点列表
		for (uint j = 0; j < degree; ++j) {
			uint neighbor;
			in >> neighbor;
			Graph[i].push_back(neighbor);
		}
	}

	in.close();

	// 对于无向二分图，总度数是边数 (E) 的两倍
	uint E = total_degrees / 2; 

	return {true, V, E};
}


int Util::ReadGraph(string dataset_path,int **&Graph, int *&degree){
	ifstream read;
	read.open(dataset_path);
	string temp;
	read>>temp;
	int graph_size=stoi(temp);
	Graph=new int*[graph_size];
	delete []degree;
	degree=new int[graph_size];
	read>>temp;
	int index=0;
	int *neg=new int[graph_size];
	char a;
	int temp_count=0;
	bool first=true;
	while(!read.eof()){
		if(first){
			read>>temp;
			first=false;
		}
		read.get(a);
		if(a=='\n'){
			if(index>=graph_size)
				break;
			degree[index]=temp_count;
			int *temp_array=new int[temp_count];
			for(int i=0;i<temp_count;++i){
				temp_array[i]=neg[i];
			}
			Graph[index]=temp_array;
			temp_count=0;
			index++;
			first=true;
			continue;
		}
		read>>temp;
		neg[temp_count]=stoi(temp);
		temp_count++;

	}
	delete []neg;
	return graph_size;
}

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
std::set<int> checkpoints;
#endif //SGMOD22_MBPE_UTILS_H