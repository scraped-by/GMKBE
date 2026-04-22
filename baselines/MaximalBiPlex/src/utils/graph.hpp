#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include "hash.hpp"

struct Graph {
	uint32_t n;
	uint32_t maxDeg;
	std::vector<uint32_t> degree;
	std::vector<CuckooHash> nbrMap;
	std::vector<std::vector<uint32_t>> nbr;

	Graph() {
		clear();
	}

	Graph(uint32_t size) {
		resize(size);
	}

	void clear() {
		n = 0;
		degree.clear();
		nbr.clear();
		nbrMap.clear();
	}

	void resize(uint32_t size) {
		n = size;
		degree.resize(size);
		nbrMap.resize(size);
		nbr.resize(size);
	}

	void addEdge(uint32_t u, uint32_t v) {
		uint32_t w = std::max(u, v);
		if (w >= n) resize(w + 1);
		if (nbrMap[u].find(v)) return;
		nbr[u].push_back(v);
		nbr[v].push_back(u);
		nbrMap[u].insert(v);
		nbrMap[v].insert(u);
		++degree[u];
		++degree[v];
		maxDeg = std::max(maxDeg, degree[u]);
		maxDeg = std::max(maxDeg, degree[v]);
	}
};

#endif