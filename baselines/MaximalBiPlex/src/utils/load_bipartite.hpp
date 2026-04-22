#ifndef LOAD_BIPARTITE_HPP
#define LOAD_BIPARTITE_HPP

#include "bigraph.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>

inline BiGraph loadBipartiteGraphFromBin(const std::string& filename) {
	FILE* fp = std::fopen(filename.c_str(), "rb");
	if (!fp) {
		std::cerr << "Error: cannot open " << filename << std::endl;
		return BiGraph();
	}

	uint32_t nEdge = 0, nodeCount = 0, bipartite_idx = 0;
	if (std::fread(&nEdge, 4, 1, fp) != 1 ||
		std::fread(&nodeCount, 4, 1, fp) != 1 ||
		std::fread(&bipartite_idx, 4, 1, fp) != 1) {
		std::cerr << "Error: failed to read header from " << filename << std::endl;
		std::fclose(fp);
		return BiGraph();
	}

	if (bipartite_idx > nodeCount) {
		std::cerr << "Error: invalid bipartite index in " << filename << std::endl;
		std::fclose(fp);
		return BiGraph();
	}

	BiGraph G;
	G.nLabels[0] = bipartite_idx;
	G.nLabels[1] = nodeCount - bipartite_idx;

	std::vector<uint32_t> neighbors;
	for (uint32_t i = 0; i < nodeCount; ++i) {
		uint32_t id = 0, deg = 0;
		if (std::fread(&id, 4, 1, fp) != 1 ||
			std::fread(&deg, 4, 1, fp) != 1) {
			std::cerr << "Error: failed to read vertex record at " << i << " from " << filename << std::endl;
			std::fclose(fp);
			return BiGraph();
		}

		neighbors.resize(deg);
		if (deg > 0 && std::fread(neighbors.data(), 4, deg, fp) != deg) {
			std::cerr << "Error: failed to read neighbors for vertex " << id << " from " << filename << std::endl;
			std::fclose(fp);
			return BiGraph();
		}

		if (id < bipartite_idx) continue;

		const uint32_t rightLabel = id - bipartite_idx;
		for (uint32_t leftGlobal : neighbors) {
			if (leftGlobal >= bipartite_idx) {
				std::cerr << "Error: invalid left vertex label " << leftGlobal
						  << " for right vertex " << id << " in " << filename << std::endl;
				std::fclose(fp);
				return BiGraph();
			}
			G.addEdgeWithLabel(leftGlobal, rightLabel);
		}
	}

	if (G.m != nEdge) {
		std::cerr << "Warning: header edge count " << nEdge
				  << " does not match loaded edge count " << G.m
				  << " from " << filename << std::endl;
	}

	std::fclose(fp);
	return G;
}

#endif
