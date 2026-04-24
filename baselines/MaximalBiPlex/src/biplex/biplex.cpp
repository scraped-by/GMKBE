#include "biplex.h"
#include "../utils/linearheap.hpp"
#include "../utils/graph.hpp"
#include "../utils/vertexset.hpp"
#include <cinttypes>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <queue>




#define SMALL_K 1

#define PRUNE_C
#define PRUNE_ONCE




#define STRINGIFY(name) #name

#define PRINT_MACRO_INFO(name) do { \
	if (#name [0] != STRINGIFY(name) [0]) \
		printf("%s: on\n", #name); \
	else \
		printf("%s: off\n", #name); \
} while (0)

namespace biplex {
	BiGraph G;
	VertexSet S[2], C[2], X[2], Q[2];
	std::vector<std::vector<uint32_t>> cand(100000);
	std::vector<std::vector<uint32_t>> nonNbrS[2];
	std::vector<uint32_t> nonNbrC;
	std::vector<uint32_t> degC[2];
	std::vector<uint32_t> ordered[2], order[2], core[2];
	std::vector<uint32_t> sup, ptrB;
	std::vector<std::vector<uint32_t>> B;
	uint32_t q, k, lb;
	bool enableUpperBound, enablePivoting, outputResults;
	Result result;
	uint64_t resultNumThres;
	std::chrono::time_point<std::chrono::steady_clock> startTime;
	SearchLogger* searchLogger;
}

/* Heuristic algorithm */





























/**
 * Core ordering, considering bigraph as a normal graph 
 */
void biplex::coreOrdering(const BiGraph& G) {
	Graph H(G.n[0] + G.n[1]);
	for (uint32_t u = 0; u < G.n[0]; ++u) 
		for (uint32_t v : G.nbr[0][u])
			H.addEdge(u, v+G.n[0]);

	LinearHeap<uint32_t> vHeap(H.degree);

	for (uint32_t side = 0; side <= 1; ++side) {
		order[side].resize(G.n[side]);
		ordered[side].resize(G.n[side]);
		core[side].resize(G.n[side]);
	}

	uint32_t nOrdered[2] = {0};

	while (!vHeap.empty()) {
		uint32_t u = vHeap.top(); vHeap.pop();
		for (uint32_t v : H.nbr[u]) {
			if (vHeap[v] <= vHeap[u]) continue;
			if (vHeap.inside(v)) vHeap.dec(v);
		}

		uint32_t side = 0;
		if (u >= G.n[0]) {
			side ^= 1;
			u -= G.n[0];
		} 

		ordered[side][nOrdered[side]] = u;
		order[side][u] = nOrdered[side]++;
		core[side][u] = vHeap[u];
	}

}

/**
 * Generate (alpha, beta)-core of a bigraph
 */
BiGraph biplex::coreReduction(const BiGraph& G, uint32_t alpha, uint32_t beta) {
	std::queue<uint32_t> q[2];
	std::vector<uint32_t> deg[2] = {G.degree[0], G.degree[1]};
	std::vector<bool> vis[2] = {
		std::vector<bool>(G.n[0]), 
		std::vector<bool>(G.n[1])
	};
	uint32_t coreNum[2] = {alpha, beta};
	uint32_t n[2] = {G.n[0], G.n[1]};
	for (uint32_t side = 0; side <= 1; ++side)
		for (uint32_t u = 0; u < G.n[side]; ++u)
			if (deg[side][u] < coreNum[side^1]) {
				vis[side][u] = true;
				q[side].push(u);
			}


	while (!q[0].empty() || !q[1].empty()) {
		for (uint32_t side = 0; side <= 1; ++side) {
			while (!q[side].empty()) {
				uint32_t u = q[side].front();
				q[side].pop();

				--n[side];
				for (uint32_t v : G.nbr[side][u]) {
					if (vis[side^1][v]) continue;
					if (--deg[side^1][v] < coreNum[side]) {
						vis[side^1][v] = true;
						q[side^1].push(v);
					}
				}
			}
		}
	}

	BiGraph C;

	for (uint32_t u = 0; u < G.n[0]; ++u) {
		if (vis[0][u]) continue;
		for (uint32_t v : G.nbr[0][u]) {
			if (vis[1][v]) continue;
			C.addEdgeWithLabel(G.vLabel[0][u], G.vLabel[1][v]);
		}
	}

	return C;
}

/**
 * Butterfly-based reduction
 */

BiGraph biplex::butterflyReduction(const BiGraph& G, uint32_t q, uint32_t k) {

	std::vector<uint32_t> cn(std::max(G.n[0], G.n[1])), deg[2] = {G.degree[0], G.degree[1]};
	std::vector<bool> rmv[2] = {std::vector<bool>(G.n[0]), std::vector<bool>(G.n[1])};
	std::vector<CuckooHash> rme(G.n[0]);

	coreOrdering(G);
	for (uint32_t t = 0; t < 2; ++t)
	for (uint32_t side = 0; side <= 1; ++side) {
		for (uint32_t i = 0; i < G.n[side]; ++i) {
			uint32_t u = ordered[side][i], cntv = 0;

			if (rmv[side][u]) continue;

			for (uint32_t v : G.nbr[side][u]) if (!rmv[side^1][v]) 
				for (uint32_t w : G.nbr[side^1][v]) if (!rmv[side][w])
					cn[w] = 0;
				
			for (uint32_t v : G.nbr[side][u]) if (!rmv[side^1][v]) 
				for (uint32_t w : G.nbr[side^1][v]) if (!rmv[side][w])
					++cn[w];

			for (uint32_t v : G.nbr[side][u]) if (!rmv[side^1][v]) {
				uint32_t cntw = 0;
				for (uint32_t w : G.nbr[side^1][v]) 
					if (!rmv[side][w] && cn[w] >= q-2*k) 
						++cntw;
				if (cntw >= q-k) ++cntv;
				else if (side == 0) rme[u].insert(v);
				else rme[v].insert(u);
			}

			if (cntv < q-k) {
				rmv[side][u] = true;
				for (uint32_t v : G.nbr[side][u])
					if (--deg[side^1][v] < q-k) 
						rmv[side^1][v] = true;
			}
		}
	}

	BiGraph newG;
	for (uint32_t u = 0; u < G.n[0]; ++u) if (!rmv[0][u])
		for (uint32_t v : G.nbr[0][u]) if (!rmv[1][v] && !rme[u].find(v))
			newG.addEdgeWithLabel(G.vLabel[0][u], G.vLabel[1][v]);

	return newG;

}


bool biplex::pruneCX(uint32_t u, uint32_t uSide) {

	for (uint32_t side = 0; side <= 1; ++side) 
		Q[side].clear();

	for (uint32_t v : C[uSide^1]) 
		if (degC[uSide^1][v] + (uint32_t)G.connect(uSide, u, v) + k < lb)
			Q[uSide^1].pushBack(v);

	for (uint32_t w : C[uSide])
		if (degC[uSide][w] + 2*k < lb) 
			Q[uSide].pushBack(w);

#ifdef PRUNE_WITH_QUEUE

	while (Q[0].size() != 0 || Q[1].size() != 0) {

		while (Q[uSide^1].size() != 0) {
			uint32_t v = Q[uSide^1][0]; Q[uSide^1].popBack(v);
			C[uSide^1].popBack(v);
			for (uint32_t w : G.nbr[uSide^1][v]) {
				--degC[uSide][w];
				if (C[uSide].inside(w) && degC[uSide][w] + 2*k < lb) {
					Q[uSide].pushBack(w);
				} 
				else if (w == u && degC[uSide][u] + k < lb) {
					return false;
				}
			}
		}
		while (Q[uSide].size() != 0) {
			uint32_t w = Q[uSide][0]; Q[uSide].popBack(w);
			C[uSide].popBack(w);
			for (uint32_t v : G.nbr[uSide][w]) {
				--degC[uSide^1][v];
				if (C[uSide^1].inside(v) && degC[uSide^1][v] + (uint32_t)G.connect(uSide, u, v) + k < lb) {
					Q[uSide^1].pushBack(v);
				}
			}
		}
	}


	for (uint32_t i = X[uSide].frontPos(); i < X[uSide].backPos(); ++i) {
		uint32_t v = X[uSide][i];
		if (degC[uSide][v] + 2*k < lb) {
			X[uSide].popBack(v); --i;
		}
	}
#else

	while (true) {
		bool flag = false;
		for (uint32_t i = C[uSide^1].frontPos(); i < C[uSide^1].backPos(); ++i) {
			uint32_t v = C[uSide^1][i];
			bool isConn = G.connect(uSide, u, v);
			if (degC[uSide^1][v] + (int32_t)isConn < q - k) {
				flag  = true;
				C[uSide^1].popBack(v); --i;
				for (uint32_t w : G.nbr[uSide^1][v]) --degC[uSide][w];
				if (degC[uSide][u] < q - k) return false;  
			}
		}
		if (flag) continue;
		for (uint32_t i = C[uSide].frontPos(); i < C[uSide].backPos(); ++i) {
			uint32_t v = C[uSide][i];
			if (degC[uSide][v] < q - 2*k) {
				flag = true;
				C[uSide].popBack(v); --i;
				for (uint32_t w : G.nbr[uSide][v]) --degC[uSide^1][w];
			}
		}

		for (uint32_t i = X[uSide].frontPos(); i < X[uSide].backPos(); ++i) {
			uint32_t v = X[uSide][i];
			if (degC[uSide][v] < q - 2*k) {
				flag = true;
				X[uSide].popBack(v); --i;
			}
		}

		if (!flag) break;
		
	}		
#endif
	return true;
}

bool biplex::constructSets(uint32_t u, uint32_t uSide) 
{


#ifndef DESTRUCT_SETS
	for (uint32_t side = 0; side <= 1; ++side) {
		S[side].clear();
		C[side].clear();
		X[side].clear();
		std::vector<std::vector<uint32_t>>(G.n[side]).swap(nonNbrS[side]);
		std::vector<uint32_t>(G.n[side]).swap(degC[side]);
	}
#endif



	S[uSide].pushBack(u);
	
	for (uint32_t v : G.nbr[uSide][u]) {
		C[uSide^1].pushBack(v);
		for (uint32_t w : G.nbr[uSide^1][v]) {
			if (w == u) continue;
			if (order[uSide][w] > order[uSide][u])
				C[uSide].pushBack(w);
			else
				X[uSide].pushBack(w);
		}
	}	



	for (uint32_t side = 0; side <= 1; ++side) {
		for (uint32_t v : C[side]) {
			for (uint32_t w : G.nbr[side][v])
				++degC[side^1][w];
		}
	}



	if (!pruneCX(u, uSide)) return false;


	for (uint32_t v : C[uSide])
		for (uint32_t w : G.nbr[uSide][v]) {
			if (C[uSide^1].inside(w)) continue;
			if (degC[uSide^1][w] + k >= lb) {
				C[uSide^1].pushBack(w);
				for (uint32_t x : G.nbr[uSide^1][w])
					++degC[uSide][x];
			}
		}

	if (!pruneCX(u, uSide)) return false;















	for (uint32_t v : C[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].push_back(u);

	for (uint32_t v : X[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].push_back(u);

	return true;
}

void biplex::destructSets(uint32_t u, uint32_t uSide) {
	for (uint32_t v : C[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].pop_back();

	for (uint32_t v : X[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].pop_back();

	for (uint32_t side = 0; side <= 1; ++side) {
		for (uint32_t v : C[side]) {
			for (uint32_t w : G.nbr[side][v])
				--degC[side^1][w];
		}
		S[side].clear();
		C[side].clear();
		X[side].clear();
	}
}

/**
 * Enumeration algorithm
 */
biplex::Result biplex::run(const BiGraph& Graph, uint32_t q, uint32_t k, uint64_t resultNum,
	bool flagUpperBound, bool flagPivoting, bool flagCoreReduction, bool flagButterflyReduction, bool flagOrdering,
	bool outputResults,
	SearchLogger* logger)
{
	biplex::searchLogger = logger;
	biplex::q = q;
	biplex::k = k;
	biplex::lb = q;
	biplex::resultNumThres = resultNum;
	biplex::enableUpperBound = flagUpperBound;
	biplex::enablePivoting = flagPivoting;
	biplex::outputResults = outputResults;

	printf("\nMacro Info:\n");
	PRINT_MACRO_INFO(DEBUG);
	PRINT_MACRO_INFO(CALC_DEGREE);
	PRINT_MACRO_INFO(ENUM_LARGER_DEGREE_SIDE);
	PRINT_MACRO_INFO(DESTRUCT_SETS);
	PRINT_MACRO_INFO(NO_ORDERING);
	PRINT_MACRO_INFO(PRUNE_C);
	PRINT_MACRO_INFO(CHOOSE_FIRST_VERTEX);


	if (flagCoreReduction && lb > k) {

		printf("\nStart core reduction\n");
		printf("Before: m = %d, nL = %d, nR = %d\n", Graph.m, Graph.n[0], Graph.n[1]);
		auto startTime = std::chrono::steady_clock::now();
		G = coreReduction(Graph, lb-k, lb-k);
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
		printf("After: m = %d, nL = %d, nR = %d\n", G.m, G.n[0], G.n[1]);
		printf("Time spent for core reduction: %ld ms\n", duration.count());
	}
	else 
		G = std::move(Graph);


	std::vector<std::vector<uint32_t>>(k+1, std::vector<uint32_t>(std::max(G.n[0], G.n[1]))).swap(B);
	std::vector<uint32_t>(std::max(G.n[0], G.n[1])).swap(sup);	

	for (uint32_t side = 0; side <= 1; ++side) {
		S[side].reserve(G.n[side]);
		C[side].reserve(G.n[side]);
		X[side].reserve(G.n[side]);
		Q[side].reserve(G.n[side]);
		std::vector<std::vector<uint32_t>>(G.n[side]).swap(nonNbrS[side]);
		std::vector<uint32_t>(G.n[side]).swap(degC[side]);	
	}

	result.numBranches = result.numBiPlexes = 0;

	if (flagButterflyReduction && lb > 2 * k) {
		printf("\nStart butterfly reduction\n");
		printf("Before: m = %d, nL = %d, nR = %d\n", G.m, G.n[0], G.n[1]);
		auto startTime = std::chrono::steady_clock::now();
		G = butterflyReduction(G, lb, k);
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
		printf("After: m = %d, nL = %d, nR = %d\n", G.m, G.n[0], G.n[1]);
		printf("Time spent for butterfly reduction: %ld ms\n", duration.count());
	}

	if (!flagOrdering || lb <= 2 * k) {
		for (uint32_t side = 0; side <= 1; ++side) {

			std::vector<std::vector<uint32_t>>(G.n[side]).swap(nonNbrS[side]);
			std::vector<uint32_t>(G.degree[side]).swap(degC[side]);

			for (uint32_t u = 0; u < G.n[side]; ++u) {
				nonNbrS[side][u].reserve(k);
				C[side].pushBack(u);
			}
		}
		printf("\nStart enumeration\n");
		startTime = std::chrono::steady_clock::now();
		if (searchLogger) searchLogger->start();
		branchNew(0);
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
		result.time = duration.count();
	
	} else {


		printf("\nStart core ordering\n");
		coreOrdering(G);


#ifdef ENUM_LARGER_DEGREE_SIDE
		uint32_t uSide = G.maxDeg[0] > G.maxDeg[1] ? 0 : 1;
#else
		uint32_t uSide = G.maxDeg[0] < G.maxDeg[1] ? 0 : 1;
#endif

		printf("\nStart enumeration\n");
		startTime = std::chrono::steady_clock::now();
		if (searchLogger) searchLogger->start();

		for (uint32_t i = 0; i < G.n[uSide]; ++i) {
			uint32_t u = ordered[uSide][i];
			if (constructSets(u, uSide)) {
#ifdef DEBUG
				printf("-------------------- Enumeration starts from %d --------------------\n", G.vLabel[uSide][u]);
#endif

#ifdef SMALL_K
				if (k <= SMALL_K) {
					nonNbrC.clear();
					for (uint32_t i = C[uSide^1].frontPos(); i < C[uSide^1].backPos(); ++i) {
						uint32_t v = C[uSide^1][i];
						if (!G.connect(uSide, u, v)) {
							nonNbrC.push_back(v);
							C[uSide^1].popBack(v); --i;
							for (uint32_t w : G.nbr[uSide^1][v])
								--degC[uSide][w];
							X[uSide^1].pushBack(v);
						}
					}
					branchSmallK(0, 0, u, uSide);
				} 
				else
#endif
					branchNew(0);
			}

#ifdef DESTRUCT_SETS
			destructSets(u, uSide);
#endif
		}
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
		result.time = duration.count();

	}

	

	return result;
}

uint32_t biplex::biplexUpperBound(uint32_t u, uint32_t uSide) 
{
	std::vector<uint32_t>(k+1).swap(ptrB);
	uint32_t supSum = 0;

	for (uint32_t v : S[uSide]) {

		sup[v] = k - nonNbrS[uSide][v].size();
		supSum += sup[v];
	}

	if (G.nbr[uSide][u].size() < C[uSide^1].size()) {
		for (uint32_t w : G.nbr[uSide][u]) {
			if (!C[uSide^1].inside(w)) continue;
			uint32_t s = nonNbrS[uSide^1][w].size();
			B[s][ptrB[s]++] = w;
		}
	}
	else {
		for (uint32_t w : C[uSide^1]) {
			if (!G.connect(uSide, u, w)) continue;
			uint32_t s = nonNbrS[uSide^1][w].size();
			B[s][ptrB[s]++] = w;
		}
	}

	uint32_t ub = S[uSide^1].size() + k - nonNbrS[uSide][u].size();

	for (uint32_t i = 0; i <= k && i <= supSum; ++i) {
		for (uint32_t j = 0; j < ptrB[i] && i <= supSum; ++j) {

			uint32_t v = B[i][j], x = -1;

			for (uint32_t w : nonNbrS[uSide^1][v])
				if (x == -1u || sup[w] < sup[x])
					x = w;

			if (x == -1u || sup[x] > 0) {
				if (x != -1u) --sup[x];
				supSum -= i;
				++ub;
			}
		}
	}

	return ub;
}


void biplex::printSet(VertexSet* V, const std::string& name) {
	std::vector<uint32_t> sortedV[2];
	for (uint32_t i = 0; i <= 1; ++i) {
		sortedV[i].reserve(V[i].size());
		for (uint32_t v : V[i])
			sortedV[i].push_back(G.vLabel[i][v]);
		std::sort(sortedV[i].begin(), sortedV[i].end());
	}
	printf("%sL: ", name.c_str());
	for (uint32_t v : sortedV[0]) printf("%d, ", v);
	printf("\n%sR: ", name.c_str());
	for (uint32_t v : sortedV[1]) printf("%d, ", v);
	printf("\n");
}

void biplex::printSets() {
	printSet(S, "S");
	printSet(C, "C");
	printSet(X, "X");

	VertexSet SC[2] = {VertexSet(G.n[0]), VertexSet(G.n[1])};
	for (uint32_t i = 0; i <= 1; ++i) {
		for (uint32_t v : S[i])
			SC[i].pushBack(v);
		for (uint32_t v : C[i])
			SC[i].pushBack(v);
	}

	printSet(SC, "S&C");

}

void biplex::printResult(Result& result) {
	printf("Number of branches: %" PRIu64 ", bipartite: %" PRIu64 ", pivoting: %" PRIu64 "\n",
			result.numBranches,
			result.numBipartiteBranches,
			result.numPivotingBranches);
	printf("Number of biplexes: %" PRIu64 "\n", result.numBiPlexes);
	printf("Branch time: %" PRIu64 " ms\n", result.time);
/*
	std::cout << "Number of branches: " << result.numBranches << \
		", bipartite: " << result.numBipartiteBranches << \
		", pivoting: " << result.numPivotingBranches << std::endl;
	std::cout << "Number of biplexes: " << result.numBiPlexes << std::endl;
	std::cout << "Branch time: " << result.time << "ms" << std::endl;
*/
}

void biplex::branchSmallK(uint32_t dep, uint32_t begin, uint32_t u, uint32_t uSide)
{

	branchNew(0);

	if ((uint32_t)dep == k) return;

	std::pair<uint32_t, uint32_t> oldPosC, oldPosX;

	for (uint32_t i = begin; i < nonNbrC.size(); ++i) {
		uint32_t& v = nonNbrC[i];

		biplexUpdate(v, uSide^1, oldPosC, oldPosX);
		biplexUpdateDeg(v, uSide^1, oldPosC);

		branchSmallK(dep+1, i+1, u, uSide);

		biplexRestoreDeg(v, uSide^1, oldPosC);
		biplexRestore(v, uSide^1, oldPosC, oldPosX);

	}

}

uint32_t biplex::calcDegree(VertexSet* V, uint32_t u, uint32_t uSide)
{
	uint32_t deg = 0;
	if (G.nbr[uSide][u].size() < V[uSide^1].size()) {
		for (uint32_t v : G.nbr[uSide][u])
			if (V[uSide^1].inside(v))
				++deg;
	}
	else {
		for (uint32_t v : V[uSide^1])
			if (G.connect(uSide, u, v))
				++deg;
	}
	return deg;	
}



void biplex::biplexPruneC(std::pair<uint32_t, uint32_t>& oldPosC) {

	oldPosC = std::make_pair(C[0].frontPos(), C[1].frontPos());

#ifdef PRUNE_ONCE	

	for (uint32_t side = 0; side <= 1; ++side) {
		for (uint32_t u : C[side]) {
#ifdef CALC_DEGREE
			degC[side][u] = calcDegree(C, u, side);
#endif
			int32_t degCu = degC[side][u];
			if (degCu + (S[side^1].size() - nonNbrS[side][u].size()) + k < lb) {
				C[side].popFront(u);
				for (uint32_t v : G.nbr[side][u])
					--degC[side^1][v];
			}
		}
	}

#else

#ifdef PRUNE_WITH_QUEUE

	for (uint32_t side = 0; side <= 1; ++side) {
		Q[side].clear();
		for (uint32_t u : C[side]) {
#ifdef CALC_DEGREE
			degC[side][u] = calcDegree(C, u, side);
#endif
			int32_t degCu = degC[side][u];
			if (degCu + (S[side^1].size() - nonNbrS[side][u].size()) + k < lb) {
				Q[side].pushBack(u);
			}
		}
	}

	while (Q[0].size() != 0 || Q[1].size() != 0) {
		for (uint32_t side = 0; side <= 1; ++side) {
			while (Q[side].size() != 0) {
				uint32_t u = Q[side][0];
				Q[side].popBack(u);
				C[side].popFront(u);
				for (uint32_t v : G.nbr[side][u]) {
					--degC[side^1][v];
					if (C[side^1].inside(v) && degC[side^1][v] + (S[side].size() - nonNbrS[side^1][v].size()) + k < lb)
				 		Q[side^1].pushBack(v);
				}
			}
		}
	}

#else
	while (true) {
		bool flag = false;
		for (uint32_t side = 0; side <= 1; ++side) {
			for (uint32_t u : C[side]) {
#ifdef CALC_DEGREE
				degC[side][u] = calcDegree(C, u, side);
#endif
				int32_t degCu = degC[side][u];
				if (degCu + (S[side^1].size() - nonNbrS[side][u].size()) + k < lb) {
					flag = true;
					C[side].popFront(u);
					for (uint32_t v : G.nbr[side][u])
						--degC[side^1][v];
				}
			}
		}
		if (!flag) break;
	}
#endif

#endif

}

void biplex::biplexRestorePruneC(std::pair<uint32_t, uint32_t>& oldPosC) {
	biplexRestoreDegC(oldPosC);
	C[0].restore(oldPosC.first);
	C[1].restore(oldPosC.second);
}

/**
 * The main algorithm.
 * */
void biplex::branchNew(uint32_t dep) 
{

	++result.numBranches;

	if (result.numBiPlexes >= resultNumThres) return;

#ifdef DEBUG
	printf("\n---------- dep = %d ----------\n", dep);
	printSets();
#endif

	std::pair<uint32_t, uint32_t> oldPosC, oldPosX, pruneOldPosC;
	

#ifdef PRUNE_C
	biplexPruneC(pruneOldPosC);	
#endif



	if ((S[0].size() + C[0].size() < lb) || (S[1].size() + C[1].size() < lb)) {
#ifdef PRUNE_C
		biplexRestorePruneC(pruneOldPosC);
#endif
		return;
	}

	if (C[0].size() == 0 && C[1].size() == 0) {
		if (X[0].size() == 0 && X[1].size() == 0) {
			if (S[0].size() >= lb && S[1].size() >= lb) {
				++result.numBiPlexes;
				if (outputResults) {
					printf("Biplex No.%" PRIu64 ":\n", result.numBiPlexes);
					printSet(S, "  S");
				}
				if (searchLogger) searchLogger->check(result.numBiPlexes);
				if (result.numBiPlexes == resultNumThres) {
					printf("Done!\n");
					auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
					result.time = duration.count();
					printResult(result);
					exit(0);
				}
#ifdef DEBUG
				printf("*** find: No.%d\n", result.numBiPlexes);
#endif
			}
		}

#ifdef PRUNE_C
		biplexRestorePruneC(pruneOldPosC);
#endif
		return;
	}




	int32_t inf = G.n[0] + G.n[1];
	uint32_t Sp[2] = {-1u, -1u};
	int32_t nonDegPInS[2] = {-inf, -inf};
	uint32_t Cp[2] = {-1u, -1u};
	int32_t nonDegPInC[2] = {-inf, -inf};


	for (uint32_t side = 0; side <= 1; ++side) {
		for (uint32_t u : S[side]) {
#ifdef CALC_DEGREE
			degC[side][u] = calcDegree(C, u, side);
#endif
			int32_t nonDegU = C[side^1].size() - degC[side][u] + nonNbrS[side][u].size();
			if (nonDegU > nonDegPInS[side]) {
				nonDegPInS[side] = nonDegU;
				Sp[side] = u;
			}
		}


		if ((int32_t)(S[side^1].size() + C[side^1].size()) - nonDegPInS[side] < (int32_t)(lb - k))  {
#ifdef PRUNE_C
			biplexRestorePruneC(pruneOldPosC);
#endif
			return;
		}
	}

	if (!enablePivoting) {

		for (uint32_t side = 0; side <= 1; ++side) {
			for (uint32_t u : C[side]) {
#ifdef CALC_DEGREE
				degC[side][u] = calcDegree(C, u, side);
#endif
				int32_t nonDegU = C[side^1].size() - degC[side][u] + nonNbrS[side][u].size();
				if (nonDegU > nonDegPInC[side]) {
					nonDegPInC[side] = nonDegU;
					Cp[side] = u;
				}
			}
		}


		if (std::max(nonDegPInC[0], nonDegPInS[0]) <= (int32_t)k &&
			std::max(nonDegPInC[1], nonDegPInS[1]) <= (int32_t)k &&
			X[0].size() == 0 && X[1].size() == 0) {
			++result.numBiPlexes;
			if (outputResults) {
				printf("Biplex No.%" PRIu64 ":\n", result.numBiPlexes);
				VertexSet SC[2] = {VertexSet(G.n[0]), VertexSet(G.n[1])};
				for (uint32_t i = 0; i <= 1; ++i) {
					for (uint32_t v : S[i]) SC[i].pushBack(v);
					for (uint32_t v : C[i]) SC[i].pushBack(v);
				}
				printSet(SC, "  S∪C");
			}
			if (searchLogger) searchLogger->check(result.numBiPlexes);
#ifdef DEBUG
			printf("*** find: No.%d\n", result.numBiPlexes);
#endif
#ifdef PRUNE_C
			biplexRestorePruneC(pruneOldPosC);
#endif
			return;
		}
	}	
	




	if (!enablePivoting || (nonDegPInS[0] > (int32_t)k || nonDegPInS[1] > (int32_t)k)) {

		uint32_t side = 0, u = -1;

		if (enablePivoting) {

			side = nonDegPInS[0] > nonDegPInS[1] ? 0 : 1;
			u = -1;
			int32_t deg_u = G.n[side], nonDegSu = -1;


			for (uint32_t v : C[side^1]) {

				int32_t nonDegSv = nonNbrS[side^1][v].size();
				int32_t deg_v = degC[side^1][v] + S[side].size() - nonDegSv;
				if (((deg_v < deg_u) || (deg_v == deg_u && nonDegSv > nonDegSu)) && !G.connect(side, Sp[side], v))  {
					u = v;
					deg_u = deg_v;
					nonDegSu = nonDegSv;
#ifdef CHOOSE_FIRST_VERTEX
					break;
#endif
				}
			}
		}

		else {
			side = S[1].size() + C[1].size() - nonDegPInC[0] < S[0].size() + C[0].size() - nonDegPInC[1] ? 1 : 0;
			u = Cp[side^1];
		}

#ifdef DEBUG
		printf("Bipartite: %d%c\n", G.vLabel[side^1][u], side^1 ? 'R' : 'L');
#endif


		assert(u != -1u);
		biplexUpdate(u, side^1, oldPosC, oldPosX);
		if (S[0].size() + C[0].size() >= lb && S[1].size() + C[1].size() >= lb) {
			if (!enableUpperBound || biplexUpperBound(u, side^1) >= lb) {
				biplexUpdateDeg(u, side^1, oldPosC);
				++result.numBipartiteBranches;				
				branchNew(dep + 1);
				biplexRestoreDeg(u, side^1, oldPosC);
			}
		}
		biplexRestore(u, side^1, oldPosC, oldPosX);
		if (S[0].size() + C[0].size() >= lb && S[1].size() + C[1].size() >= lb) {
			++result.numBipartiteBranches;				
			branchNew(dep + 1);
		}
		biplexBacktrack(u, side^1);

	}




	else {


		uint32_t p[2] = {-1u, -1u};
		int32_t degCp[2] = {-1, -1};
		
		for (uint32_t side = 0; side <= 1; ++side) {
			for (uint32_t u : C[side]) {
#ifdef CALC_DEGREE
				degC[side][u] = calcDegree(C, u, side);
#endif
				int32_t degCu = degC[side][u];
				if (degCu > degCp[side]) {
					p[side] = u;
					degCp[side] = degCu;
#ifdef CHOOSE_FIRST_VERTEX
					break;
#endif
				}
			}
#ifndef CHOOSE_FIRST_VERTEX
			for (uint32_t u : X[side]) {
#ifdef CALC_DEGREE
				degC[side][u] = calcDegree(C, u, side);
#endif
				int32_t degCu = degC[side][u];
				if (degCu > degCp[side]) {

					bool flag = false;

					for (uint32_t v : nonNbrS[side][u]) {

						if (C[side].size() - degC[side^1][v] + nonNbrS[side^1][v].size() >= k) {
							flag = true;
							break;
						}
					}

					if (!flag) {
						p[side] = u;
						degCp[side] = degCu;
					}
				}
			}
#endif
		}

		uint32_t pSide = C[1].size() - degCp[0] < C[0].size() - degCp[1] ? 0 : 1;
		uint32_t pivot = p[pSide];
	
#ifdef DEBUG
		printf("Pivoting: %d%c\n", G.vLabel[pSide][pivot], pSide ? 'R' : 'L');
#endif


		if (cand.size() <= dep) cand.resize(dep * 2);
		std::vector<uint32_t>& Cand = cand[dep];
		
		Cand.clear();
		
		Cand.reserve(C[pSide^1].size() - degCp[pSide]);

		for (uint32_t u : C[pSide^1])
			if (!G.connect(pSide, pivot, u))
				Cand.push_back(u);



		for (uint32_t u : Cand) {

			biplexUpdate(u, pSide^1, oldPosC, oldPosX);
			if (S[0].size() + C[0].size() >= lb && S[1].size() + C[1].size() >= lb) {
				if (!enableUpperBound || biplexUpperBound(u, pSide^1) >= lb) {
					++result.numPivotingBranches;
					biplexUpdateDeg(u, pSide^1, oldPosC);
					branchNew(dep + 1);
					biplexRestoreDeg(u, pSide^1, oldPosC);
				}
			}
			biplexRestore(u, pSide^1, oldPosC, oldPosX);
		}
		

		if (C[pSide].inside(pivot)) {
			biplexUpdate(pivot, pSide, oldPosC, oldPosX);
			if (S[0].size() + C[0].size() >= lb && S[1].size() + C[1].size() >= lb) {
				if (!enableUpperBound || biplexUpperBound(pivot, pSide) >= lb) {
					++result.numPivotingBranches;
					biplexUpdateDeg(pivot, pSide, oldPosC);
					branchNew(dep + 1);
					biplexRestoreDeg(pivot, pSide, oldPosC);
				}
			}
			biplexRestore(pivot, pSide, oldPosC, oldPosX);
			biplexBacktrack(pivot, pSide);
		}


		for (uint32_t u : Cand) {
			biplexBacktrack(u, pSide^1);
		}	
	}

#ifdef PRUNE_C
	biplexRestorePruneC(pruneOldPosC);
#endif

#ifdef DEBUG
	printf("\n>>>>>>>>>> dep = %d <<<<<<<<<<\n", dep);
	printSets();	
#endif

}

void biplex::biplexUpdateDegC(std::pair<uint32_t, uint32_t>& oldPosC) {
#ifndef CALC_DEGREE

	for (uint32_t side = 0; side <= 1; ++side) {
		uint32_t oldPos = side == 0 ? oldPosC.first : oldPosC.second;
		for (uint32_t i = oldPos; i < C[side].frontPos(); ++i) {
			uint32_t v = C[side][i];
			for (uint32_t w : G.nbr[side][v])
				--degC[side^1][w];
		}
	}
#endif
}



void biplex::biplexUpdate(uint32_t u, uint32_t uSide, 
	std::pair<uint32_t, uint32_t>& oldPosC, 
	std::pair<uint32_t, uint32_t>& oldPosX) 
{

	biplexUpdateCX(u, uSide, C, oldPosC);
	biplexUpdateCX(u, uSide, X, oldPosX);


	S[uSide].pushBack(u);
}

void biplex::biplexUpdateDeg(uint32_t u, uint32_t uSide, std::pair<uint32_t, uint32_t>& oldPosC) 
{


	for (uint32_t v : S[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].push_back(u);

	for (uint32_t v : C[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].push_back(u);

	for (uint32_t v : X[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].push_back(u);

	biplexUpdateDegC(oldPosC);

}


void biplex::biplexUpdateCX(uint32_t u, uint32_t uSide, VertexSet* V, std::pair<uint32_t, uint32_t>& oldPos) 
{
	oldPos = std::make_pair(V[0].frontPos(), V[1].frontPos());
	V[uSide].popFront(u);

	bool flagNonNbrSu = nonNbrS[uSide][u].size() >= k;
	for (uint32_t v : V[uSide^1]) {
		if (G.connect(uSide, u, v)) continue;
		if (flagNonNbrSu || nonNbrS[uSide^1][v].size() >= k) {
			V[uSide^1].popFront(v);
		}
	}

	for (uint32_t v : nonNbrS[uSide][u]) {
		if (nonNbrS[uSide^1][v].size() == k-1) {
			if (V[uSide].size() < G.nbr[uSide^1][v].size()) {
				for (uint32_t w : V[uSide]) {
					if (!G.connect(uSide^1, v, w)) {
						V[uSide].popFront(w);
					}
				}
			}
			else {
				uint32_t newFrontPos = V[uSide].backPos();
				for (uint32_t w : G.nbr[uSide^1][v])
					if (V[uSide].inside(w))
						V[uSide].swapByVal(w, V[uSide][--newFrontPos]);
				V[uSide].restore(newFrontPos);
			}
		}
	}
}

void biplex::biplexRestoreDegC(std::pair<uint32_t, uint32_t>& oldPosC) {
#ifndef CALC_DEGREE

	for (uint32_t side = 0; side <= 1; ++side) {
		uint32_t oldPos = side == 0 ? oldPosC.first : oldPosC.second;
		for (uint32_t i = oldPos; i < C[side].frontPos(); ++i) {
			uint32_t v = C[side][i];
			for (uint32_t w : G.nbr[side][v])
				++degC[side^1][w];

		}
	}

#endif
}

void biplex::biplexRestoreDeg(uint32_t u, uint32_t uSide, std::pair<uint32_t, uint32_t>& oldPosC) 
{


	for (uint32_t v : S[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].pop_back();

	for (uint32_t v : C[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].pop_back();

	for (uint32_t v : X[uSide^1])
		if (!G.connect(uSide, u, v))
			nonNbrS[uSide^1][v].pop_back();


	biplexRestoreDegC(oldPosC);

}

void biplex::biplexRestore(uint32_t u, uint32_t uSide, std::pair<uint32_t, uint32_t>& oldPosC, std::pair<uint32_t, uint32_t>& oldPosX) 
{	


	C[0].restore(oldPosC.first);
	C[1].restore(oldPosC.second);


	X[0].restore(oldPosX.first);
	X[1].restore(oldPosX.second);

#ifndef CALC_DEGREE
	if (C[uSide].inside(u)) {
		for (uint32_t v : G.nbr[uSide][u])
			--degC[uSide^1][v];
	}
#endif



	S[uSide].popBack(u);
	C[uSide].popFront(u);
	X[uSide].pushBack(u);

}


void biplex::biplexBacktrack(uint32_t u, uint32_t uSide)
{

#ifndef CALC_DEGREE
	if (!C[uSide].inside(u)) {
		for (uint32_t v : G.nbr[uSide][u])
			++degC[uSide^1][v];
	}
#endif

	C[uSide].pushFront(u);
	X[uSide].popBack(u);
}
