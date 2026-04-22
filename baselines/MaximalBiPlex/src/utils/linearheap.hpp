#ifndef LINEAR_HEAP
#define LINEAR_HEAP

#include <vector>
#include <algorithm>
#include <cassert>

template<typename T>
class LinearHeap {
	std::vector<uint32_t> rank, rankId, bin;
	std::vector<T> keys;
	uint32_t ptr;
public:

	LinearHeap(uint32_t n, T maxValue): rank(n+1), rankId(n+1), bin(maxValue+2), keys(n+1),ptr(0) {
		for (uint32_t i = 0; i < n; ++i) {
			rank[i] = bin[0]++;
			rankId[rank[i]] = i;
		}
	}

	LinearHeap(const std::vector<T>& nums): rank(nums.size() + 1), rankId(nums.size() + 1), keys(nums), ptr(0) {
		T maxValue = 0;
		if (!keys.empty())
			maxValue = *std::max_element(keys.begin(), keys.end());
		bin.resize(maxValue + 2);
		for (T v : keys) ++bin[v+1];
		for (T v = 1; v <= maxValue; ++v) bin[v] += bin[v-1];
		for (uint32_t i = 0; i < keys.size(); ++i) {
			rank[i] = bin[keys[i]]++;
			rankId[rank[i]] = i;
		}
	}

	void inc(uint32_t id) {
		if (!inside(id)) return;
		assert(keys[id] < bin.size());
		uint32_t otherId = rankId[--bin[keys[id]]];
		rank[otherId] = rank[id];
		rank[id] = bin[keys[id]++];
		rankId[rank[id]] = id;
		rankId[rank[otherId]] = otherId;
	}
	
	void dec(uint32_t id) {
		if (!inside(id)) return;
		assert(keys[id] > 0);
		if (keys[rankId[ptr]] == keys[id])
			bin[keys[id]-1] = ptr;
		uint32_t otherId = rankId[bin[--keys[id]]];
		rank[otherId] = rank[id];
		rank[id] = bin[keys[id]]++;
		rankId[rank[id]] = id;
		rankId[rank[otherId]] = otherId;

	}

	uint32_t top() {
		return rankId[ptr];
	}

	void pop() {
		++ptr;
	}

	bool empty() {
		return ptr >= keys.size();
	}

	bool inside(uint32_t id) {
		return rank[id] >= ptr;
	}

	uint32_t size() {
		return keys.size() - ptr;
	}

	T operator[] (uint32_t id) {
		return keys[id];
	}

};

#endif