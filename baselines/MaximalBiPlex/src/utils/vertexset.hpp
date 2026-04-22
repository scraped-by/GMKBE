#ifndef VERTEX_SET
#define VERTEX_SET

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>

class VertexSet {
	uint32_t lp, rp, capacity;
	uint64_t hash;
	std::vector<uint32_t> s, pos;

  void swapByPos(uint32_t i, uint32_t j) {
		std::swap(s[i], s[j]);
		pos[s[i]] = i;
		pos[s[j]] = j;
	}

public:
	VertexSet(uint32_t capacity) {
		reserve(capacity);
		clear();
	}

	VertexSet() {
		clear();
	}
 
  void swapByVal(uint32_t u, uint32_t v) {
    swapByPos(pos[u], pos[v]);
  }

	void reserve(uint32_t capacity) {
		this->capacity = capacity;
		s.resize(capacity);
		pos.resize(capacity);
		for (uint32_t i = 0; i < capacity; ++i)
			s[i] = pos[i] = i;
	}

	void pushFront(uint32_t v) {
		assert(lp > 0);
		if (pos[v] < lp) swapByPos(pos[v], --lp);
	}

	void pushBack(uint32_t v) {
		assert(rp < capacity);
		if (pos[v] >= rp) swapByPos(pos[v], rp++);
	}

	void popFront(uint32_t v) {
		if (inside(v)) swapByPos(pos[v], lp++);
	}

	void popBack(uint32_t v) {
		if (inside(v)) swapByPos(pos[v], --rp);
	}


	bool inside(uint32_t v) {
		return pos[v] >= lp && pos[v] < rp;
	}

  	uint32_t size() {
		return rp - lp;
	}

	uint32_t frontPos() {
		return lp;
	}

	uint32_t backPos() {
		return rp;
	}

	void restore(uint32_t pos) {
		lp = pos;
	}

	void clear() {
		lp = rp = hash = 0;
	}

	uint32_t operator [] (uint32_t index) {
		return s[index];
	}

	uint32_t* begin() {
		return s.data() + lp;
	}

	uint32_t* end() {
		return s.data() + rp;
	}


};

#endif
