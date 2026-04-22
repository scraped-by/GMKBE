















#ifndef IMB_GPU2_INCLUDE_FOR_TEST_H
#define IMB_GPU2_INCLUDE_FOR_TEST_H
#include <set>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>

template<typename T>
std::vector<T> generateRandomNumbers(int n, int lower_bound, int upper_bound, bool is_sort = false, bool is_unique = false) {
	assert(n > 0);
	assert((not is_unique) or upper_bound >= n);
	std::vector<T> numbers;
	numbers.reserve(n);
	std::set<T> unique_numbers;
	std::random_device rd;  
	std::mt19937 gen(rd()); 
	std::uniform_int_distribution<> distrib(lower_bound, upper_bound);
	while(numbers.size() < n){
		int num = distrib(gen);
		if (not is_unique or unique_numbers.insert(static_cast<T>(num)).second)
			numbers.emplace_back(static_cast<T>(num));
	}
	if(is_sort)
		std::sort(numbers.begin(), numbers.end());
	return numbers;
}




















#define TRACK_NODE_VALUE(g_state_obj, current_node, node_to_track)                                              \
    do {                                                                                                        \
        if ((current_node) == (node_to_track)) {                                                                \
            auto val = (g_state_obj).get_value(current_node);                                                   \
            printf("[TRACKING] File: %s, Line: %d, Object: G_State, Node: %u, Value: %d\n", __FILE__, __LINE__, \
                (unsigned int) (current_node), val);                                                            \
        }                                                                                                       \
    } while (0)

#define TRACK_NODE_VALUE2(g_obj, current_node, node_to_track)                                                      \
    do {                                                                                                           \
        if ((current_node) == (node_to_track)) {                                                                   \
            auto val = g_obj[current_node];                                                                        \
                                                                          \
            printf("[TRACKING] File: %s, Line: %d, Object: %s, Node: %u, Value: %d\n", __FILE__, __LINE__, #g_obj, \
                (unsigned int) (current_node), val);                                                               \
        }                                                                                                          \
    } while (0)


#endif 
