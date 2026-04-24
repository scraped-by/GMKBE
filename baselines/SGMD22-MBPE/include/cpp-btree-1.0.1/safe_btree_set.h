




























#ifndef UTIL_BTREE_SAFE_BTREE_SET_H__
#define UTIL_BTREE_SAFE_BTREE_SET_H__

#include <functional>
#include <memory>

#include "btree_container.h"
#include "btree_set.h"
#include "safe_btree.h"

namespace btree {


template <typename Key,
          typename Compare = std::less<Key>,
          typename Alloc = std::allocator<Key>,
          int TargetNodeSize = 256>
class safe_btree_set : public btree_unique_container<
  safe_btree<btree_set_params<Key, Compare, Alloc, TargetNodeSize> > > {

  typedef safe_btree_set<Key, Compare, Alloc, TargetNodeSize> self_type;
  typedef btree_set_params<Key, Compare, Alloc, TargetNodeSize> params_type;
  typedef safe_btree<params_type> btree_type;
  typedef btree_unique_container<btree_type> super_type;

 public:
  typedef typename btree_type::key_compare key_compare;
  typedef typename btree_type::allocator_type allocator_type;

 public:

  safe_btree_set(const key_compare &comp = key_compare(),
                 const allocator_type &alloc = allocator_type())
      : super_type(comp, alloc) {
  }


  safe_btree_set(const self_type &x)
      : super_type(x) {
  }


  template <class InputIterator>
  safe_btree_set(InputIterator b, InputIterator e,
                 const key_compare &comp = key_compare(),
                 const allocator_type &alloc = allocator_type())
      : super_type(b, e, comp, alloc) {
  }
};

template <typename K, typename C, typename A, int N>
inline void swap(safe_btree_set<K, C, A, N> &x,
                 safe_btree_set<K, C, A, N> &y) {
  x.swap(y);
}

}

#endif
