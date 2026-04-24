


































































































#ifndef UTIL_BTREE_BTREE_H__
#define UTIL_BTREE_BTREE_H__

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <cstring>
#include <sys/types.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <new>
#include <ostream>
#include <string>
#include <utility>

#ifndef NDEBUG
#define NDEBUG 1
#endif

namespace btree {








template <typename T>
inline void btree_swap_helper(T &a, T &b) {
  using std::swap;
  swap(a, b);
}


template<bool cond, typename A, typename B>
struct if_{
  typedef A type;
};

template<typename A, typename B>
struct if_<false, A, B> {
  typedef B type;
};


typedef char small_;

struct big_ {
  char dummy[2];
};


template <bool>
struct CompileAssert {
};

#define COMPILE_ASSERT(expr, msg) \
  typedef CompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]













struct btree_key_compare_to_tag {
};



template <typename Compare>
struct btree_is_key_compare_to
    : public std::is_convertible<Compare, btree_key_compare_to_tag> {
};









template <typename Compare>
struct btree_key_compare_to_adapter : Compare {
  btree_key_compare_to_adapter() { }
  btree_key_compare_to_adapter(const Compare &c) : Compare(c) { }
  btree_key_compare_to_adapter(const btree_key_compare_to_adapter<Compare> &c)
      : Compare(c) {
  }
};

template <>
struct btree_key_compare_to_adapter<std::less<std::string> >
    : public btree_key_compare_to_tag {
  btree_key_compare_to_adapter() {}
  btree_key_compare_to_adapter(const std::less<std::string>&) {}
  btree_key_compare_to_adapter(
      const btree_key_compare_to_adapter<std::less<std::string> >&) {}
  int operator()(const std::string &a, const std::string &b) const {
    return a.compare(b);
  }
};

template <>
struct btree_key_compare_to_adapter<std::greater<std::string> >
    : public btree_key_compare_to_tag {
  btree_key_compare_to_adapter() {}
  btree_key_compare_to_adapter(const std::greater<std::string>&) {}
  btree_key_compare_to_adapter(
      const btree_key_compare_to_adapter<std::greater<std::string> >&) {}
  int operator()(const std::string &a, const std::string &b) const {
    return b.compare(a);
  }
};




template <typename Key, typename Compare, bool HaveCompareTo>
struct btree_key_comparer {
  btree_key_comparer() {}
  btree_key_comparer(Compare c) : comp(c) {}
  static bool bool_compare(const Compare &comp, const Key &x, const Key &y) {
    return comp(x, y);
  }
  bool operator()(const Key &x, const Key &y) const {
    return bool_compare(comp, x, y);
  }
  Compare comp;
};




template <typename Key, typename Compare>
struct btree_key_comparer<Key, Compare, true> {
  btree_key_comparer() {}
  btree_key_comparer(Compare c) : comp(c) {}
  static bool bool_compare(const Compare &comp, const Key &x, const Key &y) {
    return comp(x, y) < 0;
  }
  bool operator()(const Key &x, const Key &y) const {
    return bool_compare(comp, x, y);
  }
  Compare comp;
};





template <typename Key, typename Compare>
static bool btree_compare_keys(
    const Compare &comp, const Key &x, const Key &y) {
  typedef btree_key_comparer<Key, Compare,
      btree_is_key_compare_to<Compare>::value> key_comparer;
  return key_comparer::bool_compare(comp, x, y);
}

template <typename Key, typename Compare,
          typename Alloc, int TargetNodeSize, int ValueSize>
struct btree_common_params {



  typedef typename if_<
    btree_is_key_compare_to<Compare>::value,
    Compare, btree_key_compare_to_adapter<Compare> >::type key_compare;


  typedef btree_is_key_compare_to<key_compare> is_key_compare_to;

  typedef Alloc allocator_type;
  typedef Key key_type;
  typedef ssize_t size_type;
  typedef ptrdiff_t difference_type;

  enum {
    kTargetNodeSize = TargetNodeSize,



    kNodeValueSpace = TargetNodeSize - 2 * sizeof(void*),
  };



  typedef typename if_<
    (kNodeValueSpace / ValueSize) >= 256,
    uint16_t,
    uint8_t>::type node_count_type;
};


template <typename Key, typename Data, typename Compare,
          typename Alloc, int TargetNodeSize>
struct btree_map_params
    : public btree_common_params<Key, Compare, Alloc, TargetNodeSize,
                                 sizeof(Key) + sizeof(Data)> {
  typedef Data data_type;
  typedef Data mapped_type;
  typedef std::pair<const Key, data_type> value_type;
  typedef std::pair<Key, data_type> mutable_value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  enum {
    kValueSize = sizeof(Key) + sizeof(data_type),
  };

  static const Key& key(const value_type &x) { return x.first; }
  static const Key& key(const mutable_value_type &x) { return x.first; }
  static void swap(mutable_value_type *a, mutable_value_type *b) {
    btree_swap_helper(a->first, b->first);
    btree_swap_helper(a->second, b->second);
  }
};


template <typename Key, typename Compare, typename Alloc, int TargetNodeSize>
struct btree_set_params
    : public btree_common_params<Key, Compare, Alloc, TargetNodeSize,
                                 sizeof(Key)> {
  typedef std::false_type data_type;
  typedef std::false_type mapped_type;
  typedef Key value_type;
  typedef value_type mutable_value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  enum {
    kValueSize = sizeof(Key),
  };

  static const Key& key(const value_type &x) { return x; }
  static void swap(mutable_value_type *a, mutable_value_type *b) {
    btree_swap_helper<mutable_value_type>(*a, *b);
  }
};



template <typename Key, typename Compare>
struct btree_upper_bound_adapter : public Compare {
  btree_upper_bound_adapter(Compare c) : Compare(c) {}
  bool operator()(const Key &a, const Key &b) const {
    return !static_cast<const Compare&>(*this)(b, a);
  }
};

template <typename Key, typename CompareTo>
struct btree_upper_bound_compare_to_adapter : public CompareTo {
  btree_upper_bound_compare_to_adapter(CompareTo c) : CompareTo(c) {}
  int operator()(const Key &a, const Key &b) const {
    return static_cast<const CompareTo&>(*this)(b, a);
  }
};


template <typename K, typename N, typename Compare>
struct btree_linear_search_plain_compare {
  static int lower_bound(const K &k, const N &n, Compare comp)  {
    return n.linear_search_plain_compare(k, 0, n.count(), comp);
  }
  static int upper_bound(const K &k, const N &n, Compare comp)  {
    typedef btree_upper_bound_adapter<K, Compare> upper_compare;
    return n.linear_search_plain_compare(k, 0, n.count(), upper_compare(comp));
  }
};


template <typename K, typename N, typename CompareTo>
struct btree_linear_search_compare_to {
  static int lower_bound(const K &k, const N &n, CompareTo comp)  {
    return n.linear_search_compare_to(k, 0, n.count(), comp);
  }
  static int upper_bound(const K &k, const N &n, CompareTo comp)  {
    typedef btree_upper_bound_adapter<K,
        btree_key_comparer<K, CompareTo, true> > upper_compare;
    return n.linear_search_plain_compare(k, 0, n.count(), upper_compare(comp));
  }
};


template <typename K, typename N, typename Compare>
struct btree_binary_search_plain_compare {
  static int lower_bound(const K &k, const N &n, Compare comp)  {
    return n.binary_search_plain_compare(k, 0, n.count(), comp);
  }
  static int upper_bound(const K &k, const N &n, Compare comp)  {
    typedef btree_upper_bound_adapter<K, Compare> upper_compare;
    return n.binary_search_plain_compare(k, 0, n.count(), upper_compare(comp));
  }
};


template <typename K, typename N, typename CompareTo>
struct btree_binary_search_compare_to {
  static int lower_bound(const K &k, const N &n, CompareTo comp)  {
    return n.binary_search_compare_to(k, 0, n.count(), CompareTo());
  }
  static int upper_bound(const K &k, const N &n, CompareTo comp)  {
    typedef btree_upper_bound_adapter<K,
        btree_key_comparer<K, CompareTo, true> > upper_compare;
    return n.linear_search_plain_compare(k, 0, n.count(), upper_compare(comp));
  }
};




template <typename Params>
class btree_node {
 public:
  typedef Params params_type;
  typedef btree_node<Params> self_type;
  typedef typename Params::key_type key_type;
  typedef typename Params::data_type data_type;
  typedef typename Params::value_type value_type;
  typedef typename Params::mutable_value_type mutable_value_type;
  typedef typename Params::pointer pointer;
  typedef typename Params::const_pointer const_pointer;
  typedef typename Params::reference reference;
  typedef typename Params::const_reference const_reference;
  typedef typename Params::key_compare key_compare;
  typedef typename Params::size_type size_type;
  typedef typename Params::difference_type difference_type;

  typedef btree_linear_search_plain_compare<
    key_type, self_type, key_compare> linear_search_plain_compare_type;
  typedef btree_linear_search_compare_to<
    key_type, self_type, key_compare> linear_search_compare_to_type;
  typedef btree_binary_search_plain_compare<
    key_type, self_type, key_compare> binary_search_plain_compare_type;
  typedef btree_binary_search_compare_to<
    key_type, self_type, key_compare> binary_search_compare_to_type;


  typedef typename if_<
    Params::is_key_compare_to::value,
    linear_search_compare_to_type,
    linear_search_plain_compare_type>::type linear_search_type;


  typedef typename if_<
    Params::is_key_compare_to::value,
    binary_search_compare_to_type,
    binary_search_plain_compare_type>::type binary_search_type;



  typedef typename if_<
    std::is_integral<key_type>::value ||
    std::is_floating_point<key_type>::value,
    linear_search_type, binary_search_type>::type search_type;

  struct base_fields {
    typedef typename Params::node_count_type field_type;


    bool leaf;

    field_type position;

    field_type max_count;

    field_type count;

    btree_node *parent;
  };

  enum {
    kValueSize = params_type::kValueSize,
    kTargetNodeSize = params_type::kTargetNodeSize,


    kNodeTargetValues = (kTargetNodeSize - sizeof(base_fields)) / kValueSize,



    kNodeValues = kNodeTargetValues >= 3 ? kNodeTargetValues : 3,

    kExactMatch = 1 << 30,
    kMatchMask = kExactMatch - 1,
  };

  struct leaf_fields : public base_fields {


    mutable_value_type values[kNodeValues];
  };

  struct internal_fields : public leaf_fields {



    btree_node *children[kNodeValues + 1];
  };

  struct root_fields : public internal_fields {
    btree_node *rightmost;
    size_type size;
  };

 public:


  bool leaf() const { return fields_.leaf; }


  int position() const { return fields_.position; }
  void set_position(int v) { fields_.position = v; }


  int count() const { return fields_.count; }
  void set_count(int v) { fields_.count = v; }
  int max_count() const { return fields_.max_count; }


  btree_node* parent() const { return fields_.parent; }



  bool is_root() const { return parent()->leaf(); }
  void make_root() {
    assert(parent()->is_root());
    fields_.parent = fields_.parent->parent();
  }


  btree_node* rightmost() const { return fields_.rightmost; }
  btree_node** mutable_rightmost() { return &fields_.rightmost; }


  size_type size() const { return fields_.size; }
  size_type* mutable_size() { return &fields_.size; }


  const key_type& key(int i) const {
    return params_type::key(fields_.values[i]);
  }
  reference value(int i) {
    return reinterpret_cast<reference>(fields_.values[i]);
  }
  const_reference value(int i) const {
    return reinterpret_cast<const_reference>(fields_.values[i]);
  }
  mutable_value_type* mutable_value(int i) {
    return &fields_.values[i];
  }


  void value_swap(int i, btree_node *x, int j) {
    params_type::swap(mutable_value(i), x->mutable_value(j));
  }


  btree_node* child(int i) const { return fields_.children[i]; }
  btree_node** mutable_child(int i) { return &fields_.children[i]; }
  void set_child(int i, btree_node *c) {
    *mutable_child(i) = c;
    c->fields_.parent = this;
    c->fields_.position = i;
  }


  template <typename Compare>
  int lower_bound(const key_type &k, const Compare &comp) const {
    return search_type::lower_bound(k, *this, comp);
  }

  template <typename Compare>
  int upper_bound(const key_type &k, const Compare &comp) const {
    return search_type::upper_bound(k, *this, comp);
  }



  template <typename Compare>
  int linear_search_plain_compare(
      const key_type &k, int s, int e, const Compare &comp) const {
    while (s < e) {
      if (!btree_compare_keys(comp, key(s), k)) {
        break;
      }
      ++s;
    }
    return s;
  }



  template <typename Compare>
  int linear_search_compare_to(
      const key_type &k, int s, int e, const Compare &comp) const {
    while (s < e) {
      int c = comp(key(s), k);
      if (c == 0) {
        return s | kExactMatch;
      } else if (c > 0) {
        break;
      }
      ++s;
    }
    return s;
  }



  template <typename Compare>
  int binary_search_plain_compare(
      const key_type &k, int s, int e, const Compare &comp) const {
    while (s != e) {
      int mid = (s + e) / 2;
      if (btree_compare_keys(comp, key(mid), k)) {
        s = mid + 1;
      } else {
        e = mid;
      }
    }
    return s;
  }



  template <typename CompareTo>
  int binary_search_compare_to(
      const key_type &k, int s, int e, const CompareTo &comp) const {
    while (s != e) {
      int mid = (s + e) / 2;
      int c = comp(key(mid), k);
      if (c < 0) {
        s = mid + 1;
      } else if (c > 0) {
        e = mid;
      } else {




        s = binary_search_compare_to(k, s, mid, comp);
        return s | kExactMatch;
      }
    }
    return s;
  }



  void insert_value(int i, const value_type &x);



  void remove_value(int i);


  void rebalance_right_to_left(btree_node *sibling, int to_move);
  void rebalance_left_to_right(btree_node *sibling, int to_move);


  void split(btree_node *sibling, int insert_position);



  void merge(btree_node *sibling);


  void swap(btree_node *src);


  static btree_node* init_leaf(
      leaf_fields *f, btree_node *parent, int max_count) {
    btree_node *n = reinterpret_cast<btree_node*>(f);
    f->leaf = 1;
    f->position = 0;
    f->max_count = max_count;
    f->count = 0;
    f->parent = parent;
    if (!NDEBUG) {
      memset(&f->values, 0, max_count * sizeof(value_type));
    }
    return n;
  }
  static btree_node* init_internal(internal_fields *f, btree_node *parent) {
    btree_node *n = init_leaf(f, parent, kNodeValues);
    f->leaf = 0;
    if (!NDEBUG) {
      memset(f->children, 0, sizeof(f->children));
    }
    return n;
  }
  static btree_node* init_root(root_fields *f, btree_node *parent) {
    btree_node *n = init_internal(f, parent);
    f->rightmost = parent;
    f->size = parent->count();
    return n;
  }
  void destroy() {
    for (int i = 0; i < count(); ++i) {
      value_destroy(i);
    }
  }

 private:
  void value_init(int i) {
    new (&fields_.values[i]) mutable_value_type;
  }
  void value_init(int i, const value_type &x) {
    new (&fields_.values[i]) mutable_value_type(x);
  }
  void value_destroy(int i) {
    fields_.values[i].~mutable_value_type();
  }

 private:
  root_fields fields_;

 private:
  btree_node(const btree_node&);
  void operator=(const btree_node&);
};

template <typename Node, typename Reference, typename Pointer>
struct btree_iterator {
  typedef typename Node::key_type key_type;
  typedef typename Node::size_type size_type;
  typedef typename Node::difference_type difference_type;
  typedef typename Node::params_type params_type;

  typedef Node node_type;
  typedef typename std::remove_const<Node>::type normal_node;
  typedef const Node const_node;
  typedef typename params_type::value_type value_type;
  typedef typename params_type::pointer normal_pointer;
  typedef typename params_type::reference normal_reference;
  typedef typename params_type::const_pointer const_pointer;
  typedef typename params_type::const_reference const_reference;

  typedef Pointer pointer;
  typedef Reference reference;
  typedef std::bidirectional_iterator_tag iterator_category;

  typedef btree_iterator<
    normal_node, normal_reference, normal_pointer> iterator;
  typedef btree_iterator<
    const_node, const_reference, const_pointer> const_iterator;
  typedef btree_iterator<Node, Reference, Pointer> self_type;

  btree_iterator()
      : node(NULL),
        position(-1) {
  }
  btree_iterator(Node *n, int p)
      : node(n),
        position(p) {
  }
  btree_iterator(const iterator &x)
      : node(x.node),
        position(x.position) {
  }


  void increment() {
    if (node->leaf() && ++position < node->count()) {
      return;
    }
    increment_slow();
  }
  void increment_by(int count);
  void increment_slow();

  void decrement() {
    if (node->leaf() && --position >= 0) {
      return;
    }
    decrement_slow();
  }
  void decrement_slow();

  bool operator==(const const_iterator &x) const {
    return node == x.node && position == x.position;
  }
  bool operator!=(const const_iterator &x) const {
    return node != x.node || position != x.position;
  }


  const key_type& key() const {
    return node->key(position);
  }
  reference operator*() const {
    return node->value(position);
  }
  pointer operator->() const {
    return &node->value(position);
  }

  self_type& operator++() {
    increment();
    return *this;
  }
  self_type& operator--() {
    decrement();
    return *this;
  }
  self_type operator++(int) {
    self_type tmp = *this;
    ++*this;
    return tmp;
  }
  self_type operator--(int) {
    self_type tmp = *this;
    --*this;
    return tmp;
  }


  Node *node;

  int position;
};


struct btree_internal_locate_plain_compare {
  template <typename K, typename T, typename Iter>
  static std::pair<Iter, int> dispatch(const K &k, const T &t, Iter iter) {
    return t.internal_locate_plain_compare(k, iter);
  }
};


struct btree_internal_locate_compare_to {
  template <typename K, typename T, typename Iter>
  static std::pair<Iter, int> dispatch(const K &k, const T &t, Iter iter) {
    return t.internal_locate_compare_to(k, iter);
  }
};

template <typename Params>
class btree : public Params::key_compare {
  typedef btree<Params> self_type;
  typedef btree_node<Params> node_type;
  typedef typename node_type::base_fields base_fields;
  typedef typename node_type::leaf_fields leaf_fields;
  typedef typename node_type::internal_fields internal_fields;
  typedef typename node_type::root_fields root_fields;
  typedef typename Params::is_key_compare_to is_key_compare_to;

  friend class btree_internal_locate_plain_compare;
  friend class btree_internal_locate_compare_to;
  typedef typename if_<
    is_key_compare_to::value,
    btree_internal_locate_compare_to,
    btree_internal_locate_plain_compare>::type internal_locate_type;

  enum {
    kNodeValues = node_type::kNodeValues,
    kMinNodeValues = kNodeValues / 2,
    kValueSize = node_type::kValueSize,
    kExactMatch = node_type::kExactMatch,
    kMatchMask = node_type::kMatchMask,
  };







  template <typename Base, typename Data>
  struct empty_base_handle : public Base {
    empty_base_handle(const Base &b, const Data &d)
        : Base(b),
          data(d) {
    }
    Data data;
  };

  struct node_stats {
    node_stats(ssize_t l, ssize_t i)
        : leaf_nodes(l),
          internal_nodes(i) {
    }

    node_stats& operator+=(const node_stats &x) {
      leaf_nodes += x.leaf_nodes;
      internal_nodes += x.internal_nodes;
      return *this;
    }

    ssize_t leaf_nodes;
    ssize_t internal_nodes;
  };

 public:
  typedef Params params_type;
  typedef typename Params::key_type key_type;
  typedef typename Params::data_type data_type;
  typedef typename Params::mapped_type mapped_type;
  typedef typename Params::value_type value_type;
  typedef typename Params::key_compare key_compare;
  typedef typename Params::pointer pointer;
  typedef typename Params::const_pointer const_pointer;
  typedef typename Params::reference reference;
  typedef typename Params::const_reference const_reference;
  typedef typename Params::size_type size_type;
  typedef typename Params::difference_type difference_type;
  typedef btree_iterator<node_type, reference, pointer> iterator;
  typedef typename iterator::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;

  typedef typename Params::allocator_type allocator_type;
  typedef typename allocator_type::template rebind<char>::other
    internal_allocator_type;

 public:

  btree(const key_compare &comp, const allocator_type &alloc);


  btree(const self_type &x);


  ~btree() {
    clear();
  }


  iterator begin() {
    return iterator(leftmost(), 0);
  }
  const_iterator begin() const {
    return const_iterator(leftmost(), 0);
  }
  iterator end() {
    return iterator(rightmost(), rightmost() ? rightmost()->count() : 0);
  }
  const_iterator end() const {
    return const_iterator(rightmost(), rightmost() ? rightmost()->count() : 0);
  }
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }


  iterator lower_bound(const key_type &key) {
    return internal_end(
        internal_lower_bound(key, iterator(root(), 0)));
  }
  const_iterator lower_bound(const key_type &key) const {
    return internal_end(
        internal_lower_bound(key, const_iterator(root(), 0)));
  }


  iterator upper_bound(const key_type &key) {
    return internal_end(
        internal_upper_bound(key, iterator(root(), 0)));
  }
  const_iterator upper_bound(const key_type &key) const {
    return internal_end(
        internal_upper_bound(key, const_iterator(root(), 0)));
  }




  std::pair<iterator,iterator> equal_range(const key_type &key) {
    return std::make_pair(lower_bound(key), upper_bound(key));
  }
  std::pair<const_iterator,const_iterator> equal_range(const key_type &key) const {
    return std::make_pair(lower_bound(key), upper_bound(key));
  }






  template <typename ValuePointer>
  std::pair<iterator,bool> insert_unique(const key_type &key, ValuePointer value);



  std::pair<iterator,bool> insert_unique(const value_type &v) {
    return insert_unique(params_type::key(v), &v);
  }





  iterator insert_unique(iterator position, const value_type &v);


  template <typename InputIterator>
  void insert_unique(InputIterator b, InputIterator e);





  template <typename ValuePointer>
  iterator insert_multi(const key_type &key, ValuePointer value);


  iterator insert_multi(const value_type &v) {
    return insert_multi(params_type::key(v), &v);
  }





  iterator insert_multi(iterator position, const value_type &v);


  template <typename InputIterator>
  void insert_multi(InputIterator b, InputIterator e);

  void assign(const self_type &x);




  iterator erase(iterator iter);


  int erase(iterator begin, iterator end);



  int erase_unique(const key_type &key);



  int erase_multi(const key_type &key);



  iterator find_unique(const key_type &key) {
    return internal_end(
        internal_find_unique(key, iterator(root(), 0)));
  }
  const_iterator find_unique(const key_type &key) const {
    return internal_end(
        internal_find_unique(key, const_iterator(root(), 0)));
  }
  iterator find_multi(const key_type &key) {
    return internal_end(
        internal_find_multi(key, iterator(root(), 0)));
  }
  const_iterator find_multi(const key_type &key) const {
    return internal_end(
        internal_find_multi(key, const_iterator(root(), 0)));
  }


  size_type count_unique(const key_type &key) const {
    const_iterator begin = internal_find_unique(
        key, const_iterator(root(), 0));
    if (!begin.node) {

      return 0;
    }
    return 1;
  }

  size_type count_multi(const key_type &key) const {
    return distance(lower_bound(key), upper_bound(key));
  }


  void clear();


  void swap(self_type &x);


  self_type& operator=(const self_type &x) {
    if (&x == this) {

      return *this;
    }
    assign(x);
    return *this;
  }

  key_compare* mutable_key_comp() {
    return this;
  }
  const key_compare& key_comp() const {
    return *this;
  }
  bool compare_keys(const key_type &x, const key_type &y) const {
    return btree_compare_keys(key_comp(), x, y);
  }



  void dump(std::ostream &os) const {
    if (root() != NULL) {
      internal_dump(os, root(), 0);
    }
  }


  void verify() const;


  size_type size() const {
    if (empty()) return 0;
    if (root()->leaf()) return root()->count();
    return root()->size();
  }
  size_type max_size() const { return std::numeric_limits<size_type>::max(); }
  bool empty() const { return root() == NULL; }


  size_type height() const {
    size_type h = 0;
    if (root()) {




      const node_type *n = root();
      do {
        ++h;
        n = n->parent();
      } while (n != root());
    }
    return h;
  }


  size_type leaf_nodes() const {
    return internal_stats(root()).leaf_nodes;
  }
  size_type internal_nodes() const {
    return internal_stats(root()).internal_nodes;
  }
  size_type nodes() const {
    node_stats stats = internal_stats(root());
    return stats.leaf_nodes + stats.internal_nodes;
  }


  size_type bytes_used() const {
    node_stats stats = internal_stats(root());
    if (stats.leaf_nodes == 1 && stats.internal_nodes == 0) {
      return sizeof(*this) +
          sizeof(base_fields) + root()->max_count() * sizeof(value_type);
    } else {
      return sizeof(*this) +
          sizeof(root_fields) - sizeof(internal_fields) +
          stats.leaf_nodes * sizeof(leaf_fields) +
          stats.internal_nodes * sizeof(internal_fields);
    }
  }


  static double average_bytes_per_value() {



    return sizeof(leaf_fields) / (kNodeValues * 0.75);
  }





  double fullness() const {
    return double(size()) / (nodes() * kNodeValues);
  }



  double overhead() const {
    if (empty()) {
      return 0.0;
    }
    return (bytes_used() - size() * kValueSize) / double(size());
  }

 private:

  node_type* root() { return root_.data; }
  const node_type* root() const { return root_.data; }
  node_type** mutable_root() { return &root_.data; }


  node_type* rightmost() {
    return (!root() || root()->leaf()) ? root() : root()->rightmost();
  }
  const node_type* rightmost() const {
    return (!root() || root()->leaf()) ? root() : root()->rightmost();
  }
  node_type** mutable_rightmost() { return root()->mutable_rightmost(); }


  node_type* leftmost() { return root() ? root()->parent() : NULL; }
  const node_type* leftmost() const { return root() ? root()->parent() : NULL; }


  size_type* mutable_size() { return root()->mutable_size(); }


  internal_allocator_type* mutable_internal_allocator() {
    return static_cast<internal_allocator_type*>(&root_);
  }
  const internal_allocator_type& internal_allocator() const {
    return *static_cast<const internal_allocator_type*>(&root_);
  }


  node_type* new_internal_node(node_type *parent) {
    internal_fields *p = reinterpret_cast<internal_fields*>(
        mutable_internal_allocator()->allocate(sizeof(internal_fields)));
    return node_type::init_internal(p, parent);
  }
  node_type* new_internal_root_node() {
    root_fields *p = reinterpret_cast<root_fields*>(
        mutable_internal_allocator()->allocate(sizeof(root_fields)));
    return node_type::init_root(p, root()->parent());
  }
  node_type* new_leaf_node(node_type *parent) {
    leaf_fields *p = reinterpret_cast<leaf_fields*>(
        mutable_internal_allocator()->allocate(sizeof(leaf_fields)));
    return node_type::init_leaf(p, parent, kNodeValues);
  }
  node_type* new_leaf_root_node(int max_count) {
    leaf_fields *p = reinterpret_cast<leaf_fields*>(
        mutable_internal_allocator()->allocate(
            sizeof(base_fields) + max_count * sizeof(value_type)));
    return node_type::init_leaf(p, reinterpret_cast<node_type*>(p), max_count);
  }
  void delete_internal_node(node_type *node) {
    node->destroy();
    assert(node != root());
    mutable_internal_allocator()->deallocate(
        reinterpret_cast<char*>(node), sizeof(internal_fields));
  }
  void delete_internal_root_node() {
    root()->destroy();
    mutable_internal_allocator()->deallocate(
        reinterpret_cast<char*>(root()), sizeof(root_fields));
  }
  void delete_leaf_node(node_type *node) {
    node->destroy();
    mutable_internal_allocator()->deallocate(
        reinterpret_cast<char*>(node),
        sizeof(base_fields) + node->max_count() * sizeof(value_type));
  }


  void rebalance_or_split(iterator *iter);



  void merge_nodes(node_type *left, node_type *right);





  bool try_merge_or_rebalance(iterator *iter);


  void try_shrink();

  iterator internal_end(iterator iter) {
    return iter.node ? iter : end();
  }
  const_iterator internal_end(const_iterator iter) const {
    return iter.node ? iter : end();
  }



  iterator internal_insert(iterator iter, const value_type &v);





  template <typename IterType>
  static IterType internal_last(IterType iter);










  template <typename IterType>
  std::pair<IterType, int> internal_locate(
      const key_type &key, IterType iter) const;
  template <typename IterType>
  std::pair<IterType, int> internal_locate_plain_compare(
      const key_type &key, IterType iter) const;
  template <typename IterType>
  std::pair<IterType, int> internal_locate_compare_to(
      const key_type &key, IterType iter) const;


  template <typename IterType>
  IterType internal_lower_bound(
      const key_type &key, IterType iter) const;


  template <typename IterType>
  IterType internal_upper_bound(
      const key_type &key, IterType iter) const;


  template <typename IterType>
  IterType internal_find_unique(
      const key_type &key, IterType iter) const;


  template <typename IterType>
  IterType internal_find_multi(
      const key_type &key, IterType iter) const;


  void internal_clear(node_type *node);


  void internal_dump(std::ostream &os, const node_type *node, int level) const;


  int internal_verify(const node_type *node,
                      const key_type *lo, const key_type *hi) const;

  node_stats internal_stats(const node_type *node) const {
    if (!node) {
      return node_stats(0, 0);
    }
    if (node->leaf()) {
      return node_stats(1, 0);
    }
    node_stats res(0, 1);
    for (int i = 0; i <= node->count(); ++i) {
      res += internal_stats(node->child(i));
    }
    return res;
  }

 private:
  empty_base_handle<internal_allocator_type, node_type*> root_;

 private:


  template <typename R>
  static typename if_<
   if_<is_key_compare_to::value,
             std::is_same<R, int>,
             std::is_same<R, bool> >::type::value,
   big_, small_>::type key_compare_checker(R);



  static key_compare key_compare_helper();







  COMPILE_ASSERT(
      sizeof(key_compare_checker(key_compare_helper()(key_type(), key_type()))) ==
      sizeof(big_),
      key_comparison_function_must_return_bool);



  COMPILE_ASSERT(kNodeValues <
                 (1 << (8 * sizeof(typename base_fields::field_type))),
                 target_node_size_too_large);


  COMPILE_ASSERT(sizeof(base_fields) >= 2 * sizeof(void*),
                 node_space_assumption_incorrect);
};



template <typename P>
inline void btree_node<P>::insert_value(int i, const value_type &x) {
  assert(i <= count());
  value_init(count(), x);
  for (int j = count(); j > i; --j) {
    value_swap(j, this, j - 1);
  }
  set_count(count() + 1);

  if (!leaf()) {
    ++i;
    for (int j = count(); j > i; --j) {
      *mutable_child(j) = child(j - 1);
      child(j)->set_position(j);
    }
    *mutable_child(i) = NULL;
  }
}

template <typename P>
inline void btree_node<P>::remove_value(int i) {
  if (!leaf()) {
    assert(child(i + 1)->count() == 0);
    for (int j = i + 1; j < count(); ++j) {
      *mutable_child(j) = child(j + 1);
      child(j)->set_position(j);
    }
    *mutable_child(count()) = NULL;
  }

  set_count(count() - 1);
  for (; i < count(); ++i) {
    value_swap(i, this, i + 1);
  }
  value_destroy(i);
}

template <typename P>
void btree_node<P>::rebalance_right_to_left(btree_node *src, int to_move) {
  assert(parent() == src->parent());
  assert(position() + 1 == src->position());
  assert(src->count() >= count());
  assert(to_move >= 1);
  assert(to_move <= src->count());


  for (int i = 0; i < to_move; ++i) {
    value_init(i + count());
  }



  value_swap(count(), parent(), position());
  parent()->value_swap(position(), src, to_move - 1);


  for (int i = 1; i < to_move; ++i) {
    value_swap(count() + i, src, i - 1);
  }

  for (int i = to_move; i < src->count(); ++i) {
    src->value_swap(i - to_move, src, i);
  }
  for (int i = 1; i <= to_move; ++i) {
    src->value_destroy(src->count() - i);
  }

  if (!leaf()) {

    for (int i = 0; i < to_move; ++i) {
      set_child(1 + count() + i, src->child(i));
    }
    for (int i = 0; i <= src->count() - to_move; ++i) {
      assert(i + to_move <= src->max_count());
      src->set_child(i, src->child(i + to_move));
      *src->mutable_child(i + to_move) = NULL;
    }
  }


  set_count(count() + to_move);
  src->set_count(src->count() - to_move);
}

template <typename P>
void btree_node<P>::rebalance_left_to_right(btree_node *dest, int to_move) {
  assert(parent() == dest->parent());
  assert(position() + 1 == dest->position());
  assert(count() >= dest->count());
  assert(to_move >= 1);
  assert(to_move <= count());


  for (int i = 0; i < to_move; ++i) {
    dest->value_init(i + dest->count());
  }
  for (int i = dest->count() - 1; i >= 0; --i) {
    dest->value_swap(i, dest, i + to_move);
  }



  dest->value_swap(to_move - 1, parent(), position());
  parent()->value_swap(position(), this, count() - to_move);
  value_destroy(count() - to_move);


  for (int i = 1; i < to_move; ++i) {
    value_swap(count() - to_move + i, dest, i - 1);
    value_destroy(count() - to_move + i);
  }

  if (!leaf()) {

    for (int i = dest->count(); i >= 0; --i) {
      dest->set_child(i + to_move, dest->child(i));
      *dest->mutable_child(i) = NULL;
    }
    for (int i = 1; i <= to_move; ++i) {
      dest->set_child(i - 1, child(count() - to_move + i));
      *mutable_child(count() - to_move + i) = NULL;
    }
  }


  set_count(count() - to_move);
  dest->set_count(dest->count() + to_move);
}

template <typename P>
void btree_node<P>::split(btree_node *dest, int insert_position) {
  assert(dest->count() == 0);





  if (insert_position == 0) {
    dest->set_count(count() - 1);
  } else if (insert_position == max_count()) {
    dest->set_count(0);
  } else {
    dest->set_count(count() / 2);
  }
  set_count(count() - dest->count());
  assert(count() >= 1);


  for (int i = 0; i < dest->count(); ++i) {
    dest->value_init(i);
    value_swap(count() + i, dest, i);
    value_destroy(count() + i);
  }


  set_count(count() - 1);
  parent()->insert_value(position(), value_type());
  value_swap(count(), parent(), position());
  value_destroy(count());
  parent()->set_child(position() + 1, dest);

  if (!leaf()) {
    for (int i = 0; i <= dest->count(); ++i) {
      assert(child(count() + i + 1) != NULL);
      dest->set_child(i, child(count() + i + 1));
      *mutable_child(count() + i + 1) = NULL;
    }
  }
}

template <typename P>
void btree_node<P>::merge(btree_node *src) {
  assert(parent() == src->parent());
  assert(position() + 1 == src->position());


  value_init(count());
  value_swap(count(), parent(), position());


  for (int i = 0; i < src->count(); ++i) {
    value_init(1 + count() + i);
    value_swap(1 + count() + i, src, i);
    src->value_destroy(i);
  }

  if (!leaf()) {

    for (int i = 0; i <= src->count(); ++i) {
      set_child(1 + count() + i, src->child(i));
      *src->mutable_child(i) = NULL;
    }
  }


  set_count(1 + count() + src->count());
  src->set_count(0);


  parent()->remove_value(position());
}

template <typename P>
void btree_node<P>::swap(btree_node *x) {
  assert(leaf() == x->leaf());


  for (int i = count(); i < x->count(); ++i) {
    value_init(i);
  }
  for (int i = x->count(); i < count(); ++i) {
    x->value_init(i);
  }
  int n = std::max(count(), x->count());
  for (int i = 0; i < n; ++i) {
    value_swap(i, x, i);
  }
  for (int i = count(); i < x->count(); ++i) {
    x->value_destroy(i);
  }
  for (int i = x->count(); i < count(); ++i) {
    value_destroy(i);
  }

  if (!leaf()) {

    for (int i = 0; i <= n; ++i) {
      btree_swap_helper(*mutable_child(i), *x->mutable_child(i));
    }
    for (int i = 0; i <= count(); ++i) {
      x->child(i)->fields_.parent = x;
    }
    for (int i = 0; i <= x->count(); ++i) {
      child(i)->fields_.parent = this;
    }
  }


  btree_swap_helper(fields_.count, x->fields_.count);
}



template <typename N, typename R, typename P>
void btree_iterator<N, R, P>::increment_slow() {
  if (node->leaf()) {
    assert(position >= node->count());
    self_type save(*this);
    while (position == node->count() && !node->is_root()) {
      assert(node->parent()->child(node->position()) == node);
      position = node->position();
      node = node->parent();
    }
    if (position == node->count()) {
      *this = save;
    }
  } else {
    assert(position < node->count());
    node = node->child(position + 1);
    while (!node->leaf()) {
      node = node->child(0);
    }
    position = 0;
  }
}

template <typename N, typename R, typename P>
void btree_iterator<N, R, P>::increment_by(int count) {
  while (count > 0) {
    if (node->leaf()) {
      int rest = node->count() - position;
      position += std::min(rest, count);
      count = count - rest;
      if (position < node->count()) {
        return;
      }
    } else {
      --count;
    }
    increment_slow();
  }
}

template <typename N, typename R, typename P>
void btree_iterator<N, R, P>::decrement_slow() {
  if (node->leaf()) {
    assert(position <= -1);
    self_type save(*this);
    while (position < 0 && !node->is_root()) {
      assert(node->parent()->child(node->position()) == node);
      position = node->position() - 1;
      node = node->parent();
    }
    if (position < 0) {
      *this = save;
    }
  } else {
    assert(position >= 0);
    node = node->child(position);
    while (!node->leaf()) {
      node = node->child(node->count());
    }
    position = node->count() - 1;
  }
}



template <typename P>
btree<P>::btree(const key_compare &comp, const allocator_type &alloc)
    : key_compare(comp),
      root_(alloc, NULL) {
}

template <typename P>
btree<P>::btree(const self_type &x)
    : key_compare(x.key_comp()),
      root_(x.internal_allocator(), NULL) {
  assign(x);
}

template <typename P> template <typename ValuePointer>
std::pair<typename btree<P>::iterator, bool>
btree<P>::insert_unique(const key_type &key, ValuePointer value) {
  if (empty()) {
    *mutable_root() = new_leaf_root_node(1);
  }

  std::pair<iterator, int> res = internal_locate(key, iterator(root(), 0));
  iterator &iter = res.first;
  if (res.second == kExactMatch) {

    return std::make_pair(internal_last(iter), false);
  } else if (!res.second) {
    iterator last = internal_last(iter);
    if (last.node && !compare_keys(key, last.key())) {

      return std::make_pair(last, false);
    }
  }

  return std::make_pair(internal_insert(iter, *value), true);
}

template <typename P>
inline typename btree<P>::iterator
btree<P>::insert_unique(iterator position, const value_type &v) {
  if (!empty()) {
    const key_type &key = params_type::key(v);
    if (position == end() || compare_keys(key, position.key())) {
      iterator prev = position;
      if (position == begin() || compare_keys((--prev).key(), key)) {

        return internal_insert(position, v);
      }
    } else if (compare_keys(position.key(), key)) {
      iterator next = position;
      ++next;
      if (next == end() || compare_keys(key, next.key())) {

        return internal_insert(next, v);
      }
    } else {

      return position;
    }
  }
  return insert_unique(v).first;
}

template <typename P> template <typename InputIterator>
void btree<P>::insert_unique(InputIterator b, InputIterator e) {
  for (; b != e; ++b) {
    insert_unique(end(), *b);
  }
}

template <typename P> template <typename ValuePointer>
typename btree<P>::iterator
btree<P>::insert_multi(const key_type &key, ValuePointer value) {
  if (empty()) {
    *mutable_root() = new_leaf_root_node(1);
  }

  iterator iter = internal_upper_bound(key, iterator(root(), 0));
  if (!iter.node) {
    iter = end();
  }
  return internal_insert(iter, *value);
}

template <typename P>
typename btree<P>::iterator
btree<P>::insert_multi(iterator position, const value_type &v) {
  if (!empty()) {
    const key_type &key = params_type::key(v);
    if (position == end() || !compare_keys(position.key(), key)) {
      iterator prev = position;
      if (position == begin() || !compare_keys(key, (--prev).key())) {

        return internal_insert(position, v);
      }
    } else {
      iterator next = position;
      ++next;
      if (next == end() || !compare_keys(next.key(), key)) {

        return internal_insert(next, v);
      }
    }
  }
  return insert_multi(v);
}

template <typename P> template <typename InputIterator>
void btree<P>::insert_multi(InputIterator b, InputIterator e) {
  for (; b != e; ++b) {
    insert_multi(end(), *b);
  }
}

template <typename P>
void btree<P>::assign(const self_type &x) {
  clear();

  *mutable_key_comp() = x.key_comp();
  *mutable_internal_allocator() = x.internal_allocator();



  for (const_iterator iter = x.begin(); iter != x.end(); ++iter) {
    if (empty()) {
      insert_multi(*iter);
    } else {


      internal_insert(end(), *iter);
    }
  }
}

template <typename P>
typename btree<P>::iterator btree<P>::erase(iterator iter) {
  bool internal_delete = false;
  if (!iter.node->leaf()) {


    iterator tmp_iter(iter--);
    assert(iter.node->leaf());
    assert(!compare_keys(tmp_iter.key(), iter.key()));
    iter.node->value_swap(iter.position, tmp_iter.node, tmp_iter.position);
    internal_delete = true;
    --*mutable_size();
  } else if (!root()->leaf()) {
    --*mutable_size();
  }


  iter.node->remove_value(iter.position);









  iterator res(iter);
  for (;;) {
    if (iter.node == root()) {
      try_shrink();
      if (empty()) {
        return end();
      }
      break;
    }
    if (iter.node->count() >= kMinNodeValues) {
      break;
    }
    bool merged = try_merge_or_rebalance(&iter);
    if (iter.node->leaf()) {
      res = iter;
    }
    if (!merged) {
      break;
    }
    iter.node = iter.node->parent();
  }



  if (res.position == res.node->count()) {
    res.position = res.node->count() - 1;
    ++res;
  }

  if (internal_delete) {
    ++res;
  }
  return res;
}

template <typename P>
int btree<P>::erase(iterator begin, iterator end) {
  int count = distance(begin, end);
  for (int i = 0; i < count; i++) {
    begin = erase(begin);
  }
  return count;
}

template <typename P>
int btree<P>::erase_unique(const key_type &key) {
  iterator iter = internal_find_unique(key, iterator(root(), 0));
  if (!iter.node) {

    return 0;
  }
  erase(iter);
  return 1;
}

template <typename P>
int btree<P>::erase_multi(const key_type &key) {
  iterator begin = internal_lower_bound(key, iterator(root(), 0));
  if (!begin.node) {

    return 0;
  }

  iterator end = internal_end(
      internal_upper_bound(key, iterator(root(), 0)));
  return erase(begin, end);
}

template <typename P>
void btree<P>::clear() {
  if (root() != NULL) {
    internal_clear(root());
  }
  *mutable_root() = NULL;
}

template <typename P>
void btree<P>::swap(self_type &x) {
  std::swap(static_cast<key_compare&>(*this), static_cast<key_compare&>(x));
  std::swap(root_, x.root_);
}

template <typename P>
void btree<P>::verify() const {
  if (root() != NULL) {
    assert(size() == internal_verify(root(), NULL, NULL));
    assert(leftmost() == (++const_iterator(root(), -1)).node);
    assert(rightmost() == (--const_iterator(root(), root()->count())).node);
    assert(leftmost()->leaf());
    assert(rightmost()->leaf());
  } else {
    assert(size() == 0);
    assert(leftmost() == NULL);
    assert(rightmost() == NULL);
  }
}

template <typename P>
void btree<P>::rebalance_or_split(iterator *iter) {
  node_type *&node = iter->node;
  int &insert_position = iter->position;
  assert(node->count() == node->max_count());


  node_type *parent = node->parent();
  if (node != root()) {
    if (node->position() > 0) {

      node_type *left = parent->child(node->position() - 1);
      if (left->count() < left->max_count()) {



        int to_move = (left->max_count() - left->count()) /
            (1 + (insert_position < left->max_count()));
        to_move = std::max(1, to_move);

        if (((insert_position - to_move) >= 0) ||
            ((left->count() + to_move) < left->max_count())) {
          left->rebalance_right_to_left(node, to_move);

          assert(node->max_count() - node->count() == to_move);
          insert_position = insert_position - to_move;
          if (insert_position < 0) {
            insert_position = insert_position + left->count() + 1;
            node = left;
          }

          assert(node->count() < node->max_count());
          return;
        }
      }
    }

    if (node->position() < parent->count()) {

      node_type *right = parent->child(node->position() + 1);
      if (right->count() < right->max_count()) {



        int to_move = (right->max_count() - right->count()) /
            (1 + (insert_position > 0));
        to_move = std::max(1, to_move);

        if ((insert_position <= (node->count() - to_move)) ||
            ((right->count() + to_move) < right->max_count())) {
          node->rebalance_left_to_right(right, to_move);

          if (insert_position > node->count()) {
            insert_position = insert_position - node->count() - 1;
            node = right;
          }

          assert(node->count() < node->max_count());
          return;
        }
      }
    }



    if (parent->count() == parent->max_count()) {
      iterator parent_iter(node->parent(), node->position());
      rebalance_or_split(&parent_iter);
    }
  } else {

    if (root()->leaf()) {


      parent = new_internal_root_node();
      parent->set_child(0, root());
      *mutable_root() = parent;
      assert(*mutable_rightmost() == parent->child(0));
    } else {




      parent = new_internal_node(parent);
      parent->set_child(0, parent);
      parent->swap(root());
      node = parent;
    }
  }


  node_type *split_node;
  if (node->leaf()) {
    split_node = new_leaf_node(parent);
    node->split(split_node, insert_position);
    if (rightmost() == node) {
      *mutable_rightmost() = split_node;
    }
  } else {
    split_node = new_internal_node(parent);
    node->split(split_node, insert_position);
  }

  if (insert_position > node->count()) {
    insert_position = insert_position - node->count() - 1;
    node = split_node;
  }
}

template <typename P>
void btree<P>::merge_nodes(node_type *left, node_type *right) {
  left->merge(right);
  if (right->leaf()) {
    if (rightmost() == right) {
      *mutable_rightmost() = left;
    }
    delete_leaf_node(right);
  } else {
    delete_internal_node(right);
  }
}

template <typename P>
bool btree<P>::try_merge_or_rebalance(iterator *iter) {
  node_type *parent = iter->node->parent();
  if (iter->node->position() > 0) {

    node_type *left = parent->child(iter->node->position() - 1);
    if ((1 + left->count() + iter->node->count()) <= left->max_count()) {
      iter->position += 1 + left->count();
      merge_nodes(left, iter->node);
      iter->node = left;
      return true;
    }
  }
  if (iter->node->position() < parent->count()) {

    node_type *right = parent->child(iter->node->position() + 1);
    if ((1 + iter->node->count() + right->count()) <= right->max_count()) {
      merge_nodes(iter->node, right);
      return true;
    }




    if ((right->count() > kMinNodeValues) &&
        ((iter->node->count() == 0) ||
         (iter->position > 0))) {
      int to_move = (right->count() - iter->node->count()) / 2;
      to_move = std::min(to_move, right->count() - 1);
      iter->node->rebalance_right_to_left(right, to_move);
      return false;
    }
  }
  if (iter->node->position() > 0) {




    node_type *left = parent->child(iter->node->position() - 1);
    if ((left->count() > kMinNodeValues) &&
        ((iter->node->count() == 0) ||
         (iter->position < iter->node->count()))) {
      int to_move = (left->count() - iter->node->count()) / 2;
      to_move = std::min(to_move, left->count() - 1);
      left->rebalance_left_to_right(iter->node, to_move);
      iter->position += to_move;
      return false;
    }
  }
  return false;
}

template <typename P>
void btree<P>::try_shrink() {
  if (root()->count() > 0) {
    return;
  }

  if (root()->leaf()) {
    assert(size() == 0);
    delete_leaf_node(root());
    *mutable_root() = NULL;
  } else {
    node_type *child = root()->child(0);
    if (child->leaf()) {

      child->make_root();
      delete_internal_root_node();
      *mutable_root() = child;
    } else {



      child->swap(root());
      delete_internal_node(child);
    }
  }
}

template <typename P> template <typename IterType>
inline IterType btree<P>::internal_last(IterType iter) {
  while (iter.node && iter.position == iter.node->count()) {
    iter.position = iter.node->position();
    iter.node = iter.node->parent();
    if (iter.node->leaf()) {
      iter.node = NULL;
    }
  }
  return iter;
}

template <typename P>
inline typename btree<P>::iterator
btree<P>::internal_insert(iterator iter, const value_type &v) {
  if (!iter.node->leaf()) {


    --iter;
    ++iter.position;
  }
  if (iter.node->count() == iter.node->max_count()) {

    if (iter.node->max_count() < kNodeValues) {


      assert(iter.node == root());
      iter.node = new_leaf_root_node(
          std::min<int>(kNodeValues, 2 * iter.node->max_count()));
      iter.node->swap(root());
      delete_leaf_node(root());
      *mutable_root() = iter.node;
    } else {
      rebalance_or_split(&iter);
      ++*mutable_size();
    }
  } else if (!root()->leaf()) {
    ++*mutable_size();
  }
  iter.node->insert_value(iter.position, v);
  return iter;
}

template <typename P> template <typename IterType>
inline std::pair<IterType, int> btree<P>::internal_locate(
    const key_type &key, IterType iter) const {
  return internal_locate_type::dispatch(key, *this, iter);
}

template <typename P> template <typename IterType>
inline std::pair<IterType, int> btree<P>::internal_locate_plain_compare(
    const key_type &key, IterType iter) const {
  for (;;) {
    iter.position = iter.node->lower_bound(key, key_comp());
    if (iter.node->leaf()) {
      break;
    }
    iter.node = iter.node->child(iter.position);
  }
  return std::make_pair(iter, 0);
}

template <typename P> template <typename IterType>
inline std::pair<IterType, int> btree<P>::internal_locate_compare_to(
    const key_type &key, IterType iter) const {
  for (;;) {
    int res = iter.node->lower_bound(key, key_comp());
    iter.position = res & kMatchMask;
    if (res & kExactMatch) {
      return std::make_pair(iter, static_cast<int>(kExactMatch));
    }
    if (iter.node->leaf()) {
      break;
    }
    iter.node = iter.node->child(iter.position);
  }
  return std::make_pair(iter, -kExactMatch);
}

template <typename P> template <typename IterType>
IterType btree<P>::internal_lower_bound(
    const key_type &key, IterType iter) const {
  if (iter.node) {
    for (;;) {
      iter.position =
          iter.node->lower_bound(key, key_comp()) & kMatchMask;
      if (iter.node->leaf()) {
        break;
      }
      iter.node = iter.node->child(iter.position);
    }
    iter = internal_last(iter);
  }
  return iter;
}

template <typename P> template <typename IterType>
IterType btree<P>::internal_upper_bound(
    const key_type &key, IterType iter) const {
  if (iter.node) {
    for (;;) {
      iter.position = iter.node->upper_bound(key, key_comp());
      if (iter.node->leaf()) {
        break;
      }
      iter.node = iter.node->child(iter.position);
    }
    iter = internal_last(iter);
  }
  return iter;
}

template <typename P> template <typename IterType>
IterType btree<P>::internal_find_unique(
    const key_type &key, IterType iter) const {
  if (iter.node) {
    std::pair<IterType, int> res = internal_locate(key, iter);
    if (res.second == kExactMatch) {
      return res.first;
    }
    if (!res.second) {
      iter = internal_last(res.first);
      if (iter.node && !compare_keys(key, iter.key())) {
        return iter;
      }
    }
  }
  return IterType(NULL, 0);
}

template <typename P> template <typename IterType>
IterType btree<P>::internal_find_multi(
    const key_type &key, IterType iter) const {
  if (iter.node) {
    iter = internal_lower_bound(key, iter);
    if (iter.node) {
      iter = internal_last(iter);
      if (iter.node && !compare_keys(key, iter.key())) {
        return iter;
      }
    }
  }
  return IterType(NULL, 0);
}

template <typename P>
void btree<P>::internal_clear(node_type *node) {
  if (!node->leaf()) {
    for (int i = 0; i <= node->count(); ++i) {
      internal_clear(node->child(i));
    }
    if (node == root()) {
      delete_internal_root_node();
    } else {
      delete_internal_node(node);
    }
  } else {
    delete_leaf_node(node);
  }
}

template <typename P>
void btree<P>::internal_dump(
    std::ostream &os, const node_type *node, int level) const {
  for (int i = 0; i < node->count(); ++i) {
    if (!node->leaf()) {
      internal_dump(os, node->child(i), level + 1);
    }
    for (int j = 0; j < level; ++j) {
      os << "  ";
    }
    os << node->key(i) << " [" << level << "]\n";
  }
  if (!node->leaf()) {
    internal_dump(os, node->child(node->count()), level + 1);
  }
}

template <typename P>
int btree<P>::internal_verify(
    const node_type *node, const key_type *lo, const key_type *hi) const {
  assert(node->count() > 0);
  assert(node->count() <= node->max_count());
  if (lo) {
    assert(!compare_keys(node->key(0), *lo));
  }
  if (hi) {
    assert(!compare_keys(*hi, node->key(node->count() - 1)));
  }
  for (int i = 1; i < node->count(); ++i) {
    assert(!compare_keys(node->key(i), node->key(i - 1)));
  }
  int count = node->count();
  if (!node->leaf()) {
    for (int i = 0; i <= node->count(); ++i) {
      assert(node->child(i) != NULL);
      assert(node->child(i)->parent() == node);
      assert(node->child(i)->position() == i);
      count += internal_verify(
          node->child(i),
          (i == 0) ? lo : &node->key(i - 1),
          (i == node->count()) ? hi : &node->key(i));
    }
  }
  return count;
}

}

#endif
