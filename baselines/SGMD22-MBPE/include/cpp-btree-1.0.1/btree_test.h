













#ifndef UTIL_BTREE_BTREE_TEST_H__
#define UTIL_BTREE_BTREE_TEST_H__

#include <stdio.h>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <iosfwd>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include "btree_container.h"

DECLARE_int32(test_values);
DECLARE_int32(benchmark_values);

namespace std {


template <typename T, typename U>
ostream& operator<<(ostream &os, const std::pair<T, U> &p) {
  os << "(" << p.first << "," << p.second << ")";
  return os;
}




template <typename T, typename U, typename V, typename W>
bool operator==(const std::pair<T, U> &x, const std::pair<V, W> &y) {
  return x.first == y.first && x.second == y.second;
}



template <typename T, typename U>
struct remove_const<pair<T, U> > {
  typedef pair<typename remove_const<T>::type,
               typename remove_const<U>::type> type;
};

}

namespace btree {


template <class _Pair>
struct select1st : public std::unary_function<_Pair, typename _Pair::first_type> {
  const typename _Pair::first_type& operator()(const _Pair& __x) const {
    return __x.first;
  }
};



template <typename K, typename V>
struct KeyOfValue {
  typedef select1st<V> type;
};

template <typename T>
struct identity {
  inline const T& operator()(const T& t) const { return t; }
};



template <typename K>
struct KeyOfValue<K, K> {
  typedef identity<K> type;
};


inline ptrdiff_t strcount(const char* buf_begin, const char* buf_end, char c) {
  if (buf_begin == NULL)
    return 0;
  if (buf_end <= buf_begin)
    return 0;
  ptrdiff_t num = 0;
  for (const char* bp = buf_begin; bp != buf_end; bp++) {
    if (*bp == c)
      num++;
  }
  return num;
}


inline ptrdiff_t strcount(const char* buf, size_t len, char c) {
  return strcount(buf, buf + len, c);
}

inline ptrdiff_t strcount(const std::string& buf, char c) {
  return strcount(buf.c_str(), buf.size(), c);
}





template <typename TreeType, typename CheckerType>
class base_checker {
  typedef base_checker<TreeType, CheckerType> self_type;

 public:
  typedef typename TreeType::key_type key_type;
  typedef typename TreeType::value_type value_type;
  typedef typename TreeType::key_compare key_compare;
  typedef typename TreeType::pointer pointer;
  typedef typename TreeType::const_pointer const_pointer;
  typedef typename TreeType::reference reference;
  typedef typename TreeType::const_reference const_reference;
  typedef typename TreeType::size_type size_type;
  typedef typename TreeType::difference_type difference_type;
  typedef typename TreeType::iterator iterator;
  typedef typename TreeType::const_iterator const_iterator;
  typedef typename TreeType::reverse_iterator reverse_iterator;
  typedef typename TreeType::const_reverse_iterator const_reverse_iterator;

 public:

  base_checker()
      : const_tree_(tree_) {
  }

  base_checker(const self_type &x)
      : tree_(x.tree_),
        const_tree_(tree_),
        checker_(x.checker_) {
  }

  template <typename InputIterator>
  base_checker(InputIterator b, InputIterator e)
      : tree_(b, e),
        const_tree_(tree_),
        checker_(b, e) {
  }


  iterator begin() { return tree_.begin(); }
  const_iterator begin() const { return tree_.begin(); }
  iterator end() { return tree_.end(); }
  const_iterator end() const { return tree_.end(); }
  reverse_iterator rbegin() { return tree_.rbegin(); }
  const_reverse_iterator rbegin() const { return tree_.rbegin(); }
  reverse_iterator rend() { return tree_.rend(); }
  const_reverse_iterator rend() const { return tree_.rend(); }


  template <typename IterType, typename CheckerIterType>
  IterType iter_check(
      IterType tree_iter, CheckerIterType checker_iter) const {
    if (tree_iter == tree_.end()) {
      EXPECT_EQ(checker_iter, checker_.end());
    } else {
      EXPECT_EQ(*tree_iter, *checker_iter);
    }
    return tree_iter;
  }
  template <typename IterType, typename CheckerIterType>
  IterType riter_check(
      IterType tree_iter, CheckerIterType checker_iter) const {
    if (tree_iter == tree_.rend()) {
      EXPECT_EQ(checker_iter, checker_.rend());
    } else {
      EXPECT_EQ(*tree_iter, *checker_iter);
    }
    return tree_iter;
  }
  void value_check(const value_type &x) {
    typename KeyOfValue<typename TreeType::key_type,
        typename TreeType::value_type>::type key_of_value;
    const key_type &key = key_of_value(x);
    EXPECT_EQ(*find(key), x);
    lower_bound(key);
    upper_bound(key);
    equal_range(key);
    count(key);
  }
  void erase_check(const key_type &key) {
    EXPECT_TRUE(tree_.find(key) == const_tree_.end());
    EXPECT_TRUE(const_tree_.find(key) == tree_.end());
    EXPECT_TRUE(tree_.equal_range(key).first ==
                const_tree_.equal_range(key).second);
  }


  iterator lower_bound(const key_type &key) {
    return iter_check(tree_.lower_bound(key), checker_.lower_bound(key));
  }
  const_iterator lower_bound(const key_type &key) const {
    return iter_check(tree_.lower_bound(key), checker_.lower_bound(key));
  }
  iterator upper_bound(const key_type &key) {
    return iter_check(tree_.upper_bound(key), checker_.upper_bound(key));
  }
  const_iterator upper_bound(const key_type &key) const {
    return iter_check(tree_.upper_bound(key), checker_.upper_bound(key));
  }
  std::pair<iterator,iterator> equal_range(const key_type &key) {
    std::pair<typename CheckerType::iterator,
        typename CheckerType::iterator> checker_res =
        checker_.equal_range(key);
    std::pair<iterator, iterator> tree_res = tree_.equal_range(key);
    iter_check(tree_res.first, checker_res.first);
    iter_check(tree_res.second, checker_res.second);
    return tree_res;
  }
  std::pair<const_iterator,const_iterator> equal_range(const key_type &key) const {
    std::pair<typename CheckerType::const_iterator,
        typename CheckerType::const_iterator> checker_res =
        checker_.equal_range(key);
    std::pair<const_iterator, const_iterator> tree_res = tree_.equal_range(key);
    iter_check(tree_res.first, checker_res.first);
    iter_check(tree_res.second, checker_res.second);
    return tree_res;
  }
  iterator find(const key_type &key) {
    return iter_check(tree_.find(key), checker_.find(key));
  }
  const_iterator find(const key_type &key) const {
    return iter_check(tree_.find(key), checker_.find(key));
  }
  size_type count(const key_type &key) const {
    size_type res = checker_.count(key);
    EXPECT_EQ(res, tree_.count(key));
    return res;
  }


  self_type& operator=(const self_type &x) {
    tree_ = x.tree_;
    checker_ = x.checker_;
    return *this;
  }


  int erase(const key_type &key) {
    int size = tree_.size();
    int res = checker_.erase(key);
    EXPECT_EQ(res, tree_.count(key));
    EXPECT_EQ(res, tree_.erase(key));
    EXPECT_EQ(tree_.count(key), 0);
    EXPECT_EQ(tree_.size(), size - res);
    erase_check(key);
    return res;
  }
  iterator erase(iterator iter) {
    key_type key = iter.key();
    int size = tree_.size();
    int count = tree_.count(key);
    typename CheckerType::iterator checker_iter = checker_.find(key);
    for (iterator tmp(tree_.find(key)); tmp != iter; ++tmp) {
      ++checker_iter;
    }
    typename CheckerType::iterator checker_next = checker_iter;
    ++checker_next;
    checker_.erase(checker_iter);
    iter = tree_.erase(iter);
    EXPECT_EQ(tree_.size(), checker_.size());
    EXPECT_EQ(tree_.size(), size - 1);
    EXPECT_EQ(tree_.count(key), count - 1);
    if (count == 1) {
      erase_check(key);
    }
    return iter_check(iter, checker_next);
  }

  void erase(iterator begin, iterator end) {
    int size = tree_.size();
    int count = distance(begin, end);
    typename CheckerType::iterator checker_begin = checker_.find(begin.key());
    for (iterator tmp(tree_.find(begin.key())); tmp != begin; ++tmp) {
      ++checker_begin;
    }
    typename CheckerType::iterator checker_end =
        end == tree_.end() ? checker_.end() : checker_.find(end.key());
    if (end != tree_.end()) {
      for (iterator tmp(tree_.find(end.key())); tmp != end; ++tmp) {
        ++checker_end;
      }
    }
    checker_.erase(checker_begin, checker_end);
    tree_.erase(begin, end);
    EXPECT_EQ(tree_.size(), checker_.size());
    EXPECT_EQ(tree_.size(), size - count);
  }


  void clear() {
    tree_.clear();
    checker_.clear();
  }
  void swap(self_type &x) {
    tree_.swap(x.tree_);
    checker_.swap(x.checker_);
  }

  void verify() const {
    tree_.verify();
    EXPECT_EQ(tree_.size(), checker_.size());


    typename CheckerType::const_iterator
        checker_iter(checker_.begin());
    const_iterator tree_iter(tree_.begin());
    for (; tree_iter != tree_.end();
         ++tree_iter, ++checker_iter) {
      EXPECT_EQ(*tree_iter, *checker_iter);
    }


    for (int n = tree_.size() - 1; n >= 0; --n) {
      iter_check(tree_iter, checker_iter);
      --tree_iter;
      --checker_iter;
    }
    EXPECT_TRUE(tree_iter == tree_.begin());
    EXPECT_TRUE(checker_iter == checker_.begin());


    typename CheckerType::const_reverse_iterator
        checker_riter(checker_.rbegin());
    const_reverse_iterator tree_riter(tree_.rbegin());
    for (; tree_riter != tree_.rend();
         ++tree_riter, ++checker_riter) {
      EXPECT_EQ(*tree_riter, *checker_riter);
    }


    for (int n = tree_.size() - 1; n >= 0; --n) {
      riter_check(tree_riter, checker_riter);
      --tree_riter;
      --checker_riter;
    }
    EXPECT_EQ(tree_riter, tree_.rbegin());
    EXPECT_EQ(checker_riter, checker_.rbegin());
  }


  const TreeType& tree() const { return tree_; }


  size_type size() const {
    EXPECT_EQ(tree_.size(), checker_.size());
    return tree_.size();
  }
  size_type max_size() const { return tree_.max_size(); }
  bool empty() const {
    EXPECT_EQ(tree_.empty(), checker_.empty());
    return tree_.empty();
  }
  size_type height() const { return tree_.height(); }
  size_type internal_nodes() const { return tree_.internal_nodes(); }
  size_type leaf_nodes() const { return tree_.leaf_nodes(); }
  size_type nodes() const { return tree_.nodes(); }
  size_type bytes_used() const { return tree_.bytes_used(); }
  double fullness() const { return tree_.fullness(); }
  double overhead() const { return tree_.overhead(); }

 protected:
  TreeType tree_;
  const TreeType &const_tree_;
  CheckerType checker_;
};



template <typename TreeType, typename CheckerType>
class unique_checker : public base_checker<TreeType, CheckerType> {
  typedef base_checker<TreeType, CheckerType> super_type;
  typedef unique_checker<TreeType, CheckerType> self_type;

 public:
  typedef typename super_type::iterator iterator;
  typedef typename super_type::value_type value_type;

 public:

  unique_checker()
      : super_type() {
  }

  unique_checker(const self_type &x)
      : super_type(x) {
  }

  template <class InputIterator>
  unique_checker(InputIterator b, InputIterator e)
      : super_type(b, e) {
  }


  std::pair<iterator,bool> insert(const value_type &x) {
    int size = this->tree_.size();
    std::pair<typename CheckerType::iterator,bool> checker_res =
        this->checker_.insert(x);
    std::pair<iterator,bool> tree_res = this->tree_.insert(x);
    EXPECT_EQ(*tree_res.first, *checker_res.first);
    EXPECT_EQ(tree_res.second, checker_res.second);
    EXPECT_EQ(this->tree_.size(), this->checker_.size());
    EXPECT_EQ(this->tree_.size(), size + tree_res.second);
    return tree_res;
  }
  iterator insert(iterator position, const value_type &x) {
    int size = this->tree_.size();
    std::pair<typename CheckerType::iterator,bool> checker_res =
        this->checker_.insert(x);
    iterator tree_res = this->tree_.insert(position, x);
    EXPECT_EQ(*tree_res, *checker_res.first);
    EXPECT_EQ(this->tree_.size(), this->checker_.size());
    EXPECT_EQ(this->tree_.size(), size + checker_res.second);
    return tree_res;
  }
  template <typename InputIterator>
  void insert(InputIterator b, InputIterator e) {
    for (; b != e; ++b) {
      insert(*b);
    }
  }
};




template <typename TreeType, typename CheckerType>
class multi_checker : public base_checker<TreeType, CheckerType> {
  typedef base_checker<TreeType, CheckerType> super_type;
  typedef multi_checker<TreeType, CheckerType> self_type;

 public:
  typedef typename super_type::iterator iterator;
  typedef typename super_type::value_type value_type;

 public:

  multi_checker()
      : super_type() {
  }

  multi_checker(const self_type &x)
      : super_type(x) {
  }

  template <class InputIterator>
  multi_checker(InputIterator b, InputIterator e)
      : super_type(b, e) {
  }


  iterator insert(const value_type &x) {
    int size = this->tree_.size();
    typename CheckerType::iterator checker_res = this->checker_.insert(x);
    iterator tree_res = this->tree_.insert(x);
    EXPECT_EQ(*tree_res, *checker_res);
    EXPECT_EQ(this->tree_.size(), this->checker_.size());
    EXPECT_EQ(this->tree_.size(), size + 1);
    return tree_res;
  }
  iterator insert(iterator position, const value_type &x) {
    int size = this->tree_.size();
    typename CheckerType::iterator checker_res = this->checker_.insert(x);
    iterator tree_res = this->tree_.insert(position, x);
    EXPECT_EQ(*tree_res, *checker_res);
    EXPECT_EQ(this->tree_.size(), this->checker_.size());
    EXPECT_EQ(this->tree_.size(), size + 1);
    return tree_res;
  }
  template <typename InputIterator>
  void insert(InputIterator b, InputIterator e) {
    for (; b != e; ++b) {
      insert(*b);
    }
  }
};

char* GenerateDigits(char buf[16], int val, int maxval) {
  EXPECT_LE(val, maxval);
  int p = 15;
  buf[p--] = 0;
  while (maxval > 0) {
    buf[p--] = '0' + (val % 10);
    val /= 10;
    maxval /= 10;
  }
  return buf + p + 1;
}

template <typename K>
struct Generator {
  int maxval;
  Generator(int m)
      : maxval(m) {
  }
  K operator()(int i) const {
    EXPECT_LE(i, maxval);
    return i;
  }
};

template <>
struct Generator<std::string> {
  int maxval;
  Generator(int m)
      : maxval(m) {
  }
  std::string operator()(int i) const {
    char buf[16];
    return GenerateDigits(buf, i, maxval);
  }
};

template <typename T, typename U>
struct Generator<std::pair<T, U> > {
  Generator<typename std::remove_const<T>::type> tgen;
  Generator<typename std::remove_const<U>::type> ugen;

  Generator(int m)
      : tgen(m),
        ugen(m) {
  }
  std::pair<T, U> operator()(int i) const {
    return std::make_pair(tgen(i), ugen(i));
  }
};


const std::vector<int>& GenerateNumbers(int n, int maxval) {
  static std::vector<int> values;
  static std::set<int> unique_values;

  if (values.size() < n) {

    for (int i = values.size(); i < n; i++) {
      int value;
      do {
        value = rand() % (maxval + 1);
      } while (unique_values.find(value) != unique_values.end());

      values.push_back(value);
      unique_values.insert(value);
    }
  }

  return values;
}



template <typename V>
std::vector<V> GenerateValues(int n) {
  int two_times_max = 2 * std::max(FLAGS_benchmark_values, FLAGS_test_values);
  int four_times_max = 2 * two_times_max;
  EXPECT_LE(n, two_times_max);
  const std::vector<int> &nums = GenerateNumbers(n, four_times_max);
  Generator<V> gen(four_times_max);
  std::vector<V> vec;

  for (int i = 0; i < n; i++) {
    vec.push_back(gen(nums[i]));
  }

  return vec;
}

template <typename T, typename V>
void DoTest(const char *name, T *b, const std::vector<V> &values) {
  typename KeyOfValue<typename T::key_type, V>::type key_of_value;

  T &mutable_b = *b;
  const T &const_b = *b;


  for (int i = 0; i < values.size(); ++i) {
    mutable_b.insert(values[i]);
    mutable_b.value_check(values[i]);
  }
  assert(mutable_b.size() == values.size());

  const_b.verify();
  printf("    %s fullness=%0.2f  overhead=%0.2f  bytes-per-value=%0.2f\n",
         name, const_b.fullness(), const_b.overhead(),
         double(const_b.bytes_used()) / const_b.size());


  T b_copy(const_b);
  EXPECT_EQ(b_copy.size(), const_b.size());
  EXPECT_LE(b_copy.height(), const_b.height());
  EXPECT_LE(b_copy.internal_nodes(), const_b.internal_nodes());
  EXPECT_LE(b_copy.leaf_nodes(), const_b.leaf_nodes());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(*b_copy.find(key_of_value(values[i])), values[i]);
  }


  T b_range(const_b.begin(), const_b.end());
  EXPECT_EQ(b_range.size(), const_b.size());
  EXPECT_LE(b_range.height(), const_b.height());
  EXPECT_LE(b_range.internal_nodes(), const_b.internal_nodes());
  EXPECT_LE(b_range.leaf_nodes(), const_b.leaf_nodes());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(*b_range.find(key_of_value(values[i])), values[i]);
  }


  b_range.insert(b_copy.begin(), b_copy.end());
  b_range.verify();


  b_range.clear();
  b_range.insert(b_copy.begin(), b_copy.end());
  EXPECT_EQ(b_range.size(), b_copy.size());
  EXPECT_EQ(b_range.height(), b_copy.height());
  EXPECT_EQ(b_range.internal_nodes(), b_copy.internal_nodes());
  EXPECT_EQ(b_range.leaf_nodes(), b_copy.leaf_nodes());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(*b_range.find(key_of_value(values[i])), values[i]);
  }


  b_range.operator=(b_range);
  EXPECT_EQ(b_range.size(), b_copy.size());
  EXPECT_EQ(b_range.height(), b_copy.height());
  EXPECT_EQ(b_range.internal_nodes(), b_copy.internal_nodes());
  EXPECT_EQ(b_range.leaf_nodes(), b_copy.leaf_nodes());


  b_range.clear();
  b_range = b_copy;
  EXPECT_EQ(b_range.size(), b_copy.size());
  EXPECT_EQ(b_range.height(), b_copy.height());
  EXPECT_EQ(b_range.internal_nodes(), b_copy.internal_nodes());
  EXPECT_EQ(b_range.leaf_nodes(), b_copy.leaf_nodes());


  b_range.clear();
  b_range.swap(b_copy);
  EXPECT_EQ(b_copy.size(), 0);
  EXPECT_EQ(b_range.size(), const_b.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(*b_range.find(key_of_value(values[i])), values[i]);
  }
  b_range.swap(b_copy);


  for (int i = 0; i < values.size(); ++i) {
    mutable_b.erase(key_of_value(values[i]));

    EXPECT_EQ(mutable_b.erase(key_of_value(values[i])), 0);
  }

  const_b.verify();
  EXPECT_EQ(const_b.internal_nodes(), 0);
  EXPECT_EQ(const_b.leaf_nodes(), 0);
  EXPECT_EQ(const_b.size(), 0);


  mutable_b = b_copy;
  for (int i = 0; i < values.size(); ++i) {
    mutable_b.erase(mutable_b.find(key_of_value(values[i])));
  }

  const_b.verify();
  EXPECT_EQ(const_b.internal_nodes(), 0);
  EXPECT_EQ(const_b.leaf_nodes(), 0);
  EXPECT_EQ(const_b.size(), 0);


  for (int i = 0; i < values.size(); i++) {
    mutable_b.insert(mutable_b.upper_bound(key_of_value(values[i])), values[i]);
  }

  const_b.verify();



  std::stringstream strm;
  strm << mutable_b.tree();
  EXPECT_EQ(mutable_b.size(), strcount(strm.str(), '\n'));


  mutable_b.erase(mutable_b.begin(), mutable_b.end());
  EXPECT_EQ(mutable_b.size(), 0);
  const_b.verify();


  mutable_b = b_copy;
  typename T::iterator mutable_iter_end = mutable_b.begin();
  for (int i = 0; i < values.size() / 2; ++i) ++mutable_iter_end;
  mutable_b.erase(mutable_b.begin(), mutable_iter_end);
  EXPECT_EQ(mutable_b.size(), values.size() - values.size() / 2);
  const_b.verify();


  mutable_b = b_copy;
  typename T::iterator mutable_iter_begin = mutable_b.begin();
  for (int i = 0; i < values.size() / 2; ++i) ++mutable_iter_begin;
  mutable_b.erase(mutable_iter_begin, mutable_b.end());
  EXPECT_EQ(mutable_b.size(), values.size() / 2);
  const_b.verify();


  mutable_b = b_copy;
  mutable_iter_begin = mutable_b.begin();
  for (int i = 0; i < values.size() / 4; ++i) ++mutable_iter_begin;
  mutable_iter_end = mutable_iter_begin;
  for (int i = 0; i < values.size() / 4; ++i) ++mutable_iter_end;
  mutable_b.erase(mutable_iter_begin, mutable_iter_end);
  EXPECT_EQ(mutable_b.size(), values.size() - values.size() / 4);
  const_b.verify();

  mutable_b.clear();
}

template <typename T>
void ConstTest() {
  typedef typename T::value_type value_type;
  typename KeyOfValue<typename T::key_type, value_type>::type key_of_value;

  T mutable_b;
  const T &const_b = mutable_b;


  value_type value = Generator<value_type>(2)(2);
  mutable_b.insert(value);
  EXPECT_TRUE(mutable_b.find(key_of_value(value)) != const_b.end());
  EXPECT_TRUE(const_b.find(key_of_value(value)) != mutable_b.end());
  EXPECT_EQ(*const_b.lower_bound(key_of_value(value)), value);
  EXPECT_TRUE(const_b.upper_bound(key_of_value(value)) == const_b.end());
  EXPECT_EQ(*const_b.equal_range(key_of_value(value)).first, value);


  typename T::iterator mutable_iter(mutable_b.begin());
  EXPECT_TRUE(mutable_iter == const_b.begin());
  EXPECT_TRUE(mutable_iter != const_b.end());
  EXPECT_TRUE(const_b.begin() == mutable_iter);
  EXPECT_TRUE(const_b.end() != mutable_iter);
  typename T::reverse_iterator mutable_riter(mutable_b.rbegin());
  EXPECT_TRUE(mutable_riter == const_b.rbegin());
  EXPECT_TRUE(mutable_riter != const_b.rend());
  EXPECT_TRUE(const_b.rbegin() == mutable_riter);
  EXPECT_TRUE(const_b.rend() != mutable_riter);


  typename T::const_iterator const_iter(mutable_iter);
  EXPECT_TRUE(const_iter == mutable_b.begin());
  EXPECT_TRUE(const_iter != mutable_b.end());
  EXPECT_TRUE(mutable_b.begin() == const_iter);
  EXPECT_TRUE(mutable_b.end() != const_iter);
  typename T::const_reverse_iterator const_riter(mutable_riter);
  EXPECT_EQ(const_riter, mutable_b.rbegin());
  EXPECT_TRUE(const_riter != mutable_b.rend());
  EXPECT_EQ(mutable_b.rbegin(), const_riter);
  EXPECT_TRUE(mutable_b.rend() != const_riter);


  const_b.verify();
  EXPECT_FALSE(const_b.empty());
  EXPECT_EQ(const_b.size(), 1);
  EXPECT_GT(const_b.max_size(), 0);
  EXPECT_EQ(const_b.height(), 1);
  EXPECT_EQ(const_b.count(key_of_value(value)), 1);
  EXPECT_EQ(const_b.internal_nodes(), 0);
  EXPECT_EQ(const_b.leaf_nodes(), 1);
  EXPECT_EQ(const_b.nodes(), 1);
  EXPECT_GT(const_b.bytes_used(), 0);
  EXPECT_GT(const_b.fullness(), 0);
  EXPECT_GT(const_b.overhead(), 0);
}

template <typename T, typename C>
void BtreeTest() {
  ConstTest<T>();

  typedef typename std::remove_const<typename T::value_type>::type V;
  std::vector<V> random_values = GenerateValues<V>(FLAGS_test_values);

  unique_checker<T, C> container;


  std::vector<V> sorted_values(random_values);
  sort(sorted_values.begin(), sorted_values.end());
  DoTest("sorted:    ", &container, sorted_values);


  reverse(sorted_values.begin(), sorted_values.end());
  DoTest("rsorted:   ", &container, sorted_values);


  DoTest("random:    ", &container, random_values);
}

template <typename T, typename C>
void BtreeMultiTest() {
  ConstTest<T>();

  typedef typename std::remove_const<typename T::value_type>::type V;
  const std::vector<V>& random_values = GenerateValues<V>(FLAGS_test_values);

  multi_checker<T, C> container;


  std::vector<V> sorted_values(random_values);
  sort(sorted_values.begin(), sorted_values.end());
  DoTest("sorted:    ", &container, sorted_values);


  reverse(sorted_values.begin(), sorted_values.end());
  DoTest("rsorted:   ", &container, sorted_values);


  DoTest("random:    ", &container, random_values);


  std::vector<V> duplicate_values(random_values);
  duplicate_values.insert(
      duplicate_values.end(), random_values.begin(), random_values.end());
  DoTest("duplicates:", &container, duplicate_values);


  std::vector<V> identical_values(100);
  fill(identical_values.begin(), identical_values.end(), Generator<V>(2)(2));
  DoTest("identical: ", &container, identical_values);
}

template <typename T, typename Alloc = std::allocator<T> >
class TestAllocator : public Alloc {
 public:
  typedef typename Alloc::pointer pointer;
  typedef typename Alloc::size_type size_type;

  TestAllocator() : bytes_used_(NULL) { }
  TestAllocator(int64_t *bytes_used) : bytes_used_(bytes_used) { }


  template <class U>
  TestAllocator(const TestAllocator<U>& x)
      : Alloc(x),
        bytes_used_(x.bytes_used()) {
  }

  pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0) {
    EXPECT_TRUE(bytes_used_ != NULL);
    *bytes_used_ += n * sizeof(T);
    return Alloc::allocate(n, hint);
  }

  void deallocate(pointer p, size_type n) {
    Alloc::deallocate(p, n);
    EXPECT_TRUE(bytes_used_ != NULL);
    *bytes_used_ -= n * sizeof(T);
  }


  template <class U> struct rebind {
    typedef TestAllocator<U, typename Alloc::template rebind<U>::other> other;
  };

  int64_t* bytes_used() const { return bytes_used_; }

 private:
  int64_t *bytes_used_;
};

template <typename T>
void BtreeAllocatorTest() {
  typedef typename T::value_type value_type;

  int64_t alloc1 = 0;
  int64_t alloc2 = 0;
  T b1(typename T::key_compare(), &alloc1);
  T b2(typename T::key_compare(), &alloc2);


  swap(b1, b2);

  for (int i = 0; i < 1000; i++) {
    b1.insert(Generator<value_type>(1000)(i));
  }


  EXPECT_LE(b1.bytes_used(), alloc2 + sizeof(b1));
  EXPECT_GT(alloc2, alloc1);
}

template <typename T>
void BtreeMapTest() {
  typedef typename T::value_type value_type;
  typedef typename T::mapped_type mapped_type;

  mapped_type m = Generator<mapped_type>(0)(0);
  (void) m;

  T b;


  for (int i = 0; i < 1000; i++) {
    value_type v = Generator<value_type>(1000)(i);
    b[v.first] = v.second;
  }
  EXPECT_EQ(b.size(), 1000);




  EXPECT_EQ(b.begin()->first, Generator<value_type>(1000)(0).first);
  EXPECT_EQ(b.begin()->second, Generator<value_type>(1000)(0).second);
  EXPECT_EQ(b.rbegin()->first, Generator<value_type>(1000)(999).first);
  EXPECT_EQ(b.rbegin()->second, Generator<value_type>(1000)(999).second);
}

template <typename T>
void BtreeMultiMapTest() {
  typedef typename T::mapped_type mapped_type;
  mapped_type m = Generator<mapped_type>(0)(0);
  (void) m;
}

}

#endif
