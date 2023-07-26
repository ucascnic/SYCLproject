//==---- vector.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_VECTOR_H__
#define __DPCT_VECTOR_H__


#include <CL/sycl.hpp>

#include "memory.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "../device.hpp"
#include "memory.h"

namespace dpct {

namespace internal {

template <typename Iter, typename Void = void> // for non-iterators
struct is_iterator : std::false_type {};

template <typename Iter> // For iterators
struct is_iterator<
    Iter,
    typename std::enable_if<
        !std::is_void<typename Iter::iterator_category>::value, void>::type>
    : std::true_type {};

template <typename T> // For pointers
struct is_iterator<T *> : std::true_type {};

} // namespace internal


template <typename T,
          typename Allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>>
class device_vector {
public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

private:
  Allocator _alloc;
  size_type _size;
  size_type _capacity;
  pointer _storage;

  size_type _min_capacity() const { return size_type(1); }

public:
  template <typename OtherA> operator const std::vector<T, OtherA>() & {
    auto __tmp = std::vector<T, OtherA>(this->size());
    //std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
    //          this->begin(), this->end(), __tmp.begin());
    get_default_queue().memcpy(__tmp.begin().get(), this->begin().get(), this->size()).wait();
    return __tmp;
  }
  device_vector()
      : _alloc(get_default_queue()), _size(0), _capacity(_min_capacity()) {
    _storage = _alloc.allocate(_capacity);
  }
  ~device_vector() /*= default*/ { _alloc.deallocate(_storage, _capacity); };
  explicit device_vector(size_type n) : _alloc(get_default_queue()), _size(n) {
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
  }
  explicit device_vector(size_type n, const T &value)
      : _alloc(get_default_queue()), _size(n) {
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
    //std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()),
    //          begin(), end(), T(value));
    get_default_queue().fill(begin().get(), T(value), end()-begin()).wait();
  }
  device_vector(const device_vector &other) : _alloc(get_default_queue()) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = _alloc.allocate(_capacity);
    //std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
    //          other.begin(), other.end(), begin());
    get_default_queue().copy(other.begin().get(), begin().get(), other.end() - other.begin()).wait();
  }
  device_vector(device_vector &&other)
      : _alloc(get_default_queue()), _size(other.size()),
        _capacity(other.capacity()) {}
  template <typename OtherAllocator>
  device_vector(const device_vector<T, OtherAllocator> &v)
      : _alloc(get_default_queue()), _storage(v.real_begin()), _size(v.size()),
        _capacity(v.capacity()) {}
#if 1
  template <typename OtherAllocator>
  device_vector(std::vector<T, OtherAllocator> &v)
      : _alloc(get_default_queue()), _size(v.size()) {
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
    //std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
    //          v.begin(), v.end(), this->begin());
    get_default_queue().memcpy(this->begin().get(), v.data(), v.size()).wait();
  }

  template <typename OtherAllocator>
  device_vector &operator=(const std::vector<T, OtherAllocator> &v) {
    resize(v.size());
    //std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
    //          v.begin(), v.end(), begin());
    get_default_queue().memcpy(begin().get(), v.data(), v.size()).wait();
    return *this;
  }
#endif
  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    resize(other.size());
    //std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
    //          other.begin(), other.end(), begin());
    get_default_queue().memcpy(begin().get(), other.begin().get(), other.size()).wait();
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    this->_size = std::move(other._size);
    this->_capacity = std::move(other._capacity);
    this->_storage = std::move(other._storage);
    return *this;
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(_storage, 0); }
  iterator end() { return device_iterator<T>(_storage, size()); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(_storage, 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(_storage, size()); }
  const_iterator cend() const { return end(); }
  T *real_begin() { return _storage; }
  const T *real_begin() const { return _storage; }
  void swap(device_vector &v) {
    auto temp = std::move(v._storage);
    v._storage = std::move(this->_storage);
    this->_storage = std::move(temp);
    std::swap(_size, v._size);
    std::swap(_capacity, v._capacity);
  }
  reference operator[](size_type n) { return _storage[n]; }
  const_reference operator[](size_type n) const { return _storage[n]; }
  void reserve(size_type n) {
    if (n > capacity()) {
      // allocate buffer for new size
      auto tmp = _alloc.allocate(2 * n);
      // copy content (old buffer to new buffer)
      //std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()), begin(), end(), tmp);
      get_default_queue().memcpy(tmp, begin().get(), _capacity).wait();
      // deallocate old memory
      _alloc.deallocate(_storage, _capacity);
      _storage = tmp;
      _capacity = 2 * n;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (_size < new_size) {
      //std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()), begin() + _size, begin() + new_size, x);
      get_default_queue().fill(begin().get()+_size, x, new_size-_size).wait();
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const { return _capacity; }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return _storage; }
  const_pointer data(void) const { return _storage; }
 void clear(void) { _size = 0; }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
 Allocator get_allocator() const { return _alloc; }
};


} // namespace dpct

#endif // __DPCT_VECTOR_H__
