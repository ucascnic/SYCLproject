//==---- memory.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MEMORY_H__
#define __DPCT_MEMORY_H__

#include "../device.hpp"
#include "../memory.hpp"

#include <CL/sycl.hpp>

// Memory management section:
// device_pointer, device_reference, swap, device_iterator, malloc_device,
// device_new, free_device, device_delete

namespace dpct {

template <typename T> class device_pointer;

template <typename T> struct device_reference {
  using pointer = device_pointer<T>;
  using value_type = T;
  template <typename OtherT>
  device_reference(const device_reference<OtherT> &input)
      : value(input.value) {}
  device_reference(const pointer &input) : value((*input).value) {}
  device_reference(value_type &input) : value(input) {}
  template <typename OtherT>
  device_reference &operator=(const device_reference<OtherT> &input) {
    value = input;
    return *this;
  };
  device_reference &operator=(const device_reference &input) {
    T val = input.value;
    value = val;
    return *this;
  };
  device_reference &operator=(const value_type &x) {
    value = x;
    return *this;
  };
  pointer operator&() const { return pointer(&value); };
  operator value_type() const { return T(value); }
  device_reference &operator++() {
    ++value;
    return *this;
  };
  device_reference &operator--() {
    --value;
    return *this;
  };
  device_reference operator++(int) {
    device_reference ref(*this);
    ++(*this);
    return ref;
  };
  device_reference operator--(int) {
    device_reference ref(*this);
    --(*this);
    return ref;
  };
  device_reference &operator+=(const T &input) {
    value += input;
    return *this;
  };
  device_reference &operator-=(const T &input) {
    value -= input;
    return *this;
  };
  device_reference &operator*=(const T &input) {
    value *= input;
    return *this;
  };
  device_reference &operator/=(const T &input) {
    value /= input;
    return *this;
  };
  device_reference &operator%=(const T &input) {
    value %= input;
    return *this;
  };
  device_reference &operator&=(const T &input) {
    value &= input;
    return *this;
  };
  device_reference &operator|=(const T &input) {
    value |= input;
    return *this;
  };
  device_reference &operator^=(const T &input) {
    value ^= input;
    return *this;
  };
  device_reference &operator<<=(const T &input) {
    value <<= input;
    return *this;
  };
  device_reference &operator>>=(const T &input) {
    value >>= input;
    return *this;
  };
  void swap(device_reference &input) {
    T tmp = (*this);
    *this = (input);
    input = (tmp);
  }
  T &value;
};

namespace internal {

// struct for checking if iterator is heterogeneous or not
template <typename Iter,
          typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : std::false_type {};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<
    Iter, typename std::enable_if<Iter::is_hetero::value, void>::type>
    : std::true_type {};

} // namespace internal

template <typename T> class device_iterator;

template <typename ValueType, typename Derived> class device_pointer_base {
protected:
  ValueType *ptr;

public:
  using pointer = ValueType *;
  using difference_type = std::make_signed<std::size_t>::type;

  device_pointer_base(ValueType *p) : ptr(p) {}
  device_pointer_base(const std::size_t count) {
    cl::sycl::queue default_queue = dpct::get_default_queue();
    ptr = static_cast<ValueType *>(cl::sycl::malloc_device(
        count, default_queue.get_device(), default_queue.get_context()));
  }
  device_pointer_base() {}
  pointer get() const { return ptr; }
  operator ValueType *() { return ptr; }
  operator ValueType *() const { return ptr; }

  ValueType &operator[](difference_type idx) { return ptr[idx]; }
  ValueType &operator[](difference_type idx) const { return ptr[idx]; }

  Derived operator+(difference_type forward) const {
    return Derived{ptr + forward};
  }
  Derived operator-(difference_type backward) const {
    return Derived{ptr - backward};
  }
  Derived operator++(int) {
    Derived p(ptr);
    ++ptr;
    return p;
  }
  Derived operator--(int) {
    Derived p(ptr);
    --ptr;
    return p;
  }
  difference_type operator-(const Derived &it) const { return ptr - it.ptr; }
};

template <typename T>
class device_pointer : public device_pointer_base<T, device_pointer<T>> {
private:
  using base_type = device_pointer_base<T, device_pointer<T>>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(T *p) : base_type(p) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer &operator=(const device_iterator<T> &in) {
    this->ptr = static_cast<device_pointer<T>>(in).ptr;
    return *this;
  }

  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    ++(this->ptr);
    return *this;
  }
  device_pointer &operator--() {
    --(this->ptr);
    return *this;
  }
  device_pointer &operator+=(difference_type forward) {
    this->ptr = this->ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->ptr = this->ptr - backward;
    return *this;
  }
};

template <>
class device_pointer<void>
    : public device_pointer_base<dpct::byte_t, device_pointer<void>> {
private:
  using base_type = device_pointer_base<dpct::byte_t, device_pointer<void>>;

public:
  using value_type = dpct::byte_t;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = void *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(void *p) : base_type(static_cast<value_type *>(p)) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  pointer get() const { return static_cast<pointer>(this->ptr); }
  operator void *() { return this->ptr; }
  operator void *() const { return this->ptr; }

  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    ++(this->ptr);
    return *this;
  }
  device_pointer &operator--() {
    --(this->ptr);
    return *this;
  }
  device_pointer &operator+=(difference_type forward) {
    this->ptr = this->ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->ptr = this->ptr - backward;
    return *this;
  }
};

template <typename T> class device_iterator : public device_pointer<T> {
  using Base = device_pointer<T>;

protected:
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = typename Base::pointer;
  using reference = typename Base::reference;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required
  static constexpr sycl::access_mode mode =
      cl::sycl::access_mode::read_write; // required

  device_iterator() : Base(nullptr), idx(0) {}
  device_iterator(T *vec, std::size_t index) : Base(vec), idx(index) {}
  template <cl::sycl::access_mode inMode>
  device_iterator(const device_iterator<T> &in)
      : Base(in.ptr), idx(in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::operator=(in);
    idx = in.idx;
    return *this;
  }

  reference operator*() const { return *(Base::ptr + idx); }

  reference operator[](difference_type i) { return Base::ptr[idx + i]; }
  reference operator[](difference_type i) const { return Base::ptr[idx + i]; }
  device_iterator &operator++() {
    ++idx;
    return *this;
  }
  device_iterator &operator--() {
    --idx;
    return *this;
  }
  device_iterator operator++(int) {
    device_iterator it(*this);
    ++(*this);
    return it;
  }
  device_iterator operator--(int) {
    device_iterator it(*this);
    --(*this);
    return it;
  }
  device_iterator operator+(difference_type forward) const {
    const auto new_idx = idx + forward;
    return {Base::ptr, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::ptr, idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return idx - it.idx;
  }

  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - it.get_idx();
  }

  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() const { return idx; } // required

  device_iterator &get_buffer() { return *this; } // required

  std::size_t size() const { return idx; }
};

template <typename T> T *get_raw_pointer(const device_pointer<T> &ptr) {
  return ptr.get();
}

template <typename Pointer> Pointer get_raw_pointer(const Pointer &ptr) {
  return ptr;
}

} // namespace dpct

#endif // __DPCT_MEMORY_H__
