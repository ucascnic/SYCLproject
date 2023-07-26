//==---- device.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DEVICE_HPP__
#define __DPCT_DEVICE_HPP__

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <map>
#include <vector>
#include <thread>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif
#if defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif


namespace dpct {

/// SYCL default exception handler
auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
  }
};

/// dpct device extension
class device_ext : public cl::sycl::device {
public:
  device_ext() : cl::sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto &task : _tasks) {
      if (task.joinable())
        task.join();
    }
    _tasks.clear();
    _queues.clear();
  }
  device_ext(const cl::sycl::device &base)
      : cl::sycl::device(base), _ctx(*this) {
    _queues.push_back(std::make_shared<cl::sycl::queue>(
        _ctx, base, exception_handler, cl::sycl::property::queue::in_order()));
    _saved_queue = _default_queue = _queues[0].get();
  }

  cl::sycl::queue &default_queue() { return *_default_queue; }

  cl::sycl::context get_context() const { return _ctx; }

private:

  cl::sycl::queue *_default_queue;
  cl::sycl::queue *_saved_queue;
  cl::sycl::context _ctx;
  std::vector<std::shared_ptr<cl::sycl::queue>> _queues;
  mutable std::mutex m_mutex;
  std::vector<std::thread> _tasks;
};

static inline unsigned int get_tid() {
#if defined(__linux__)
  return syscall(SYS_gettid);
#elif defined(_WIN64)
  return GetCurrentThreadId();
#else
#error "Only support Windows and Linux."
#endif
}

/// device manager
class dev_mgr {

public:

  device_ext &current_device() {
    unsigned int dev_id=current_device_id();
    check_id(dev_id);
    return *_devs[dev_id];
  }

  unsigned int current_device_id() const {
   std::lock_guard<std::mutex> lock(m_mutex);
   auto it=_thread2dev_map.find(get_tid());
   if(it != _thread2dev_map.end())
      return it->second;
    return DEFAULT_DEVICE_ID;
  }

  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  mutable std::mutex m_mutex;
  dev_mgr() {
    cl::sycl::device default_device =
        cl::sycl::device(cl::sycl::default_selector{});
    _devs.push_back(std::make_shared<device_ext>(default_device));

    std::vector<cl::sycl::device> sycl_all_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::all);
    // Collect other devices except for the default device.
    if (default_device.is_cpu())
      _cpu_device = 0;
    for (auto &dev : sycl_all_devs) {
      if (dev == default_device) {
        continue;
      }
      _devs.push_back(std::make_shared<device_ext>(dev));
      if (_cpu_device == -1 && dev.is_cpu()) {
        _cpu_device = _devs.size() - 1;
      }
    }
  }

  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }

  std::vector<std::shared_ptr<device_ext>> _devs;
  /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const unsigned int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<unsigned int, unsigned int> _thread2dev_map;
  int _cpu_device = -1;
};

/// Util function to get the default queue of current device in
/// dpct device manager.
static inline cl::sycl::queue &get_default_queue() {
  return dev_mgr::instance().current_device().default_queue();
}

/// Util function to get the current device.
static inline device_ext &get_current_device() {
  return dev_mgr::instance().current_device();
}

/// Util function to get the context of the default queue of current
/// device in dpct device manager.
static inline cl::sycl::context get_default_context() {
  return dpct::get_current_device().get_context();
}

} // namespace dpct

#endif // __DPCT_DEVICE_HPP__
