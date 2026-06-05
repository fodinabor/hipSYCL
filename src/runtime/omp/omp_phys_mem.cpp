/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/runtime/omp/omp_phys_mem.hpp"

// Mac OSX implementation:
#if defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>

std::optional<std::size_t> hipsycl::rt::get_physical_memory() {
  int mib[] = {CTL_HW, HW_MEMSIZE};
  int64_t value = 0;
  size_t length = sizeof(value);

  if (-1 == sysctl(mib, 2, &value, &length, NULL, 0)) {
    return std::nullopt;
  }
  return value;
}

#elif _WIN32

#include <windows.h>

std::optional<std::size_t> hipsycl::rt::get_physical_memory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}

#else

// Linux/BSD implementation, if not just return std::nullopt
#ifdef __has_include
#if __has_include(<sys/sysinfo.h>)
#include <sys/sysinfo.h>
#endif
#endif

std::optional<std::size_t> hipsycl::rt::get_physical_memory() {
#ifdef __has_include
#if __has_include(<sys/sysinfo.h>)
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram;
#endif
#endif
  return std::nullopt;
}

#endif