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

#include "hipSYCL/sycl/libkernel/sscp/builtins/assert.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include <cstdlib>
#include <cstdio>

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_assert_fail(const char *assertion,
                                                  const char *file,
                                                  __acpp_uint32 line,
                                                  const char *function) {
  printf("[AdaptiveCpp][host] device assertion '%s' failed in file '%s', "
         "function '%s', line '%d'\n",
         assertion, file, function, line);
  std::abort();
  __builtin_unreachable();
}

HIPSYCL_SSCP_BUILTIN void
__acpp_sscp_glibcxx_assert_fail(const char *file, __acpp_int32 line,
                                const char *function, const char *assertion) {
  printf("[AdaptiveCpp][host] GLIBCXX device assertion %s failed in file '%s', "
         "function '%s', line '%d'\n",
         assertion, file, function, line);
  std::abort();
  __builtin_unreachable();
}
