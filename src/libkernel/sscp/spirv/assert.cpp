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

template <typename... Args>
extern int __spirv_ocl_printf(const char *Format, Args... args);

static const char assert_fail_string[] =
    "[AdaptiveCpp][spirv] device assertion '%s' failed in file '%s', function '%s', "
    "line %d (kernel abortion is unsupported on this target)\n";

static const char glibcxx_assert_fail_string[] =
    "[AdaptiveCpp][spirv] GLIBCXX device assertion '%s' failed in file '%s', "
    "function '%s', "
    "line '%d' (kernel abortion is unsupported on this target)\n";

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_assert_fail(const char *assertion,
                                                  const char *file,
                                                  __acpp_uint32 line,
                                                  const char *function) {
  // It seems that printf is incorrectly handled by Intel CPU OpenCL
  // let's ignore for now.
  //__spirv_ocl_printf(assert_fail_string, assertion, file, function, line);
}

HIPSYCL_SSCP_BUILTIN void
__acpp_sscp_glibcxx_assert_fail(const char *file, __acpp_int32 line,
                                const char *function, const char *assertion) {
  // It seems that printf is incorrectly handled by Intel CPU OpenCL
  // let's ignore for now.
  //__spirv_ocl_printf(glibcxx_assert_fail_string, assertion, file, function, line);
}
