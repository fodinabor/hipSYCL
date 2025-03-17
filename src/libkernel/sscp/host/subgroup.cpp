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
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/sycl/libkernel/host/rv.h"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/RV.h"

extern "C" size_t __acpp_cbs_local_id_subgroup;
extern "C" size_t __acpp_cbs_id_subgroup;
extern "C" size_t __acpp_cbs_subgroup_size;
extern "C" size_t __acpp_cbs_num_subgroups;

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_local_id() {
#if USE_RV
  return rv_lane_id();
#else
  return __acpp_cbs_local_id_subgroup;
#endif
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_size() {
#if USE_RV
  return rv_num_lanes();
#else
  return __acpp_cbs_subgroup_size;
#endif
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_max_size() {
  return hipsycl::compiler::SGSize;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_id() {
  #if USE_RV
    return rv_is_uniform(__acpp_cbs_id_subgroup);
  #else
    return __acpp_cbs_id_subgroup;
  #endif
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups() {
  return __acpp_cbs_num_subgroups;
}
