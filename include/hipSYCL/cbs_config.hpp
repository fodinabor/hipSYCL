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
#ifndef HIPSYCL_CBS_CONFIG_HPP
#define HIPSYCL_CBS_CONFIG_HPP

// Configuration flags for the (hierarchical) Continuation-Based Synchronization
// host code generation.

// Use the dedicated CBS shuffle intrinsics for sub-group shuffles instead of
// the shared-memory based fallback.
#define USE_CBS_SHUFFLE false
// Use the CBS reduce intrinsic for sub-group reductions instead of the
// shared-memory based fallback.
#define USE_REDUCE_INTRINSIC true
// Optimize handling of incomplete sub-groups (work-group sizes not divisible by
// the sub-group size).
#define INCOMPLETE_SGS_OPT true
// Enable work-group level SSCP optimizations.
#define WG_SSCP_OPT true
// Always create the sub-group sub-CFGs, even if the kernel does not contain
// sub-group barriers.
#define ALWAYS_CREATE_SUBGROUP_SUB_CFGS false

#endif // HIPSYCL_CBS_CONFIG_HPP
