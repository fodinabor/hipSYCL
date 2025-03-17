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


#ifndef ACPP_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP
#define ACPP_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../detail/mem_fence.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include "hipSYCL/RV.h"
#include "cbs_intrinsics.h"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include <type_traits>
#include "rv_shuffle.h"

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST

namespace hipsycl {
namespace sycl::detail::host_builtins {

// barrier
template <int Dim>
HIPSYCL_LOOP_SPLIT_BARRIER ACPP_KERNEL_TARGET inline void
__acpp_group_barrier(group<Dim> g,
                        memory_scope fence_scope = group<Dim>::fence_scope) {
  if (fence_scope == memory_scope::device) {
    mem_fence<>();
  }
  g.barrier();
}

[[clang::annotate("hipsycl_sub_barrier")]] __attribute__((noinline))
ACPP_KERNEL_TARGET inline void
__acpp_group_barrier(sub_group g, memory_scope fence_scope = sub_group::fence_scope) {
  // doesn't need sync
}

namespace detail {
// reduce implementation
template <int Dim, typename T, typename BinaryOperation>
ACPP_KERNEL_TARGET T __acpp_group_reduce(group<Dim> g, T x, BinaryOperation binary_op,
                                            T *scratch) {
  const auto lid = g.get_local_linear_id();
  const std::size_t local_range = g.get_local_linear_range();
  sub_group sg = g.get_sub_group();

  T result = reduce_over_group(sg, x, binary_op);

  // Collect results from sub-groups
  if (sg.leader()) {
    scratch[sg.get_group_linear_id()] = result;
  }

  __acpp_group_barrier(g); // as this starts a new loop with CBS,
  // LLVM detects it's actually just one iteration and optimizes it

  // First work-item does reduction on the results of the sub-group reductions
  if (g.leader()) {
    T y = scratch[0];
    for (uint32_t j = 1u; j < sg.get_group_linear_range(); j += 1ul) {
      y = binary_op(y, scratch[j]);
    }
    scratch[0] = y;
  }

  __acpp_group_barrier(g);
  T endResult = scratch[0];
  __acpp_group_barrier(g);

  return endResult;
}

} // namespace detail

template <int Dim, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T __acpp_reduce_over_group(group<Dim> g, T x, BinaryOperation binary_op) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  T tmp = detail::__acpp_group_reduce(g, x, binary_op, scratch);

  return tmp;
}

// broadcast
template <int Dim, typename T>
ACPP_KERNEL_TARGET 
T __acpp_group_broadcast(
    group<Dim> g, T x,
    typename group<Dim>::linear_id_type local_linear_id = 0) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid = g.get_local_linear_id();

  if (lid == local_linear_id) {
    scratch[0] = x;
  }

  __acpp_group_barrier(g);
  T tmp = scratch[0];
  __acpp_group_barrier(g);

  return tmp;
}

template <int Dim, typename T>
ACPP_KERNEL_TARGET T __acpp_group_broadcast(
    group<Dim> g, T x, typename group<Dim>::id_type local_id) {
  const size_t target_lid =
      linear_id<g.dimensions>::get(local_id, g.get_local_range());
  return __acpp_group_broadcast(g, x, target_lid);
}

template <typename T>
ACPP_KERNEL_TARGET T
__acpp_group_broadcast(sub_group g, T x, typename sub_group::linear_id_type local_linear_id = 0) {
  static_assert(std::is_trivially_copyable_v<T>);
#if USE_RV
  return extract_impl(x, local_linear_id);
#else
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    if (g.get_local_linear_id() == local_linear_id)
      scratch[0] = x;

    __acpp_group_barrier(g);
    T y = scratch[0];
    __acpp_group_barrier(g);
    return y;
#endif
}

template <typename T>
ACPP_KERNEL_TARGET T __acpp_group_broadcast(sub_group g, T x,
                                               typename sub_group::id_type local_id) {
  return __acpp_group_broadcast(g, x, local_id);
}

// any_of
namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET bool __acpp_leader_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = false;

  if (g.leader()) {
    while (first != last) {
      if (pred(*(first++))) {
        result = true;
        break;
      }
    }
  }
  return result;
}
} // namespace detail

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET bool __acpp_joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const bool result = detail::__acpp_leader_any_of(g, first, last, pred);
  return group_broadcast(g, result);
}

template <int Dim> ACPP_KERNEL_TARGET inline bool __acpp_any_of_group(group<Dim> g, bool pred) {
  return __acpp_reduce_over_group(g, static_cast<uint8_t>(pred), sycl::maximum<uint8_t>{}) > 0;
}

ACPP_KERNEL_TARGET
inline bool __acpp_any_of_group(sub_group g, bool pred) {
#if USE_RV
  return rv_any(pred);
#else
  auto val = static_cast<uint8_t>(pred);
  __acpp_group_barrier(g);
  const bool res = __cbs_reduce(val, static_cast<int>(ReduceOp::MAX)) > 0;
  __acpp_group_barrier(g);
  return res;
#endif
}

// all_of
namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET bool __acpp_leader_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (!pred(*(first++))) {
        result = false;
        break;
      }
    }
  }
  return result;
}
} // namespace detail

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET bool __acpp_joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const bool result = detail::__acpp_leader_all_of(g, first, last, pred);
  return group_broadcast(g, result);
}

template <int Dim> ACPP_KERNEL_TARGET inline bool __acpp_all_of_group(group<Dim> g, bool pred) {
  return __acpp_reduce_over_group(g, static_cast<uint8_t>(pred), sycl::minimum<uint8_t>{}) > 0;
}

ACPP_KERNEL_TARGET
inline bool __acpp_all_of_group(sub_group g, bool pred) {
#if USE_RV
  return rv_all(pred);
#else
  auto val = static_cast<uint8_t>(pred);
  __acpp_group_barrier(g);
  const bool res = __cbs_reduce(val, static_cast<int>(ReduceOp::MIN)) > 0;
  __acpp_group_barrier(g);
  return res;
#endif
}

// none_of
namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET bool __acpp_leader_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (pred(*(first++))) {
        result = false;
        break;
      }
    }
  }
  return result;
}
} // namespace detail

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET bool __acpp_joint_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  auto result = detail::__acpp_leader_none_of(g, first, last, pred);
  return group_broadcast(g, result);
}

template<int Dim>
ACPP_KERNEL_TARGET
inline bool __acpp_none_of_group(group<Dim> g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = true;
  __acpp_group_barrier(g);

  if (pred)
    scratch[0] = false;

  __acpp_group_barrier(g);

  bool tmp = scratch[0];

  __acpp_group_barrier(g);

  return tmp;
}

ACPP_KERNEL_TARGET
inline bool __acpp_none_of_group(sub_group g, bool pred) {
  return __acpp_all_of_group(g, not pred);
}

// reduce
namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET
T __acpp_leader_reduce(Group g, T *first, T *last, 
                          BinaryOperation binary_op) {
  T result{};

  if (first >= last) {
    return T{};
  }

  if (g.leader()) {
#pragma omp simd
    for (T *i = first; i < last; ++i)
      result = binary_op(result, *i);
  }
  return result;
}

template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET
T __acpp_leader_reduce(Group g, T *first, T *last, V init, 
                          BinaryOperation binary_op) {
  auto result = __acpp_leader_reduce(g, first, last, binary_op);

  if (g.leader()) {
    result = binary_op(result, init);
  }
  return result;
}
} // namespace detail

template <typename T>
HIPSYCL_KERNEL_TARGET T __acpp_shift_group_left(sub_group g, T x,
                                                typename sub_group::linear_id_type delta = 1) {
#if USE_RV
  return shuffle_down_impl(x, delta);
#else
  if constexpr (std::is_integral_v<T> && USE_CBS_SHUFFLE) {
    __acpp_group_barrier(g);
    const auto pos = g.get_local_linear_id() + delta >= g.get_local_range().size()
                         ? 0
                         : g.get_local_linear_id() + delta;
    auto tmp = __cbs_shuffle(x, pos);
    __acpp_group_barrier(g);
    return tmp;
  } else {
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    auto lid = g.get_local_linear_id();
    auto target_lid = lid + delta;

    scratch[lid] = x;
    __acpp_group_barrier(g);

    if (target_lid >= g.get_local_range().size())
      target_lid = 0;

    T tmp = scratch[target_lid];
    __acpp_group_barrier(g);
    return tmp;
  }
#endif
}

template <typename T>
HIPSYCL_KERNEL_TARGET T __acpp_shift_group_right(sub_group g, T x,
                                                 typename sub_group::linear_id_type delta = 1) {
#if USE_RV
  return shuffle_up_impl(x, delta);
#else
  if constexpr (std::is_integral_v<T> && USE_CBS_SHUFFLE) {
    __acpp_group_barrier(g);
    const auto pos = g.get_local_linear_id() - delta >= g.get_local_range().size()
                         ? g.get_max_local_range().size()
                         : g.get_local_linear_id() - delta;
    auto tmp = __cbs_shuffle(x, pos);
    __acpp_group_barrier(g);
    return tmp;
  } else {
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    auto lid = g.get_local_linear_id();
    auto target_lid = lid - delta;

    scratch[lid] = x;
    __acpp_group_barrier(g);

    if (target_lid > g.get_local_range().size() || target_lid < 0)
      target_lid = 0;

    T tmp = scratch[target_lid];
    __acpp_group_barrier(g);
    return tmp;
  }
#endif
}

template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET
typename std::iterator_traits<Ptr>::value_type
__acpp_joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  const auto result = detail::__acpp_leader_reduce(g, first, last, binary_op);

  return group_broadcast(g, result);
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET
T __acpp_joint_reduce(Group g, Ptr first, Ptr last, T init,
                         BinaryOperation binary_op) {
  const auto result =
      detail::__acpp_leader_reduce(g, first, last, init, binary_op);

  return __acpp_group_broadcast(g, result);
}


template <typename BinaryOperation, typename T> constexpr int reduce_supported_op() {
  if constexpr (std::is_same_v<BinaryOperation, sycl::plus<T>>) {
    return 0;
  } else if constexpr (std::is_same_v<BinaryOperation, sycl::multiplies<T>>) {
    return 1;
  } else if constexpr (std::is_same_v<BinaryOperation, sycl::minimum<T>>) {
    return 2;
  } else if constexpr (std::is_same_v<BinaryOperation, sycl::maximum<T>>) {
    return 3;
  }
  return -1;
}

template <typename T, typename BinaryOperation>
ACPP_KERNEL_TARGET T __acpp_reduce_over_group(sub_group g, T x, BinaryOperation binary_op) {
  static_assert(std::is_fundamental_v<T>);
  constexpr int op = reduce_supported_op<BinaryOperation, T>();
#if USE_RV
  if constexpr (op >= 0 && USE_REDUCE_INTRINSIC) {
    return rv_reduce(x, op);
  } else {
    auto local_x = x;
#pragma unroll
    for (auto i = g.get_local_linear_range() / 2; i > 0; i /= 2) {
      auto other_x = __acpp_shift_group_left(g, local_x, i);
      local_x = binary_op(local_x, other_x);
    }
    return __acpp_group_broadcast(g, local_x, 0);
  }
#else
  if constexpr (op >= 0 && USE_REDUCE_INTRINSIC) {
    __acpp_group_barrier(g);
    T tmp = __cbs_reduce(x, op);
    __acpp_group_barrier(g);
    return tmp;
  } else {

 	T *scratch = static_cast<T *>(g.get_local_memory_ptr());
	scratch[g.get_local_linear_id()] = x;
	__acpp_group_barrier(g);

	if (g.leader()) {
		for (auto i = 1ul; i < g.get_local_linear_range(); ++i) {
			scratch[0] = binary_op(scratch[0], scratch[i]);
		}
	}

	__acpp_group_barrier(g);
	auto result = scratch[0];
	__acpp_group_barrier(g);
	return result;
  }
#endif
}

// exclusive_scan
namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr __acpp_leader_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                                                      BinaryOperation binary_op) {

  if (g.leader()) {
    *(result++) = init;
    while (first != last - 1) {
      *result = binary_op(*(result - 1), *(first++));
      ++result;
    }
  }
  return result;
}

template <typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET
OutPtr __acpp_leader_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                                   BinaryOperation binary_op) {
  using value_type = std::remove_reference_t<decltype(*result)>;
  return __acpp_leader_exclusive_scan(g, first, last, result, value_type{},
                                         binary_op);
}
} // namespace detail

template <typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr
__acpp_joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               T init, BinaryOperation binary_op) {
  const auto ret = detail::__acpp_leader_exclusive_scan(
      g, first, last, result, init, binary_op);
  return group_broadcast(g, ret);
}

template <typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr
__acpp_joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op) {
  using value_type = std::remove_reference_t<decltype(*result)>;
  return host_builtins::__acpp_joint_exclusive_scan(
      g, first, last, result, value_type{},
      binary_op);
}

template <int Dim, typename V, typename T, typename BinaryOperation>
ACPP_KERNEL_TARGET T __acpp_exclusive_scan_over_group(
    group<Dim> g, V x, T init, BinaryOperation binary_op) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  if (lid + 1 < 1024)
    scratch[lid + 1] = x;
  __acpp_group_barrier(g);

  if (g.leader()) {
    scratch[0] = init;
    for (int i = 1; i < g.get_local_range().size(); ++i)
      scratch[i] = binary_op(scratch[i], scratch[i - 1]);
  }

  __acpp_group_barrier(g);
  T tmp = scratch[lid];
  __acpp_group_barrier(g);

  return tmp;
}

template <typename V, typename T, typename BinaryOperation>
ACPP_KERNEL_TARGET T __acpp_exclusive_scan_over_group(
    sub_group g, V x, T init, BinaryOperation binary_op) {
  return binary_op(x, init);
}

template <typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET T
__acpp_exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  return __acpp_exclusive_scan_over_group(g, x, T{}, binary_op);
}

// inclusive_scan
namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr
__acpp_leader_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                                BinaryOperation binary_op, T init) {
  if (first == last)
    return result;

  if (g.leader()) {
    *(result++) = binary_op(init, *(first++));
    while (first != last) {
      *result = binary_op(*(result - 1), *(first++));
      ++result;
    }
  }
  return result;
}

template <typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr
__acpp_leader_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                                BinaryOperation binary_op) {
  using value_type = std::remove_reference_t<decltype(*result)>;
  return __acpp_leader_inclusive_scan(g, first, last, result, binary_op, value_type{});
}
} // namespace detail

template <typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr
__acpp_joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op, T init) {
  auto ret = detail::__acpp_leader_inclusive_scan(g, first, last, result,
                                                     binary_op, init);
  return __acpp_group_broadcast(g, ret);
}

template <typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET OutPtr
__acpp_joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op) {
  using value_type = std::remove_reference_t<decltype(*result)>;
  return __acpp_joint_inclusive_scan(
      g, first, last, result, binary_op,
      value_type{});
}

template <typename T, typename BinaryOperation>
ACPP_KERNEL_TARGET T __acpp_inclusive_scan_over_group(sub_group g, T x,
                                                         BinaryOperation binary_op) {
#if USE_RV
  const size_t lid = g.get_local_linear_id();
  const size_t lrange = g.get_local_linear_range();
  auto local_x = x;
#pragma unroll
  for (size_t i = 1ul; i < lrange; i *= 2) {
    auto other_x = __acpp_shift_group_right(g, local_x, i);
    if (i <= lid)
      local_x = binary_op(local_x, other_x);
  }
  return local_x;
#else
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid = g.get_local_linear_id();

  scratch[lid] = x;
  __acpp_group_barrier(g);

  if (g.leader()) {
    for (auto i = 1ul; i < g.get_local_range().size(); ++i)
      scratch[i] = binary_op(scratch[i], scratch[i - 1ul]);
  }

  __acpp_group_barrier(g);
  T tmp = scratch[lid];
  __acpp_group_barrier(g);

  return tmp;
#endif
}

template <int Dim, typename T, typename BinaryOperation>
ACPP_KERNEL_TARGET T __acpp_inclusive_scan_over_group(group<Dim> g, T x,
                                                         BinaryOperation binary_op) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());
  auto sg = g.get_sub_group();
  size_t sgId = sg.get_group_linear_id();

  x = __acpp_inclusive_scan_over_group(sg, x, binary_op);

  // Last work-item in sub-group
  if (sg.get_local_linear_id() + 1ul == sg.get_local_linear_range()) {
    // sg group id
    scratch[sgId] = x;
  }
  __acpp_group_barrier(g);

  if (g.leader()) {
    for (auto i = 1ul; i < sg.get_group_linear_range()-1; ++i) {
      scratch[i] = binary_op(scratch[i - 1], scratch[i]);
    }
  }
  __acpp_group_barrier(g);

  // Not the first sub-group
  const auto glid = sg.get_group_linear_id();
  if (glid >= 1) {
    x = binary_op(scratch[glid-1], x);
  }
  __acpp_group_barrier(g);
  return x;
}


template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
ACPP_KERNEL_TARGET T __acpp_inclusive_scan_over_group(
    Group g, V x, T init, BinaryOperation binary_op) {
  T scan = __acpp_inclusive_scan_over_group(g, T{x}, binary_op);
  return binary_op(scan, init);
}

// shift_left
template <int Dim, typename T>
ACPP_KERNEL_TARGET
T __acpp_shift_group_left(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid + delta;

  scratch[lid] = x;
  __acpp_group_barrier(g);

  if (target_lid >= g.get_local_range().size())
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_group_barrier(g);

  return x;
}

// shift_right
template <int Dim, typename T>
ACPP_KERNEL_TARGET T __acpp_shift_group_right(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid - delta;

  scratch[lid] = x;
  __acpp_group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid >= g.get_local_range().size() || target_lid < 0)
    target_lid = 0;
  // TODO SHIFT RIGHT 0 => I have implemented it with 31

  x = scratch[target_lid];
  __acpp_group_barrier(g);

  return x;
}

// permute_group_by_xor
template <int Dim, typename T>
ACPP_KERNEL_TARGET T __acpp_permute_group_by_xor(
    group<Dim> g, T x, typename group<Dim>::linear_id_type mask) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid ^ mask;

  scratch[lid] = x;
  __acpp_group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_group_barrier(g);

  return x;
}

// permute_group_by_xor
template <typename T>
ACPP_KERNEL_TARGET T __acpp_permute_group_by_xor(
    sub_group g, T x, typename sub_group::linear_id_type mask) {
  return x;
}

// select_from_group
template <int Dim, typename T>
ACPP_KERNEL_TARGET T __acpp_select_from_group(
    group<Dim> g, T x, typename group<Dim>::id_type remote_local_id) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid =
      linear_id<g.dimensions>::get(remote_local_id, g.get_local_range());

  scratch[lid] = x;
  __acpp_group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_group_barrier(g);

  return x;
}

template <typename T>
ACPP_KERNEL_TARGET T __acpp_select_from_group(sub_group g, T x,
                                                 typename sub_group::id_type remote_local_id) {
#if USE_RV
  return shuffle_impl(x, remote_local_id);
#else
  if constexpr (std::is_integral_v<T> && USE_CBS_SHUFFLE) {
    __acpp_group_barrier(g);
    auto tmp = __cbs_shuffle(x, remote_local_id);
    __acpp_group_barrier(g);
    return tmp;
  } else {
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    auto lid = g.get_local_linear_id();
    auto target_lid = remote_local_id;

    scratch[lid] = x;
    __acpp_group_barrier(g);

    if (target_lid > g.get_local_range().size() || target_lid < 0)
      target_lid = 0;

    x = scratch[target_lid];
    __acpp_group_barrier(g);

    return x;
  }
#endif
}

} // namespace sycl::detail::host_builtins
} // namespace hipsycl

#endif

#endif // ACPP_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

