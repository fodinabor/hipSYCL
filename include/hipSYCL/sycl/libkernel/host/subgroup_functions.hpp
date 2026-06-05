#ifndef SUBGROUP_FUNCTIONS_HPP
#define SUBGROUP_FUNCTIONS_HPP

#include "group_functions.hpp"
#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../detail/mem_fence.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"

namespace hipsycl {
namespace sycl {

template <typename T> HIPSYCL_FORCE_INLINE int dot_product(sub_group sg, T a, T b) {
  T c = a*b;
  return detail::host_builtins::__acpp_reduce_over_group(sg, c, plus<T>{});
}

template <typename T, typename Pred> HIPSYCL_FORCE_INLINE int find_if(sub_group sg, T a, Pred pred) {
  T *scratch = static_cast<T *>(sg.get_local_memory_ptr());
  scratch[sg.get_local_linear_id()] = a;
  detail::host_builtins::__acpp_group_barrier(sg);
  int j = -1;
  if (sg.leader()) {
    for (auto i = 0; i < sg.get_local_linear_range(); ++i) {
      if (pred(scratch[i])) {
        j = i;
        break;
      }
    }
  }
  return detail::host_builtins::__acpp_group_broadcast(sg, j, 0);
}

template <typename T, typename Pred> HIPSYCL_FORCE_INLINE uint32_t count_if(sub_group sg, T a, Pred pred) {
  uint32_t x = pred(a) ? 1 : 0;
  return detail::host_builtins::__acpp_reduce_over_group(sg, x, sycl::plus<uint32_t>{});
/*
  T *scratch = static_cast<T *>(sg.get_local_memory_ptr());
  scratch[sg.get_local_linear_id()] = a;
  detail::host_builtins::__acpp_group_barrier(sg);
  int j = 0;
  if (sg.leader()) {
    for (auto i = 0; i < sg.get_local_linear_range(); ++i) {
      if (pred(scratch[i])) {
        ++j;
      }
    }
  }
  return detail::host_builtins::__acpp_group_broadcast(sg, j, 0);
*/
}

template <typename T, typename Pred> HIPSYCL_FORCE_INLINE uint32_t partition(sub_group sg, T& a, Pred pred) {
  T *scratch = static_cast<T *>(sg.get_local_memory_ptr());
  scratch[sg.get_local_linear_id()] = a;

  detail::host_builtins::__acpp_group_barrier(sg);
  int j = 0;
  if (sg.leader()) {
    for (auto i = 0; i < sg.get_local_linear_range(); ++i) {
      if (pred(scratch[i])) {
          std::swap(scratch[i], scratch[j]);
          ++j;
      }
    }
  }
  detail::host_builtins::__acpp_group_barrier(sg);
  a = scratch[sg.get_local_linear_id()];
  detail::host_builtins::__acpp_group_barrier(sg);
  return detail::host_builtins::__acpp_group_broadcast(sg, j, 0);
}


}
}

#endif //SUBGROUP_FUNCTIONS_HPP
