/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_GLUE_HOST_SEQUENTIAL_REDUCER_HPP
#define HIPSYCL_GLUE_HOST_SEQUENTIAL_REDUCER_HPP

#include <algorithm>
#include <new>
#include <memory>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"

namespace hipsycl {
namespace glue {
namespace host {

#ifndef HIPSYCL_FORCE_CACHE_LINE_SIZE
#define HIPSYCL_FORCE_CACHE_LINE_SIZE 128
#endif

#ifdef HIPSYCL_FORCE_CACHE_LINE_SIZE
constexpr std::size_t cache_line_size = HIPSYCL_FORCE_CACHE_LINE_SIZE;
#else
// This C++17 feature is unfortunately not yet widely supported
constexpr std::size_t cache_line_size =
    std::hardware_destructive_interference_size;
#endif

template <class T> struct cache_line_aligned {
  alignas(cache_line_size) T value;
};

template<class ReductionDescriptor>
class sequential_reducer {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using combiner_type = typename ReductionDescriptor::combiner_type;

  sequential_reducer(int num_threads, size_t wg_size, ReductionDescriptor &desc)
      : _desc{desc}
      , _num_threads{num_threads}
      , _wg_size{wg_size}
      , _per_thread_results{std::make_unique<
            std::unique_ptr<cache_line_aligned<value_type>[]>[]>(num_threads)} {
    for (int t = 0; t < num_threads; ++t) {
      _per_thread_results[t] =
          std::make_unique<cache_line_aligned<value_type>[]>(wg_size);
      std::fill_n(_per_thread_results[t].get(), wg_size,
                  cache_line_aligned<value_type>{identity()});
    }
  }

  value_type identity() const { return _desc.identity; }

  void combine(int my_thread_id, size_t wi_id, const value_type& v) {
    assert(my_thread_id < _num_threads);
    _per_thread_results[my_thread_id][wi_id].value =
        _desc.combiner(_per_thread_results[my_thread_id][wi_id].value, v);
  }

  // This should be executed in a single threaded scope.
  // Sums up all the partial results and stores in the result data buffer
  void finalize_result() {
    for (std::size_t t = 0; t < _num_threads; ++t) {
      for (std::size_t w = 0; w < _wg_size; ++w){
        if(t != 0 || w != 0)
          _per_thread_results[0][0].value = _desc.combiner(
              _per_thread_results[0][0].value, _per_thread_results[t][w].value);
      }
    }
    
    *(_desc.get_pointer()) = _per_thread_results[0][0].value;
  }
private:
  ReductionDescriptor &_desc;
  int _num_threads;
  size_t _wg_size;
  // TODO: new does not necessarily respect over-aligned alignas requirements.
  // Depending on the value of std::max_align_t and cache_line_size,
  // alignment may be off and not match cache lines.
  std::unique_ptr<std::unique_ptr<cache_line_aligned<value_type>[]>[]> _per_thread_results;

};

}
}
}

#endif
