/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#include "CL/sycl/detail/async_worker.hpp"

#include <cassert>

namespace cl {
namespace sycl {
namespace detail {


worker_thread::worker_thread()
  : _is_operation_pending{false},
    _continue{true}
{
  _worker_thread = std::thread{[this](){ work(); } };
}


worker_thread::~worker_thread()
{
  halt();

  if(_worker_thread.joinable())
    _worker_thread.join();

  assert(_enqueued_operations.empty());
}

void worker_thread::wait()
{
  if(_is_operation_pending)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    // Wait until no operation is pending
    _condition_wait.wait(lock, [this]{return !_is_operation_pending;});
  }
}


bool worker_thread::is_currently_working() const
{
  return _is_operation_pending;
}

void worker_thread::halt()
{
  _continue = false;
  _condition_wait.notify_one();
}

void worker_thread::work()
{
  // This is the main function executed by the worker thread.
  // The loop is executed as long as there are enqueued operations,
  // (_is_operation_pending) or we should wait for new operations
  // (_continue).
  while(_continue || _is_operation_pending)
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);

      // Wait until we have work, or until _continue becomes false
      _condition_wait.wait(lock,
                           [this](){return _is_operation_pending || !_continue;});
    }

    // In any way, process the pending operations
    if(_is_operation_pending)
    {
      async_function operation = [](){};

      {
        std::lock_guard<std::mutex> lock(_mutex);

        if(!_enqueued_operations.empty())
        {
          operation = _enqueued_operations.front();
          _enqueued_operations.pop();
        }
      }

      operation();

      {
        std::lock_guard<std::mutex> lock(_mutex);
        if(_enqueued_operations.empty())
          _is_operation_pending = false;
      }
      _condition_wait.notify_one();
    }
  }
}

void worker_thread::operator()(worker_thread::async_function f)
{
  std::unique_lock<std::mutex> lock(_mutex);

  _is_operation_pending = true;
  _enqueued_operations.push(f);

  lock.unlock();
  _condition_wait.notify_one();
}

}
}
}
