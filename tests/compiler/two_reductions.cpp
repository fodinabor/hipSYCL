// RUN: %syclcc %s -o %t --opensycl-targets=omp --opensycl-use-accelerated-cpu 
// RUN: %t | FileCheck %s
// RUN: %syclcc %s -o %t --opensycl-targets=omp --opensycl-use-accelerated-cpu -O3
// RUN: %t | FileCheck %s

#include <array>
#include <clang-c/Index.h>
#include <functional>
#include <iostream>

#include <SYCL/sycl.hpp>
#include <numeric>

int main() {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  using T = unsigned int;

  sycl::queue queue;
  T *input0 = sycl::malloc_shared<T>(global_size, queue);
  T *input1 = sycl::malloc_shared<T>(global_size, queue);
  input1[global_size - 1] = 2;
  T *output0 = sycl::malloc_shared<T>(1, queue);
  T *output1 = sycl::malloc_shared<T>(1, queue);
  *output1 = 1;

  std::iota(input0, input0 + global_size, 0);
  std::fill(input1, input1 + global_size, 1);

  queue.parallel_for(
      sycl::nd_range(sycl::range{global_size}, sycl::range{local_size}),
      sycl::reduction(output0, T{0}, sycl::plus<T>{}),
      sycl::reduction(output1, T{1}, sycl::multiplies<T>{}),
      [=](sycl::nd_item<1> idx, auto &add_reducer, auto &mul_reducer) {
        add_reducer += input0[idx.get_global_linear_id()];
        mul_reducer *= input1[idx.get_global_linear_id()];
      }).wait();

  // CHECK: 523776
  // CHECK: 1
  std::cout << *output0 << "\n";
  std::cout << *output1 << "\n";
}
