// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-use-accelerated-cpu -O3
// RUN: ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  constexpr size_t local_size = 32;
  constexpr size_t global_size = 32;

  std::vector<int> host_buf(global_size, 2);
  cl::sycl::queue queue;
  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 1, mode::read_write, target::local>{local_size, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
          cl::sycl::nd_range<1>{global_size, local_size}, [=](cl::sycl::nd_item<1> item) noexcept {
            const auto g = item.get_group();
            const auto sg = item.get_sub_group();

            int hello = sg.get_local_linear_id();
            cl::sycl::group_barrier(sg);
            hello = sg.get_local_linear_id();
            //std::cout << sg.get_local_linear_id() << "\n";
            acc[item.get_global_id()] = hello;
          });
    });
  }
  for (size_t i = 28; i < 33; ++i) {
    // CHECK: 28
    // CHECK: 29
    // CHECK: 30
    // CHECK: 31
    // CHECK: 0
    std::cout << host_buf[i] << "\n";
  }
}
