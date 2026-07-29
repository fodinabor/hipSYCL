// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-use-accelerated-cpu -O3
// RUN: ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for (size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch =
          cl::sycl::accessor<int, 1, mode::read_write, target::local>{32, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
          cl::sycl::nd_range<1>{global_size, local_size}, [=](cl::sycl::nd_item<1> item) noexcept {
            const auto g = item.get_group();

            auto val = acc[item.get_global_id()];
            acc[item.get_global_id()] = cl::sycl::inclusive_scan_over_group(g, val, cl::sycl::plus<int>());
          });
    });
  }

  // CHECK: 0
  std::cout << host_buf[0] << "\n";
  // CHECK: 1
  std::cout << host_buf[1] << "\n";
  // CHECK: 3
  std::cout << host_buf[2] << "\n";
  // CHECK: 6
  std::cout << host_buf[3] << "\n";
  // CHECK: 32640
  std::cout << host_buf[255] << "\n";

  // CHECK: 256
  std::cout << host_buf[256] << "\n";
  // CHECK: 513
  std::cout << host_buf[257] << "\n";
  // CHECK: 771
  std::cout << host_buf[258] << "\n";
}