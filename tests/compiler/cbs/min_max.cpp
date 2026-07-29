// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-use-accelerated-cpu -O3
// RUN: ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  constexpr size_t local_size = 1024;
  constexpr size_t global_size = 1024;

  std::vector<int> host_buf(global_size*2, 0);
  std::iota(host_buf.begin(), host_buf.end(), 0);
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
            std::pair<int, size_t> v{acc[item.get_global_id()], item.get_global_id()};

            auto eval = [](std::pair<int, size_t> v1, std::pair<int, size_t> v2) {
                    if (v1.first > v2.first || v1.first == v2.first && v1.second < v2.second)   {
                            return v1;
                    }
                    return v2;
            };

            for (int id = item.get_global_id(); id < item.get_global_range()[0]*2; id += sg.get_local_linear_range()) {
              v = eval(v, {acc[id], id});
            }

            for (auto i = sg.get_local_linear_range() >> 1; i > 0; i >>= 1) {
             int shftl_val = cl::sycl::shift_group_left(sg, v.first, i);
             size_t shftl_index = cl::sycl::shift_group_left(sg, v.second, i);
                         v = eval(v, {shftl_val, shftl_index});
            }

            acc[item.get_global_id()] = v.first;
          });
    });
  }

  // CHECK: 2047
  std::cout << host_buf[0] << "\n";
  // CHECK: 2047
  std::cout << host_buf[32] << "\n";
}