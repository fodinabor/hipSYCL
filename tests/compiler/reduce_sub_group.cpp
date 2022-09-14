// RUN: %syclcc %s -o %t --hipsycl-targets=omp -DHIPSYCL_NO_FIBERS -DHIPSYCL_MANUALLY_CREATED_LOOPS -DHIPSYCL_HAS_CPU_SG
// RUN: %t | FileCheck %s
// RUN: %syclcc %s -o %t --hipsycl-targets=omp -DHIPSYCL_NO_FIBERS -DHIPSYCL_MANUALLY_CREATED_LOOPS -DHIPSYCL_HAS_CPU_SG -O
// RUN: %t | FileCheck %s

#include <iostream>

#include <CL/sycl.hpp>

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
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch =
          cl::sycl::accessor<int, 1, mode::read_write, target::local>{32, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
          cl::sycl::nd_range<1>{global_size, local_size},
          [=](cl::sycl::nd_item<1> item) noexcept {
            const auto g = item.get_group();
            const auto sg = item.get_sub_group();
            const auto lid = item.get_local_id(0);

            // auto val = acc[item.get_global_id()];
            // // for (int offset = 16; offset > 0; offset /= 2)
            // //   val += cl::sycl::shift_group_left(sg, val, offset);
            // val = cl::sycl::group_reduce(sg, val, cl::sycl::plus<int>());

            // if (sg.leader())
            //   scratch[sg.get_group_linear_id()] = val;

            // cl::sycl::group_barrier(g);

            // if (sg.get_group_linear_id() == 0) {
            //   val = scratch[lid];
            //   // for (int offset = 16; offset > 0; offset /= 2)
            //   //   val += cl::sycl::shift_group_left(sg, val, offset);
            //   val = cl::sycl::group_reduce(sg, val, cl::sycl::plus<int>());

            //   if (sg.leader())
            //     acc[sg.get_group_linear_id()] = val;
            // }
            acc[item.get_global_id()] = cl::sycl::group_reduce(item.get_group(), acc[item.get_global_id()], cl::sycl::plus<int>{});
          });
    });
  }
  for (size_t i = 0; i < global_size / local_size; ++i) {
    // CHECK: 32640
    // CHECK: 98176
    // CHECK: 163712
    // CHECK: 229248
    std::cout << host_buf[i * local_size] << "\n";
  }
}

// __global__ void reduce(int *array, int n) {
//   int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   __shared__ int shrd[32];
// #define FULL_MASK 0xffffffff

//   int val = array[tid];
//   for (int offset = 16; offset > 0; offset /= 2)
//     val += __shfl_down_sync(FULL_MASK, val, offset);

//   if (threadIdx.x % 32 == 0)
//     shrd[threadIdx.x / 32] = val;
//   __syncthreads();

//   if (threadIdx.x < 32) {
//     val = shrd[threadIdx.x];
//     for (int offset = 16; offset > 0; offset /= 2)
//       val += __shfl_down_sync(FULL_MASK, val, offset);
//     if (threadIdx.x % 32 == 0)
//       array[blockIdx.x] = val;
//   }
// }