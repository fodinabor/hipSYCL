// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <CL/sycl.hpp>
#include <iostream>

void multiply(size_t n, const std::vector<uint32_t>& A, const std::vector<uint32_t>& B, std::vector<uint32_t>& Out) {
    auto idx = [n](auto x, auto y) {
        return x + n * y;
    };
    for (auto x = 0ul; x < n; ++x) {
        for (auto y = 0ul; y < n; ++y) {
            Out[idx(x, y)] = 0;
            for (auto k = 0ul; k < n; ++k) {
                Out[idx(x, y)] += A[idx(k, y)] * B[idx(x, k)];
            }
        }
    }
}

int main() {
    constexpr size_t local_size = 32*32;
    constexpr size_t global_size = 32*32;

    std::vector<uint32_t> matrixA;
    std::vector<uint32_t> matrixB;
    std::vector<uint32_t> matrixC;
    std::vector<uint32_t> matrixCheck;
    for (auto i = 0; i < local_size; i++) {
        matrixA.push_back(1);
        matrixB.push_back(1);
        matrixC.push_back(0);
        matrixCheck.push_back(0);
    }

    cl::sycl::queue queue;
    {
        cl::sycl::buffer<uint32_t, 2> bufA{matrixA.data(), cl::sycl::range<2>{32, 32}};
        cl::sycl::buffer<uint32_t, 2> bufB{matrixB.data(), cl::sycl::range<2>{32, 32}};
        cl::sycl::buffer<uint32_t, 2> bufC{matrixC.data(), cl::sycl::range<2>{32, 32}};

        queue.submit([&](cl::sycl::handler &cgh) {
          using namespace cl::sycl::access;
          using namespace cl::sycl::access;
          auto matrixA = bufA.get_access<mode::read_write>(cgh);
          auto matrixB = bufB.get_access<mode::read_write>(cgh);
          auto matrixC = bufC.get_access<mode::read_write>(cgh);

          auto K = 32;
          auto t = 32;



          cgh.parallel_for<class dynamic_local_memory_reduction>(
              cl::sycl::nd_range<2>{{32, 32}, {32, 32}}, [=](cl::sycl::nd_item<2> item) noexcept {
                const auto sg = item.get_sub_group();
                auto m = item.get_local_id()[0];
                auto n = item.get_local_id()[1];

                uint32_t sum = 0;
                for (auto kk = 0ul; kk < K; kk += t) {
                    auto tile = matrixA[m][kk + sg.get_local_linear_id()];
                    for (auto k = 0ul; k < t; ++k) {
                        sum += cl::sycl::group_broadcast(sg, tile, k) * matrixB[kk + k][n];
                    }
                }

                matrixC[m][n] = sum;
              });
        });
    }

    multiply(32, matrixA, matrixB, matrixCheck);

    bool x = true;
    for (size_t i = 0; i < 32*32; ++i) {
        if ( matrixCheck[i] != matrixC[i]) {
          std::cout << matrixC[i] << "\n";
        }
        x &= matrixCheck[i] == matrixC[i];
    }

    // CHECK: true
    std::cout << (x ? "true" : "false") << "\n";
}