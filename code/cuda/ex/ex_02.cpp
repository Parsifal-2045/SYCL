#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

int main()
{
    int N = 1024;
    std::vector<int> h_a(N); 
    size_t memSize = N * sizeof(int);
    auto queue = sycl::queue{sycl::default_selector{}};
    auto d_a = sycl::malloc_device<int>(N, queue);
    queue.memcpy(d_a, h_a.data(), memSize);
    queue.submit([&](sycl::handler &cgh)
    {
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) 
        { 
            auto i = idx[0];
            d_a[i] = i + 42;
        });
    }).wait();

    queue.memcpy(h_a.data(), d_a, memSize).wait();

    for(int i = 0; i != N; i++)
    {
        assert(h_a[i] == i + 42);
    }

    std::cout << "Correct!" << '\n';
}