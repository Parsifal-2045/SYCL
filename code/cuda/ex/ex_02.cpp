#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

int main()
{
    int numBlocks = 8;
    int numThreadsPerBlock = 8;
    int N = numBlocks * numThreadsPerBlock;
    std::vector<int> h_a(N); 
    size_t memSize = N * sizeof(int);
    auto queue = sycl::queue{sycl::default_selector{}};
    auto d_a = sycl::malloc_device<int>(N, queue);
   
    queue.memcpy(d_a, h_a.data(), memSize);
    queue.submit([&](sycl::handler &cgh)
    {
        cgh.parallel_for(sycl::range<2>(numBlocks, numThreadsPerBlock), [=](sycl::id<2> id) 
        { 
            auto index = id[0] * numThreadsPerBlock + id[1];
            d_a[index] = id[0] + id[1] + 42;
        });
    }).wait();

    queue.memcpy(h_a.data(), d_a, memSize).wait();
    sycl::free(d_a, queue);

    for(int i = 0; i != numBlocks; i++)
    {
        for(int j = 0; j != numThreadsPerBlock; j++)
        {
            assert(h_a[i * numThreadsPerBlock + j] == i + j + 42);
        }
    }

    std::cout << "Correct!" << '\n';
}