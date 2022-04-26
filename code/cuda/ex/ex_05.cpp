#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include "../../Intel_gpu_selector.hpp"

int main()
{
    const int num_elements = 16;
    const int threads = 4;
    const int blocks = num_elements / threads;
    std::vector<int> h_input(num_elements);
    auto queue = sycl::queue{Intel_gpu_selector{}};
    for (int i = 0; i != num_elements; i++)
    {
        h_input[i] = i;
    }
    const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);
    
    size_t memSize = num_elements * sizeof(int);
    auto d_input = sycl::malloc_device<int>(num_elements, queue);
    auto sdata = sycl::malloc_shared<int>(threads, queue);
    auto d_result = sycl::malloc_device<int>(1, queue);
    queue.memcpy(d_input, h_input.data(), memSize).wait();

    queue.submit([&](sycl::handler &cgh)
    {
        cgh.parallel_for(sycl::range<2>(blocks, threads), [=](sycl::id<2> id)
        {
            auto linear_id = id[1] + blocks * id[0];
            auto tid = id[1];
            sdata[tid] = d_input[tid];
            for (unsigned int s = blocks / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                {
                    sdata[tid] += sdata [tid + s];
                }
            }           
            if (tid == 0)
            {
                d_result[id[0]] = sdata[0];                
            }
        });
    }).wait();
    int device_result;
    queue.memcpy(&device_result, d_result, sizeof(int)).wait();
    std::cout << "Host sum: " << host_result << '\n';
    std::cout << "Device sum: " << device_result << '\n';

    sycl::free(d_input, queue);
    sycl::free(sdata, queue);
    sycl::free(d_result, queue);
}