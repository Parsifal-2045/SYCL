#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "../../Intel_gpu_selector.hpp"

constexpr int numThreadsPerBlock = 256;

void block_sum(const int* input, int* output, sycl::nd_item<3> item, int* sdata)
{
    unsigned int tid = item.get_local_id(0); // threadIdx.x
    unsigned int i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0); // blockIdx.x * blockDim.x + threadIdx.x
    sdata[tid] = input[i];

    item.barrier();

    for (unsigned int s = 1; s < item.get_local_range().get(0); s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        item.barrier();
    }

    if (tid == 0)
    {
        output[item.get_group(0)] = sdata[0];
    }

    item.barrier();
}

int main()
{
    auto queue = sycl::queue(sycl::gpu_selector{});
    int numInputElements = 2135;
    int numOutputElements;
    numOutputElements = numInputElements / (numThreadsPerBlock / 2);
    if (numInputElements % (numThreadsPerBlock / 2))
    {
        numOutputElements++;
    }
    std::vector<int> h_input(numInputElements);    
    for (int i = 0; i != numInputElements; i++)
    {
        h_input[i] = 1;        
    }

    const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);

    const sycl::range<3> blockSize(numThreadsPerBlock, 1, 1);
    const sycl::range<3> gridSize(numOutputElements, 1, 1);

    int* d_input;
    d_input = sycl::malloc_device<int>(numInputElements, queue);
    queue.memcpy(d_input, h_input.data(), numInputElements * sizeof(int)).wait();

    std::vector<int> zero(numOutputElements);

    int* d_output;
    d_output = sycl::malloc_device<int>(numOutputElements, queue);   
    queue.memcpy(d_output, zero.data(), numOutputElements * sizeof(int)).wait();

    std::cout << blockSize.get(0) << '\n';
    std::cout << gridSize.get(0) << '\n';

    queue.submit([&](sycl::handler &cgh) 
    {
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range<1>(numThreadsPerBlock), cgh);
        cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [=](sycl::nd_item<3> item)
        {
            block_sum(d_input, d_output, item, sdata.get_pointer());
        });
    }).wait();

    int device_result[numOutputElements];
    queue.memcpy(&device_result, d_output, numOutputElements * sizeof(int)).wait();
    for (int i = 1; i != numOutputElements; i++)
    {
        device_result[0] += device_result[i];
    }

    std::cout << "Host sum: " << host_result << std::endl;
    std::cout << "Device sum: " << device_result[0] << std::endl;

    sycl::free(d_input, queue);
    sycl::free(d_output, queue);
    return 0;
}