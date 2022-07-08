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

}

int main()
{
    auto queue = sycl::queue(sycl::gpu_selector{});
    int numInputElements = 2048;
    int numOutputElements;
    numOutputElements = numInputElements / (numThreadsPerBlock / 2);
    if (numInputElements % (numThreadsPerBlock / 2))
    {
        numOutputElements++;
    }
    std::vector<int> h_input;

    for (int i = 0; i != numInputElements; i++)
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dis(0, 100);
        auto value = dis(gen);
        h_input.push_back(value);
    }

    const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);

    const sycl::range<3> blockSize(numThreadsPerBlock, 1, 1);
    const sycl::range<3> gridSize(numOutputElements, 1, 1);

    int* d_input = sycl::malloc_device<int>(numInputElements, queue);
    queue.memcpy(d_input, h_input.data(), numInputElements * sizeof(int)).wait();

    int* d_output = sycl::malloc_device<int>(numOutputElements, queue);   
    
    std::cout << blockSize.get(0) << '\n';
    std::cout << gridSize.get(0) << '\n';

    queue.submit([&](sycl::handler& cgh) 
    {
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range<1>(numThreadsPerBlock), cgh);
        cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [=](sycl::nd_item<3> item)
        {
            block_sum(d_input, d_output, item, sdata.get_pointer());
        });
    }).wait();

    queue.submit([&](sycl::handler &cgh)
    {
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range<1>(numThreadsPerBlock), cgh);
        cgh.parallel_for(sycl::nd_range<3>(blockSize, blockSize), [=](sycl::nd_item<3> item)
        {
            block_sum(d_output, d_output, item, sdata.get_pointer());
        });
    });
    
    int device_result;
    queue.memcpy(&device_result, d_output, sizeof(int)).wait();

    std::cout << "Host sum: " << host_result << std::endl;
    std::cout << "Device sum: " << device_result << std::endl;
    
    sycl::free(d_input, queue);
    sycl::free(d_output, queue);
}