#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

static const int num_elements = 2 << 20;
static const int threads = 1024;
static const int blocks = num_elements / threads;

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int *input, int *result)
{
    __shared__ int sdata[threads];
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = input[id];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        result[blockIdx.x] = sdata[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
    std::vector<int> h_input(num_elements);
    /*
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(-10, 10);
        for (auto &elt : h_input)
        {
            elt = distrib(gen);
        }
    */
    for (int i = 0; i != num_elements; i++)
    {
        h_input[i] = i;
    }
    const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);
    std::cerr << "Host sum: " << host_result << std::endl;

    int *d_input;
    size_t memSize = num_elements * sizeof(int);
    cudaMalloc(&d_input, memSize);
    cudaMemcpy(d_input, h_input.data(), memSize, cudaMemcpyHostToDevice);

    int *d_result;
    cudaMalloc(&d_result, sizeof(int));
    block_sum<<<blocks, threads>>>(d_input, d_result);
    int device_result;
    cudaMemcpy(&device_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Device sum: " << device_result << std::endl;

    // // Part 1 of 6: deallocate device memory
    cudaFree(d_input);
    cudaFree(d_result);
    return 0;
}