#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

constexpr int numThreadsPerBlock = 1024;

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int *input, int *output)
{
    __shared__ int sdata[numThreadsPerBlock];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = input[i];

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
    int numInputElements = 2048;
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
    
    const dim3 blockSize(numThreadsPerBlock, 1, 1);
    const dim3 gridSize(numOutputElements, 1, 1);

    int *d_input;
    cudaMalloc(&d_input, numInputElements * sizeof(int));
    cudaMemcpy(d_input, h_input.data(), numInputElements * sizeof(int), cudaMemcpyHostToDevice);

    int *d_output;
    cudaMalloc(&d_output, numOutputElements * sizeof(int));
    
    block_sum<<<gridSize, blockSize>>>(d_input, d_output);

    int device_result[numOutputElements];
    cudaMemcpy(&device_result, d_output, numOutputElements * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 1; i != numOutputElements; i++)
    {
        device_result[0] += device_result[i];
    }

    std::cout << "Host sum: " << host_result << std::endl;
    std::cout << "Device sum: " << device_result[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}