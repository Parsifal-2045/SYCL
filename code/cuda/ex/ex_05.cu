#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

static const int num_elements = 1 << 18;
static const int block_size = 1024;
static const int grid_size = 32;

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int *input, int *per_block_results, const size_t n)
{
    int idx = threadIdx.x;
    int gidx = idx + blockIdx.x * block_size;
    const int gridSize = block_size * gridDim.x;
    int sum = 0;
    for (int i = gidx; i < num_elements; i += gridSize)
    {
        sum += input[i];
    }
    __shared__ int sdata[block_size];
    sdata[idx] = sum;
    __syncthreads();
    for (int size = grid_size / 2; size > 0; size /= 2)
    {
        if (idx < size)
        {
            sdata[idx] += sdata[idx + size];
            __syncthreads();
        }
    }
    if (idx == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
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

    // //Part 1 of 6: move input to device memory
    int *d_input;
    size_t memSize = num_elements * sizeof(int);
    cudaMalloc(&d_input, memSize);
    cudaMemcpy(d_input, h_input.data(), memSize, cudaMemcpyHostToDevice);

    // // Part 1 of 6: allocate the partial sums: How much space does it need?
    int *d_partial_sums_and_total;
    size_t partial_mem = grid_size * sizeof(int) ;
    cudaMalloc(&d_partial_sums_and_total, partial_mem);

    // // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How
    // much shared memory does it need?
    block_sum<<<grid_size, block_size>>>(d_input, d_partial_sums_and_total, num_elements);
    // d_partial_sums_and_total holds the partial result
    block_sum<<<1, block_size>>>(d_partial_sums_and_total, d_partial_sums_and_total, num_elements);
    // d_partial_sums_and_total[0] holds the final result
    cudaDeviceSynchronize();
    int device_result;
    cudaMemcpy(&device_result, d_partial_sums_and_total, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Device sum: " << device_result << std::endl;

    // // Part 1 of 6: deallocate device memory
    cudaFree(d_input);
    cudaFree(d_partial_sums_and_total);
    return 0;
}