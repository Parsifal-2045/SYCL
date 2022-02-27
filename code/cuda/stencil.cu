#include <iostream>

__global__ void stencil_1d(const int *in, int *out)
{
    __shared__ int temp[blockDim.x + 2 * radius];
    auto g_index = threadIdx.x + blockIdx-x * blockDim.x;
    auto s_index = threadId.x + radius
    // Read input elements into shared memory
    temp[s_index] = in[g_index];
    if (threadId.x < radius)
    {
        temp[s_index - radius] = in[g_index - radius];
        temp[s_index + BLOCK_SIZE] = in [g_index + BLOCK_SIZE];
    }
    //We need to wait for all the threads to have finished the indexing
    __syncthreads();
    // Apply the stencil
    int result = 0;
    for (int offset = -radius; offset <= radius; offset++)
    {
        result += temp[s_index + offset]; 
    }          
    out[g_index] = result;
}