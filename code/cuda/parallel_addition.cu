//Parallel blocks

#include <iostream>

//add<<N,1>> launches N blocks of 1 thread running add() in parallel

__global__ void add(const int *a, const int*b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x] // each blockId will run one iteration in parallel
    // block 0 : c[0] = a[0] + b[0] ... block N : c[N] = a[N] + b[N]
}

int main()
{
    int *d_a, *d_b, *d_c;
    int N = 512;
    std::vector<int> a, b, c;
    a.resize(N);
    b.resize(N);
    c.resize(N);
    int size = N * sizeof(int);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // Setup input values
    random_fill(a);
    random_fill(b);
    // Copy inputs to device
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b,d_c);
    // Copy result to Host
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

// Parallel Threads

#include <iostream>

//add<<1,N>> launches N threads in 1 block

__global__ void add(const int *a, const int*b, int *c)
{
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x] // each blockId will run one iteration in parallel
    // block 0 : c[0] = a[0] + b[0] ... block N : c[N] = a[N] + b[N]
}

int main()
{
    int *d_a, *d_b, *d_c;
    int N = 512;
    std::vector<int> a, b, c;
    a.resize(N);
    b.resize(N);
    c.resize(N);
    int size = N * sizeof(int);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // Setup input values
    random_fill(a);
    random_fill(b);
    // Copy inputs to device
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b,d_c);
    // Copy result to Host
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

// Combine blocks and threads: each block takes a chunck of memory and contains some number of thread
// First choose the number of thread per blocks, then scale the number of blocks to solve the problem
// This allows to index each thread as threadId + blockId * blockDim (with the desired blockDim)

__global__ void add(const int *a, const int *b, int *c)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

int main()
{
    constexpr int N = 2048*2048;
    int threads_per_block = 512;
    std::vector<int> a, b, c;
    a.resize(N);
    b.resize(N);
    c.resize(N);
    int size = N * sizeof(int);
    // Input
    random_fill(a);
    random_fill(b);
    // Copy to Device memory
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    // Launch add()
    add<<<N/threads_per_block, thread_per_block>>>(d_a, d_b, d_c);
    // Copy result to Host
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    //Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}