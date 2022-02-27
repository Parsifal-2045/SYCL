#include <iostream>

__global__ void add (const int *a, const int *b, int *c)
{
    *c = *a + *b; // a, b, c must be on the GPU memory -> we need to allocate memory on the GPU
}

int main()
{
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);
    // Allocate space in Device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // Input values
    a = 2;
    b = 7;
    // Copy inputs to Device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice); // destination, origin, how much memory
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, size, cudaMemcpyHostToDevice);
    // Launch add() on the Device
    add<<<1,1>>>(d_a, d_b, d_c);
    // Result is in d_c, we need to copy it back
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}