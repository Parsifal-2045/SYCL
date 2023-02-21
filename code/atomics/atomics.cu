#include <iostream>

__global__ void testAdd(unsigned int *a)
{
  for (int i = 0; i < 100; i++)
  {
    atomicAdd(&a[i], 1);
  }
}

int main()
{

  unsigned int *d_data, *h_data;
  h_data = (unsigned int *)malloc(100 * sizeof(unsigned int));
  cudaMalloc((void **)&d_data, 100 * sizeof(unsigned int));
  cudaMemset(d_data, 0, 100 * sizeof(unsigned int));
  testAdd<<<1, 10>>>(d_data);
  cudaMemcpy(h_data, d_data, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 100; i++)
    if (h_data[i] != 10)
    {
      printf("mismatch at %d, was %d, should be %f\n", i, h_data[i], 10.0f);
      return 1;
    }
  printf("Success\n");
  return 0;
}