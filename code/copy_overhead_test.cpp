#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

int main()
{
  sycl::queue q;

  try
  {
    q = sycl::queue(sycl::cpu_selector{}, sycl::property::queue::in_order());
  }
  catch (sycl::exception const &e)
  {
    std::cout << "Cannot select a GPU\n" << e.what() << "\n";
    std::cout << "Using a CPU device\n";
    q = sycl::queue(sycl::cpu_selector{}, sycl::property::queue::in_order());
  }

  std::cout << "Using " << q.get_device().get_info<sycl::info::device::name>() << '\n';

  int n = 2e7;
  std::vector<float> vec(n, 1);
  auto d_a = sycl::malloc_device<float>(n, q);
  auto d_b = sycl::malloc_device<float>(n, q);
  q.memcpy(d_a, vec.data(), n * sizeof(float));
  q.memcpy(d_b, vec.data(), n * sizeof(float)).wait();
  auto d_r = sycl::malloc_device<float>(n, q);
  int rep = 10;
  for (int j = 0; j != rep; j++)
  {
    std::cout << "Iteration " << j << '\n';
    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler& h)
    {
      h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
      {
        d_r[i] = d_a[i] + d_b[i];
      });
    });
  
    std::vector<float> res(n);
    q.memcpy(res.data(), d_r, n * sizeof(float)).wait();
    for (int i = 0; i != n; i++)
    {
      assert(res[i] == 2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() * 1000 << "ms\n";
  }

}