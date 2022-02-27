#include <CL/sycl.hpp>

int main()
{
  auto queue = sycl::queue{};
  queue.submit([&] (sycl::handler &cgh)
  {
    auto os = sycl::stream(1024, 128, cgh);

    cgh.single_task([=] () 
    {
      os << "Hello world!" << '\n';
    });
  }).wait();
}