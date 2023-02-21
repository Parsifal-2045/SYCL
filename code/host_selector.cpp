#include <CL/sycl.hpp>

int main()
{
  sycl::device dev;
  std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
  if (dev.is_host())
  {
    std::cout << "dev is host!" << std::endl;
  }
}
