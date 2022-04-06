#include <CL/sycl.hpp>
#include "USM_selector.hpp"

int main()
{
    auto queue = sycl::queue{USM_device_selector{}};
    std::cout << "Chosen device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';
}