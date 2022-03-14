#include <CL/sycl.hpp>
#include <cassert>
#include "USM_selector.hpp"

int square(int x)
{
    auto queue = sycl::queue{USM_device_selector{}};
    auto device_ptr = sycl::malloc_device<int>(1, queue);
    queue.memcpy(device_ptr, &x, sizeof(int)).wait();
    queue.submit([&](sycl::handler &cgh)
    {
        cgh.single_task([=]()
        {
            *device_ptr = (*device_ptr) * (*device_ptr);
        });
    }).wait();
    queue.memcpy(&x, device_ptr, sizeof(int)).wait();
    return x;
    sycl::free(device_ptr, queue);
}

int main()
{
    assert(square(2) == 4);
    assert(square(-2) == 4);
    assert(square(0) == 0);
    assert(square(10) == 100);
    std::cout << "Correct!" << '\n';
}