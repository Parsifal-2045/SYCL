#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

int main()
{
    int dimx = 1024;
    int dimy = 1024;
    std::vector<int> h_a(dimx * dimy);
    int memSize = dimx * dimy * sizeof(int);
    auto queue = sycl::queue{sycl::default_selector{}};
    auto d_a = sycl::malloc_device<int>(dimx * dimy, queue);

    queue.memcpy(d_a, h_a.data(), memSize);
    queue.submit([&](sycl::handler &cgh)
    {
        cgh.parallel_for(sycl::range<2>(dimx, dimy), [=](sycl::id<2> id)
        {
            auto linear_id = id[0] * dimx + id[1];
            d_a[linear_id] = linear_id;
        });
    }).wait();
    queue.memcpy(h_a.data(), d_a, memSize).wait();
    sycl::free(d_a, queue);

    for(int i = 0; i != dimx; i++)
    {
        for(int j = 0; j != dimy; j++)
        {
            assert(h_a[i * dimx + j] == i * dimx + j);
        }
    }
    std::cout << "Correct!" << '\n';
}