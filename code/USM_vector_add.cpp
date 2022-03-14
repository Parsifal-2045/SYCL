#include <CL/sycl.hpp>
#include <cassert>
#include "USM_selector.hpp"

int main()
{
    constexpr size_t N = 1024;
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> r(N);
    for (int i = 0; i != N; i++)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
        r[i] = 0.0f;
    }

    auto queue = sycl::queue{USM_device_selector{}};
    auto d_a = sycl::malloc_device<float>(N, queue);
    auto d_b = sycl::malloc_device<float>(N, queue);
    auto d_r = sycl::malloc_device<float>(N, queue);
    queue.memcpy(d_a, a.data(), N * sizeof(float)).wait();
    queue.memcpy(d_b, b.data(), N * sizeof(float)).wait();

    queue.submit([&](sycl::handler &cgh)
    {
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            auto global_id = idx[0];
            d_r[global_id] = d_a[global_id] + d_b[global_id];
        });
    }).wait();

    queue.memcpy(r.data(), d_r, N * sizeof(float)).wait();
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_r, queue);
    for (int i = 0; i != N; i++)
    {
        assert(a[i] == static_cast<float>(i));
        assert(b[i] == static_cast<float>(i));
        assert(r[i] == a[i] + b[i]);
    }
    std::cout << "Correct!" << '\n';
}
