#include <CL/sycl.hpp>
#include <cassert>

int main()
{
    constexpr size_t N = 1024;
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    
    for (int i = 0; i != N; i++)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
        c[i] = 0.0f;
    }

    auto buf_a = sycl::buffer<float, 1>(a.data(), sycl::range<1>(N * sizeof(float)));
    auto buf_b = sycl::buffer<float, 1>(b.data(), sycl::range<1>(N * sizeof(float)));
    auto buf_c = sycl::buffer<float, 1>(c.data(), sycl::range<1>(N * sizeof(float)));
    auto queue = sycl::queue{sycl::default_selector{}};

    queue.submit([&](sycl::handler &cgh)
    {
        auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
        auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
        auto out = buf_c.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            out[idx] = in_a[idx] + in_b[idx];
        });

    }).wait();

    for (int i = 0; i != N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }

    std::cout << "Correct!" << '\n';

}