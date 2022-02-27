#include <CL/sycl.hpp>

int main()
{
    auto queue = sycl::queue{};
    int a = 18;
    int b = 24;
    int r;
    auto buf_a = sycl::buffer<int, 1>(&a, sycl::range<1>(sizeof(int)));
    auto buf_b = sycl::buffer<int, 1>(&b, sycl::range<1>(sizeof(int)));
    auto buf_r = sycl::buffer<int, 1>(&r, sycl::range<1>(sizeof(int)));

    queue.submit([&](sycl::handler &cgh)
    {
        auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
        auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
        auto out = buf_r.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task([=]
        {
            out[0] = in_a[0] + in_b[0];
        });
    }).wait(); // without wait() result is incorrect, gets printed before the calculation occurs
    std::cout << r << '\n';
}