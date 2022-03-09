#include <CL/sycl.hpp>
#include <cassert>


int main()
{
    constexpr size_t data_size = 1024;
    constexpr size_t workgroup_size = 128;
    std::vector<float> a(data_size);
    std::vector<float> b(data_size);
    std::vector<float> c(data_size);
    
    for(int i = 0; i != data_size; i++)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
        c[i] = 0.0f;
    }

    auto buf_a = sycl::buffer<float, 1>(a.data(), sycl::range<1>(data_size));
    auto buf_b = sycl::buffer<float, 1>(b.data(), sycl::range<1>(data_size));
    auto buf_c = sycl::buffer<float, 1>(c.data(), sycl::range<1>(data_size));

    auto queue = sycl::queue{sycl::default_selector{}};

    queue.submit([&](sycl::handler &cgh)
    {
        auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
        auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
        auto out = buf_c.get_access<sycl::access::mode::write>(cgh);
        auto range = sycl::nd_range(sycl::range<1>(data_size), sycl::range<1>(workgroup_size));

        cgh.parallel_for(range, [=](sycl::nd_item<1> i)
        {
            auto global_id = i.get_global_id();
            
            out[global_id] = in_a[global_id] + in_b[global_id];

        });
    }).wait();

    for (int i = 0; i != data_size; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
    std::cout << "Correct!" << '\n';
}