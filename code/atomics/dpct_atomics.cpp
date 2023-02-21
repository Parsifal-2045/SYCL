#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

void testAdd(uint32_t *a)
{
    for (int i = 0; i < 100; i++)
    {
        dpct::atomic_fetch_add<uint32_t,
                               sycl::access::address_space::generic_space>(
            &a[i], (uint32_t)1);
    }
}

int main()
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    std::cout << dev_ct1.get_info<sycl::info::device::name>() << " on backend " << dev_ct1.get_backend() << std::endl;
    uint32_t *d_data, *h_data;
    h_data = (uint32_t *)malloc(100 * sizeof(uint32_t));
    d_data = sycl::malloc_device<uint32_t>(100, q_ct1);
    q_ct1.memset(d_data, 0, 100 * sizeof(uint32_t)).wait();
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 10), sycl::range<3>(1, 1, 10)),
        [=](sycl::nd_item<3> item_ct1)
        {
            testAdd(d_data);
        });
    q_ct1.wait();
    q_ct1.memcpy(h_data, d_data, 100 * sizeof(uint32_t)).wait();
    for (int i = 0; i < 100; i++)
        if (h_data[i] != 10)
        {
            printf("mismatch at %d, was %d, should be %f\n", i, h_data[i], 10.0f);
            return 1;
        }
    printf("Success\n");
    return 0;
}