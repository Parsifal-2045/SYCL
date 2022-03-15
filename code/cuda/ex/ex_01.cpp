#include <CL/sycl.hpp>

int main()
{
    int dimA = 8;
    std::vector<float> h_a(dimA);
    for(int i = 0; i != dimA; i++)
    {
        h_a[i] = i;
    }
    size_t memSize = dimA * sizeof(float);
    auto queue = sycl::queue{sycl::default_selector{}};
    auto d_a = sycl::malloc_device<float>(dimA, queue);
    auto d_b = sycl::malloc_device<float>(dimA, queue);
    queue.memcpy(d_a, h_a.data(), memSize).wait();
    queue.memcpy(d_b, d_a, memSize).wait();
    std::fill(h_a.begin(), h_a.end(), 0);
    queue.memcpy(h_a.data(), d_b, memSize).wait();
    for(int i = 0; i != dimA; i++)
    {
        assert(h_a[i] == static_cast<float>(i));
    }
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    std::cout << "Correct!" << '\n';
}