#include <CL/sycl.hpp>
#include <memory>
#include <iostream>
#include <numeric>

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> const &v)
{
    os << '(';
    for (int i = 0; i != v.size() - 1; i++)
    {
        os << v[i] << ", ";
    }
    os << v[v.size() - 1] << ')';
    return os;
}

class DeviceDeleter
{
public:
    DeviceDeleter() = default;
    DeviceDeleter(sycl::queue stream) : stream_{stream} {}

    void operator()(void *ptr)
    {
        if (stream_)
        {
            sycl::free(ptr, *stream_);
            std::cout << "Freed device memory\n";
        }
    }

private:
    std::optional<sycl::queue> stream_;
};

template <typename T>
typename std::unique_ptr<T, DeviceDeleter> make_device_unique(size_t n, sycl::queue stream)
{
    void *mem = sycl::malloc_device(n * sizeof(T), stream);
    return std::unique_ptr<T, DeviceDeleter>{reinterpret_cast<T *>(mem), DeviceDeleter{stream}};
}

void device_normal()
{
    int n = 10;
    auto queue = sycl::queue(sycl::gpu_selector());
    auto d_a = sycl::malloc_device<int>(n, queue);
    std::vector<int> vec(n);
    std::vector<int> a(n);
    std::iota(vec.begin(), vec.end(), 0);
    queue.memcpy(d_a, vec.data(), n * sizeof(int)).wait();
    queue.memcpy(a.data(), d_a, n * sizeof(int)).wait();   
    std::cout << "Before -> " << a << '\n';
    queue.submit([&](sycl::handler& h)
    {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
        {
           d_a[i] = 42;
        });
    }).wait();
    queue.memcpy(a.data(), d_a, n * sizeof(int)).wait();
    std::cout << "After -> " << a << '\n';
    sycl::free(d_a, queue);
}

void device_unique()
{
    int n = 10;
    auto queue = sycl::queue(sycl::gpu_selector());
    auto d_a = make_device_unique<int>(n, queue);
    std::vector<int> vec(n);
    std::vector<int> a(n);
    std::iota(vec.begin(), vec.end(), 0);
    queue.memcpy(d_a.get(), vec.data(), n * sizeof(int)).wait();
    queue.memcpy(a.data(), d_a.get(), n * sizeof(int)).wait();   
    std::cout << "Before -> " << a << '\n';
    queue.submit([&](sycl::handler& h)
    {
        auto d_a_kernel = d_a.get();
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
        {
           d_a_kernel[i] = 42;
        });
    }).wait();
    queue.memcpy(a.data(), d_a.get(), n * sizeof(int)).wait();
    std::cout << "After -> " << a << '\n';

}



int main()
{
    #ifdef Normal
    std::cout << "Normal allocation on device:\n";
    device_normal();
    #endif 
    #ifdef Unique
    std::cout << "Allocating unique pointers on device:\n";
    device_unique();
    #endif
}