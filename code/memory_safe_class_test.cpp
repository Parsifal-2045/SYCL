#include <CL/sycl.hpp>
#include <iostream>
#include <optional>

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


class Test
{
public:
    Test() = default;

    ~Test() {};

    std::unique_ptr<int, DeviceDeleter> a;
    std::unique_ptr<int, DeviceDeleter> b;

};

int main()
{
    int n = 10;
    sycl::queue queue = sycl::queue{sycl::gpu_selector{}, sycl::property::queue::in_order()};
    Test test;
    std::vector<int> vec(n, 1);
    test.a = make_device_unique<int>(n, queue);
    test.b = make_device_unique<int>(n, queue);
    queue.memcpy(test.a.get(), vec.data(), n * sizeof(int));
    queue.memcpy(test.b.get(), vec.data(), n * sizeof(int)).wait();
    
    queue.submit([&](sycl::handler& h)
    {
        auto a = test.a.get();
        auto b = test.b.get();
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
        {
            a[i] = 2;
            b[i] = 3;
        });
    });

    std::vector<int> a(n);
    std::vector<int> b(n);

    queue.memcpy(a.data(), test.a.get(), n * sizeof(int));
    queue.memcpy(b.data(), test.b.get(), n * sizeof(int)).wait();

    std::cout << a << '\n';
    std::cout << b << '\n';
}