#include <CL/sycl.hpp>
#include <iostream>
#include <optional>

class Test
{
public:
    Test() = default;

    ~Test()
    {
        sycl::free(a, q_);
        sycl::free(b, q_);
    }

    sycl::queue GetQueue()
    {
        return q_;
    }

    void SetQueue(sycl::queue queue)
    {
        q_ = queue;
    }

    int *a = nullptr;
    int *b = nullptr;

private:
    sycl::queue q_;
};

int main()
{
    int n = 10;
    sycl::queue queue = sycl::queue{sycl::cpu_selector{}, sycl::property::queue::in_order()};
    Test test;
    std::cout << "Inner queue (before setting): " << test.GetQueue().get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "Outer queue : " << queue.get_device().get_info<sycl::info::device::name>() << '\n';
    test.SetQueue(queue);
    std::cout << "Inner queue: " << test.GetQueue().get_device().get_info<sycl::info::device::name>() << '\n';
    std::vector<int> vec(n, 1);
    test.a = sycl::malloc_device<int>(n, queue);
    test.b = sycl::malloc_device<int>(n, queue);
    queue.memcpy(test.a, vec.data(), n * sizeof(int));
    queue.memcpy(test.b, vec.data(), n * sizeof(int));
    
    queue.submit([&](sycl::handler& h)
    {
        auto a = test.a;
        auto b = test.b;
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
        {
            a[i] = 2;
            b[i] = 3;
        });
    });

    std::vector<int> a(n);
    std::vector<int> b(n);

    queue.memcpy(a.data(), test.a, n * sizeof(int));
    queue.memcpy(b.data(), test.b, n * sizeof(int)).wait();

    for (int i = 0; i != n; i++)
    {
        std::cout << a[i] << ", ";
    }
    std::cout << '\n';
    for (int i = 0; i != n; i++)
    {
        std::cout << b[i] << ", ";
    }
    std::cout << '\n';
}