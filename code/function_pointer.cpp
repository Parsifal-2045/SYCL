#include <iostream>
#include <CL/sycl.hpp>

// https://codeplay.com/portal/blogs/2019/09/24/alternatives-to-cpp-function-pointers-in-sycl-using-function-objects.html
// Doesn't compile with dpcpp: SYCL kernel cannot call through a function pointer
template <typename T>
auto add(T left, T right) -> T { return left + right; }
template <typename T>
auto subtract(T left, T right) -> T { return left - right; }
template <typename T>
auto multiply(T left, T right) -> T { return left * right; }
template <typename T>
auto divide(T left, T right) -> T { return left / right; }

template <typename T>
auto calculate(T left, T right, int (*binary_op)(T, T)) -> T
{
    return (*binary_op)(left, right);
}
// usage: calculate(6, 3.5, add);

int main()
{
    auto q = sycl::queue(sycl::default_selector{}, sycl::property::queue::in_order());
    int n = 10;
    std::vector<int> vec(n, 1);
    auto d_a = sycl::malloc_device<int>(n, q);
    auto d_b = sycl::malloc_device<int>(n, q);

    q.memcpy(d_a, vec.data(), n * sizeof(int));
    q.memcpy(d_b, vec.data(), n * sizeof(int)).wait();

    auto d_r = sycl::malloc_device<int>(n, q);

    q.submit([&](sycl::handler &h)
             { h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
                              { d_r[i] = calculate(*d_a, *d_b, add); }); });

    // std::cout << calculate(3, 5, add) << '\n';
}