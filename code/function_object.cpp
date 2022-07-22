#include <iostream>
#include <CL/sycl.hpp>

// https://codeplay.com/portal/blogs/2019/09/24/alternatives-to-cpp-function-pointers-in-sycl-using-function-objects.html
// Correct version of function_pointer for SYCL
template <typename T>
class add
{
public:
    T operator()(T left, T right)
    {
        return left + right;
    }
};

template <typename T>
class sub
{
public:
    T operator()(T left, T right)
    {
        return left - right;
    }
};

template <typename T>
class multiply
{
public:
    T operator()(T left, T right)
    {
        return left * right;
    }
};

template <typename T>
class divide
{
public:
    T operator()(T left, T right)
    {
        return left / right;
    }
};

template <typename T, class Operation>
T calculate(T left, T right, Operation binary_op)
{
    return binary_op(left, right);
}

int main()
{
    auto q = sycl::queue(sycl::default_selector{});
    int n = 10;
    std::vector<int> vec(n, 2);
    auto d_a = sycl::malloc_device<int>(n, q);
    auto d_b = sycl::malloc_device<int>(n, q);

    q.memcpy(d_a, vec.data(), n * sizeof(int));
    q.memcpy(d_b, vec.data(), n * sizeof(int)).wait();

    auto d_r = sycl::malloc_device<int>(n, q);

    q.submit([&](sycl::handler &h)
             { h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i)
                              { d_r[i] = calculate(d_a[i], d_b[i], add<int>{}); }); });
    q.wait();
    std::vector<int> res(n);
    q.memcpy(res.data(), d_r, n * sizeof(int)).wait();
    for (int i = 0; i != n; i++)
    {
        std::cout << res[i] << ", ";
    }
    std::cout << '\n';

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_r, q);
}