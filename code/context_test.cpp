#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

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

int main()
{
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::context> contexts;

    for (auto &platform : platforms)
    {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        auto devices = platform.get_devices();
        contexts.emplace_back(sycl::context(devices));
    }
    std::cout << contexts.size() << std::endl;
    for (auto context : contexts)
    {
        auto devices = context.get_devices();
        for (auto dev : devices)
        {
            std::cout << dev.get_info<sycl::info::device::name>() << ", ";
        }
        std::cout << std::endl;
    }
}