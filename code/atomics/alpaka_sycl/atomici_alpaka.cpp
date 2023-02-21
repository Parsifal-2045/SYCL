#include <CL/sycl.hpp>
#include <iostream>

namespace hierarchy
{
    struct Blocks
    {
    };

    struct Threads
    {
    };
} // namespace hierarchy

template <typename THierarchy>
struct SyclMemoryScope
{
};

template <>
struct SyclMemoryScope<hierarchy::Blocks>
{
    static constexpr auto value = sycl::memory_scope::device;
};

template <>
struct SyclMemoryScope<hierarchy::Threads>
{
    static constexpr auto value = sycl::memory_scope::work_group;
};

template <typename T>
inline auto get_global_ptr(T *const addr)
{
    return sycl::make_ptr<T, sycl::access::address_space::global_space>(addr);
}

template <typename T>
inline auto get_local_ptr(T *const addr)
{
    return sycl::make_ptr<T, sycl::access::address_space::local_space>(addr);
}

template <typename T, typename THierarchy>
using global_ref = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    SyclMemoryScope<THierarchy>::value,
    sycl::access::address_space::global_space>;

template <typename T, typename THierarchy>
using local_ref = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    SyclMemoryScope<THierarchy>::value,
    sycl::access::address_space::local_space>;

template <typename THierarchy, typename T, typename TOp>
inline auto callAtomicOp(T *const addr, TOp &&op)
{
    if (auto ptr = get_global_ptr(addr); ptr != nullptr)
    {
        auto ref = global_ref<T, THierarchy>{*addr};
        return op(ref);
    }
    else
    {
        auto ref = local_ref<T, THierarchy>{*addr};
        return op(ref);
    }
}

template <typename T, typename THierarchy>
auto atomicAdd(T *const addr, T const &value) -> T
{
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

    return callAtomicOp<THierarchy>(
        addr,
        [&value](auto &ref)
        { return ref.fetch_add(value); });
};

void testAdd(int *a, int *b)
{
    for (int i = 0; i < 100; i++)
    {
        atomicAdd<int, hierarchy::Blocks>(&a[i], i);
        atomicAdd<int, hierarchy::Threads>(&a[i], i);
        atomicAdd<int, hierarchy::Blocks>(&b[i], i);
        atomicAdd<int, hierarchy::Threads>(&b[i], i);
    }
}

int main()
{
    sycl::queue queue = sycl::queue(sycl::gpu_selector());

    int *d_data, *h_data;
    h_data = (int *)malloc(100 * sizeof(int));
    d_data = sycl::malloc_device<int>(100, queue);
    queue.memset(d_data, 0, 100 * sizeof(int)).wait();
    queue.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(1, 1, 10), sycl::range<3>(1, 1, 10)),
             [=](sycl::nd_item<3> item)
             {
                 auto bbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int[100]>(item.get_group());
                 int *b = (int *)bbuff.get();
                 testAdd(d_data, b);
             })
        .wait();
    printf("Success\n");
    return 0;
}
