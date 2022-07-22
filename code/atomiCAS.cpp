#include <atomic>
#include <CL/sycl.hpp>

int main()
{
    //old = atomicCAS(address, compare, val);

    std::atomic<unsigned long long> address;
    unsigned long long old;
    unsigned long long val;
    address.compare_exchange_strong(old, val, std::memory_order::relaxed);

    unsigned long long* coso;    
    unsigned long long* old2;
    unsigned long long* val2;
    sycl::ext::oneapi::atomic_ref<unsigned long long*, sycl::ext::oneapi::memory_order::relaxed, sycl::ext::oneapi::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> pippo(coso);
    pippo.compare_exchange_strong(old2, val2, sycl::ext::oneapi::detail::memory_order::relaxed, sycl::ext::oneapi::detail::memory_scope::device);
}