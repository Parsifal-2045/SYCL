#include <CL/sycl.hpp>
#include <iostream>

template <typename T,
              sycl::access::address_space addrSpace = cl::sycl::access::address_space::global_space,
              sycl::memory_scope Scope = cl::sycl::memory_scope::device,
              sycl::memory_order memOrder = cl::sycl::memory_order::relaxed>
    inline T atomic_fetch_add(T* addr, T operand){

      auto atm = sycl::atomic_ref<T, memOrder, Scope, addrSpace>(addr[0]);

      return atm.fetch_add(operand);
    }

void testAdd(int *a, int *b)
{
	for (int i = 0; i < 100 ; i++)
	{
        	atomic_fetch_add<int,sycl::access::address_space::global_space,sycl::memory_scope::device>(a, i);
        	atomic_fetch_add<int,sycl::access::address_space::global_space,sycl::memory_scope::work_group>(a, i);
        	atomic_fetch_add<int,sycl::access::address_space::local_space,sycl::memory_scope::device>(b, i);
        	atomic_fetch_add<int,sycl::access::address_space::local_space,sycl::memory_scope::work_group>(b, i);
	}
}


int main() {
  sycl::queue queue = sycl::queue(sycl::cpu_selector());

  int *d_data;
  d_data = sycl::malloc_device<int>(100, queue);
  queue.memset(d_data, 0, 100*sizeof(int)).wait();
  queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(10), sycl::range<1>(10)),
        [=](sycl::nd_item<1> item) {
	auto n0buff = sycl::ext::oneapi::group_local_memory_for_overwrite<int[100]>(item.get_group());
	int* shared_b = (int*)n0buff.get();
            testAdd(d_data, shared_b);
        }).wait();
  printf("Success\n");
  return 0;
}
