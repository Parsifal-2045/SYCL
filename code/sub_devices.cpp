#include <iostream>
#include <string>
#include <CL/sycl.hpp>

sycl::device chooseDevice(int id) {
  const std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::all);
  auto const& device = devices[id % devices.size()];
  if (device.is_gpu() and device.get_backend() == sycl::backend::ext_oneapi_level_zero) {
    try {
      std::vector<sycl::device> subDevices =
          device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
              sycl::info::partition_affinity_domain::next_partitionable);
      auto const& subDevice = subDevices[id % subDevices.size()];
      std::cerr << "Stream " << id << " offload to tile " << id % subDevices.size() << " on device "
                << id % devices.size() << std::endl;
      return subDevice;
    } catch (sycl::exception const& e) {
      std::cerr << "This GPU does not support splitting in multiple sub devices" << std::endl;
      std::cerr << "Stream " << id << " offload to " << device.get_info<cl::sycl::info::device::name>()
                << " on backend " << device.get_backend() << std::endl;
      return device;
    }
  } else {
    std::cerr << "Stream " << id << " offload to " << device.get_info<cl::sycl::info::device::name>() << " on backend "
              << device.get_backend() << std::endl;
    return device;
  }
}

int main() {
  for (int i = 0; i != 10; i++) {
    auto dev = chooseDevice(i);
    std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
  }
}