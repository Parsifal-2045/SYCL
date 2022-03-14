#include <CL/sycl.hpp>

class USM_device_selector : public sycl::device_selector
{
    int operator()(const sycl::device &dev) const override
    {
        if (dev.has(sycl::aspect::usm_device_allocations))
        {
            return 1;
        }
        return -1;
    }
};

int main()
{
    // Task: create a queue to a device which supports USM allocations
    // Remember to check for exceptions
    auto usmQueue = sycl::queue{USM_device_selector{}};
    std::cout << "Chosen device: " << usmQueue.get_device().get_info<sycl::info::device::name>() << '\n';
}

// returns Intel FPGA Emulation Device